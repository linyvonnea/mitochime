'''PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep \ 
  --mode cnn \
  --train-tsv data/processed/PAIR_train_seq_L150.tsv \
  --test-tsv  data/processed/PAIR_test_seq_L150.tsv \
  --L 150 --epochs 30 --batch 128 --lr 0.001 \
  --seed 42 \
  --select-best-by f1 \
  --weight-decay 1e-4 \
  --out-dir models/deep/cnn_final_L150_seed42 \
  --reports-dir reports/deep/cnn_final_L150_seed42 \
  --save-predictions'''
# src/mitochime/deep_learning/train_deep.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from .dl_data import ReadSeqDataset, SeqConfig
from .dl_cnn import CNN1D
from .dl_transformer import KmerTransformer


def set_seed(seed: int) -> None:
    """Best-effort reproducibility for CPU/GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_full(model, loader, device):
    """
    Full evaluation:
      returns avg_loss, y_true, y_pred, y_prob (P(class=1))
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    ys, preds, probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * y.size(0)

            p = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)
            pred = torch.argmax(logits, dim=1)

            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
            probs.append(p.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC needs both classes present
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=["clean", "chimeric"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
        "n_neg": int(len(y_true) - np.sum(y_true)),
        "classification_report": report_txt,
    }


def save_reports(
    out_dir: Path,
    mode: str,
    report: dict,
    test_loss: float,
    args_dict: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    read_ids: list[str] | None,
    save_predictions: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(report)
    payload["test_loss"] = float(test_loss)
    payload["mode"] = mode
    payload["args"] = args_dict

    (out_dir / f"{mode}_metrics.json").write_text(json.dumps(payload, indent=2))

    cm = np.array(report["confusion_matrix"])
    cm_path = out_dir / f"{mode}_confusion_matrix.tsv"
    with cm_path.open("w") as f:
        f.write(" \tpred_0\tpred_1\n")
        f.write(f"true_0\t{cm[0,0]}\t{cm[0,1]}\n")
        f.write(f"true_1\t{cm[1,0]}\t{cm[1,1]}\n")

    (out_dir / f"{mode}_classification_report.txt").write_text(
        report["classification_report"] + "\n"
    )

    if save_predictions:
        pred_path = out_dir / f"{mode}_predictions.tsv"
        with pred_path.open("w") as f:
            if read_ids is None:
                f.write("idx\ty_true\ty_pred\tprob_chimeric\n")
                for i in range(len(y_true)):
                    f.write(
                        f"{i}\t{int(y_true[i])}\t{int(y_pred[i])}\t{float(y_prob[i]):.6f}\n"
                    )
            else:
                f.write("read_id\ty_true\ty_pred\tprob_chimeric\n")
                for rid, yt, yp, pr in zip(read_ids, y_true, y_pred, y_prob):
                    f.write(f"{rid}\t{int(yt)}\t{int(yp)}\t{float(pr):.6f}\n")


def train_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cnn", "transformer"], required=True)

    ap.add_argument("--train-tsv", required=True)
    ap.add_argument("--test-tsv", required=True)

    ap.add_argument("--out-dir", default="models_dl")
    ap.add_argument("--reports-dir", default="reports/metrics_dl")

    # cnn params
    ap.add_argument("--L", type=int, default=300)
    ap.add_argument("--use-qual", action="store_true")  # unused in dl_data.py

    # transformer params
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--L-kmers", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)

    # training params
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # checkpoint selection
    ap.add_argument(
        "--select-best-by",
        choices=["acc", "f1", "auc"],
        default="acc",
        help="Metric used to pick best checkpoint on the eval set.",
    )

    # reporting extras
    ap.add_argument("--save-predictions", action="store_true")

    args = ap.parse_args()

    # reproducibility
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    rep_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SeqConfig(L=args.L, use_qual=args.use_qual, k=args.k, L_kmers=args.L_kmers)
    train_ds = ReadSeqDataset(args.train_tsv, mode=args.mode, cfg=cfg)
    test_ds = ReadSeqDataset(args.test_tsv, mode=args.mode, cfg=cfg)

    if args.mode == "cnn":
        model = CNN1D(in_ch=4).to(device)  # one_hot_4ch => 4 channels
    else:
        vocab = (4 ** args.k) + 1
        model = KmerTransformer(
            vocab_size=vocab,
            d_model=args.d_model,
            nhead=args.heads,
            num_layers=args.layers,
            max_len=args.L_kmers,
        ).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # best checkpoint tracking
    best_score = -1.0
    best_path = out_dir / f"{args.mode}_best.pt"

    # richer log (helps notebook)
    log_path = rep_dir / f"{args.mode}_training_log.tsv"
    log_path.write_text("epoch\ttrain_loss\ttest_loss\ttest_acc\ttest_f1\ttest_auc\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * y.size(0)

        tr_loss = total / len(train_loader.dataset)

        te_loss, y_true, y_pred, y_prob = eval_full(model, test_loader, device)
        te_acc = float((y_true == y_pred).mean())
        te_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            te_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            te_auc = float("nan")

        print(
            f"epoch={epoch:02d} train_loss={tr_loss:.4f} "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} "
            f"test_f1={te_f1:.4f} test_auc={te_auc:.4f}"
        )

        with log_path.open("a") as f:
            f.write(f"{epoch}\t{tr_loss:.6f}\t{te_loss:.6f}\t{te_acc:.6f}\t{te_f1:.6f}\t{te_auc:.6f}\n")

        if args.select_best_by == "acc":
            score = te_acc
        elif args.select_best_by == "f1":
            score = te_f1
        else:
            score = te_auc if not np.isnan(te_auc) else -1.0

        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)

    print(f"Saved best checkpoint: {best_path} (best_{args.select_best_by}={best_score:.4f})")

    # -----------------------
    # Final report using BEST
    # -----------------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    te_loss, y_true, y_pred, y_prob = eval_full(model, test_loader, device)
    report = compute_metrics(y_true, y_pred, y_prob)

    read_ids = getattr(test_ds, "read_ids", None)

    save_reports(
        out_dir=rep_dir,
        mode=args.mode,
        report=report,
        test_loss=float(te_loss),
        args_dict=vars(args),
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        read_ids=read_ids,
        save_predictions=args.save_predictions,
    )

    print("\nFINAL TEST METRICS (best checkpoint):")
    print(
        f"loss={float(te_loss):.4f} acc={report['accuracy']:.4f} "
        f"prec={report['precision']:.4f} rec={report['recall']:.4f} "
        f"f1={report['f1']:.4f} auc={report['roc_auc']:.4f}"
    )
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(report["confusion_matrix"]))
    print(f"\nWrote reports to: {rep_dir}")


if __name__ == "__main__":
    train_main()