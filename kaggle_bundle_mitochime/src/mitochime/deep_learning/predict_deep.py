# src/mitochime/deep_learning/predict_deep.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# reuse your existing tokenizer helpers
from .dl_data import seq_to_kmer_tokens

# IMPORTANT: import the exact model class you used in training rnn_kmer_gru
# If your class name differs, change this import accordingly.
from .dl_rnn_kmer import RNNKmerClassifier  # <-- must exist in your repo


class InferSeqDataset(Dataset):
    """
    Inference dataset: input TSV columns: read_id, seq
    Produces x for rnn_kmer models: (L_kmers,) int64 tokens
    """
    def __init__(self, tsv_path: str, L: int, k: int, L_kmers: int):
        df = pd.read_csv(tsv_path, sep="\t")
        if "read_id" not in df.columns or "seq" not in df.columns:
            raise ValueError("Inference TSV must contain columns: read_id, seq")

        self.read_ids = df["read_id"].astype(str).tolist()
        seqs = df["seq"].astype(str).str.upper().tolist()

        # safety: enforce exact L like training
        fixed = []
        for s in seqs:
            if len(s) != L:
                s = (s[:L] + ("N" * max(0, L - len(s))))
            fixed.append(s)
        self.seqs = fixed

        self.L = L
        self.k = k
        self.L_kmers = L_kmers

        print(
            f"[InferSeqDataset] loaded={len(self.read_ids):,} from {tsv_path} | "
            f"L={L} k={k} L_kmers={L_kmers}"
        )

    def __len__(self) -> int:
        return len(self.read_ids)

    def __getitem__(self, idx: int):
        seq = self.seqs[idx]
        tok = seq_to_kmer_tokens(seq, k=self.k, L_kmers=self.L_kmers)  # (L_kmers,)
        x = torch.from_numpy(tok)  # int64
        return x


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs = []
    for x in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]  # P(chimeric)
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="TSV from make_seq_tsv_infer (read_id, seq)")
    ap.add_argument("--ckpt", required=True, help="*.pt checkpoint from training")

    ap.add_argument("--L", type=int, default=150)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--L-kmers", type=int, default=147)

    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rnn-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--pool", choices=["last", "mean", "max"], default="last")
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=512)

    ap.add_argument("--out", required=True, help="predictions TSV output")
    ap.add_argument("--remove-ids", required=True, help="output remove_ids_raw.txt (one per line)")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    ds = InferSeqDataset(args.tsv, L=args.L, k=args.k, L_kmers=args.L_kmers)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # load model
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = RNNKmerClassifier(
        rnn_type="gru",
        vocab_size=(4 ** args.k) + 1,   # UNK=0, real kmers 1..4^k
        embed_dim=args.embed_dim,
        hidden_size=args.hidden,
        num_layers=args.rnn_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    p = predict_probs(model, loader, device=device)
    if len(p) != len(ds.read_ids):
        raise RuntimeError("Probability count mismatch with read_ids")

    yhat = (p >= args.threshold).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rem_path = Path(args.remove_ids)
    rem_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "read_id": ds.read_ids,
        "prob_chimeric": p,
        "is_chimeric": yhat,
    })
    df.to_csv(out_path, sep="\t", index=False)

    remove = df.loc[df["is_chimeric"] == 1, "read_id"]
    remove.to_csv(rem_path, index=False, header=False)

    print(f"[OK] wrote predictions: {out_path} (n={len(df):,})")
    print(f"[OK] wrote remove_ids_raw: {rem_path} (n_remove={len(remove):,})")
    print(f"[INFO] threshold={args.threshold:.3f}")

if __name__ == "__main__":
    main()