#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mitochime.deep_learning.dl_transformer import KmerTransformer


def load_ids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ from encoder (contains X)")
    ap.add_argument("--ids", required=True, help="ids.txt (base read ids per row)")
    ap.add_argument("--model", required=True, help="transformer_best.pt")
    ap.add_argument("--out", required=True, help="predictions TSV")
    ap.add_argument("--remove-ids", required=True, help="base IDs to remove (raw)")
    ap.add_argument("--threshold", type=float, required=True)

    # If you want to override (usually not needed if checkpoint has args)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--L-kmers", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)

    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoded tokens + ids
    z = np.load(args.npz)
    X = z["X"]  # (N, T)
    ids = load_ids(args.ids)
    if len(ids) != X.shape[0]:
        raise SystemExit(f"[ERROR] ids count ({len(ids)}) != X rows ({X.shape[0]})")

    # Load checkpoint
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # Resolve hyperparams: prefer CLI override, else checkpoint args, else defaults
    k = args.k if args.k is not None else int(ckpt_args.get("k", 6))
    T = args.L_kmers if args.L_kmers is not None else int(ckpt_args.get("L_kmers", 256))
    d_model = args.d_model if args.d_model is not None else int(ckpt_args.get("d_model", 128))
    layers = args.layers if args.layers is not None else int(ckpt_args.get("layers", 4))
    heads = args.heads if args.heads is not None else int(ckpt_args.get("heads", 4))

    if X.shape[1] != T:
        raise SystemExit(f"[ERROR] Encoded T={X.shape[1]} but expected T={T}. Re-encode with --L-kmers {T}.")

    vocab = (4 ** k) + 1  # UNK/PAD=0, kmers 1..4^k

    model = KmerTransformer(
        vocab_size=vocab,
        d_model=d_model,
        nhead=heads,
        num_layers=layers,
        max_len=T,
        pad_id=0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Inference
    probs = np.zeros((X.shape[0],), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, X.shape[0], args.batch):
            end = min(start + args.batch, X.shape[0])
            xb = torch.from_numpy(X[start:end]).to(device)  # int64
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]  # P(class=1 = chimeric)
            probs[start:end] = p.detach().cpu().numpy()

    # Remove rule: remove if prob_chimeric >= threshold
    remove_mask = probs >= float(args.threshold)
    remove_ids = [ids[i] for i in np.where(remove_mask)[0].tolist()]

    # Write predictions TSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("read_id\tprob_chimeric\n")
        for rid, pr in zip(ids, probs):
            f.write(f"{rid}\t{float(pr):.6f}\n")

    # Write remove ids raw (base ids)
    rem_path = Path(args.remove_ids)
    rem_path.parent.mkdir(parents=True, exist_ok=True)
    rem_path.write_text("\n".join(remove_ids) + ("\n" if remove_ids else ""))

    n = X.shape[0]
    nrem = len(remove_ids)
    print(f"[OK] transformer inference done on N={n:,}")
    print(f"[OK] threshold={args.threshold} -> remove={nrem:,} ({(nrem/n)*100:.2f}%)")
    print(f"[OK] wrote pred: {out_path}")
    print(f"[OK] wrote remove ids: {rem_path}")


if __name__ == "__main__":
    main()