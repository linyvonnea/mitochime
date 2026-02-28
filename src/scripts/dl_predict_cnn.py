# src/scripts/dl_predict_cnn.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import pandas as pd
import torch

def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model_state" in ckpt):
        raise RuntimeError("Expected checkpoint dict with model_state")

    # IMPORTANT: import your trained architecture
    from mitochime.deep_learning.dl_cnn import CNN1D

    # checkpoint indicates in_ch=4 for L150 model
    model = CNN1D(in_ch=4, dropout=0.2)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt.get("args", {})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove-ids", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    X = np.load(args.npz)["X"]  # expects (N,4,150)

    with open(args.ids) as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != X.shape[0]:
        raise ValueError(f"IDs ({len(ids)}) != samples ({X.shape[0]})")

    model, ckpt_args = load_model(args.model)

    probs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], args.batch_size):
            xb = torch.from_numpy(X[i:i+args.batch_size])
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())

    p_all = np.concatenate(probs)
    is_chimera = (p_all >= args.threshold).astype(int)

    df = pd.DataFrame({"read_id": ids, "p_chimera": p_all, "is_chimera": is_chimera})
    df.to_csv(args.out, sep="\t", index=False)

    remove = df.loc[df["is_chimera"] == 1, "read_id"]
    remove.to_csv(args.remove_ids, index=False, header=False)

    print(f"[OK] checkpoint args: {ckpt_args}")
    print(f"[OK] pred -> {args.out}")
    print(f"[OK] remove_ids_raw -> {args.remove_ids} (n={len(remove)})")

if __name__ == "__main__":
    main()