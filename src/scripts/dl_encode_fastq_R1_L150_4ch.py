#!/usr/bin/env python3
import argparse, gzip
import numpy as np

def open_text(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def norm_id(h: str) -> str:
    s = h.strip().split()[0]
    if s.startswith("@"):
        s = s[1:]
    if s.endswith("/1") or s.endswith("/2"):
        s = s[:-2]
    return s

def iter_fastq(path: str):
    with open_text(path, "rt") as f:
        while True:
            h = f.readline()
            if not h:
                break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline()
            if not qual:
                break
            yield h, seq

def onehot_4ch(seq: str, L: int) -> np.ndarray:
    # shape (4, L)
    arr = np.zeros((4, L), dtype=np.float32)
    seq = seq.upper()
    # pad/truncate
    if len(seq) < L:
        seq = seq + ("N" * (L - len(seq)))
    else:
        seq = seq[:L]

    # A,C,G,T mapping
    for i, b in enumerate(seq):
        if b == "A":
            arr[0, i] = 1.0
        elif b == "C":
            arr[1, i] = 1.0
        elif b == "G":
            arr[2, i] = 1.0
        elif b == "T":
            arr[3, i] = 1.0
        # else: N/others remain 0
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--out-ids", required=True, help="output ids.txt")
    ap.add_argument("--read-len", type=int, default=150)
    args = ap.parse_args()

    L = args.read_len

    ids = []
    X_list = []

    for h, seq in iter_fastq(args.r1):
        rid = norm_id(h)
        ids.append(rid)
        X_list.append(onehot_4ch(seq, L))

    if not X_list:
        raise SystemExit("[ERROR] No reads found in R1")

    X = np.stack(X_list, axis=0)  # (N, 4, L)
    np.savez_compressed(args.out, X=X)

    with open(args.out_ids, "wt") as f:
        for rid in ids:
            f.write(rid + "\n")

    print(f"[OK] Encoded {X.shape[0]} reads -> {args.out}  shape={X.shape}")
    print(f"[OK] IDs -> {args.out_ids}")

if __name__ == "__main__":
    main()