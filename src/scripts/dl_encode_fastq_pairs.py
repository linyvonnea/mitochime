#!/usr/bin/env python3
import argparse
import gzip
import numpy as np

MAP = {
    "A": 0, "C": 1, "G": 2, "T": 3,
    "a": 0, "c": 1, "g": 2, "t": 3,
}

def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def fastq_iter(path):
    with open_maybe_gz(path) as f:
        while True:
            h = f.readline().rstrip()
            if not h:
                break
            seq = f.readline().rstrip()
            _ = f.readline()
            _ = f.readline()
            yield h, seq

def norm_id(h: str) -> str:
    # "@id/1 ..." -> "id/1"
    if h.startswith("@"):
        h = h[1:]
    return h.split()[0]

def base_id(read_id: str) -> str:
    # "X/1" -> "X"
    if read_id.endswith("/1") or read_id.endswith("/2"):
        return read_id[:-2]
    return read_id

def encode_seq_4ch(seq: str, L: int) -> np.ndarray:
    """
    4-channel one-hot (A,C,G,T). Any non-ACGT becomes all-zeros at that position.
    Output: (4, L)
    """
    x = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq[:L]):
        idx = MAP.get(ch, None)
        if idx is not None:
            x[idx, i] = 1.0
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True, help="R1 FASTQ(.gz)")
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument("--out-ids", required=True, help="Output base read_ids.txt (one per read)")
    ap.add_argument("--max-reads", type=int, default=0, help="0 = all")
    ap.add_argument("--read-len", type=int, default=150)
    args = ap.parse_args()

    X_list = []
    ids = []

    n = 0
    for h, s in fastq_iter(args.r1):
        rid = norm_id(h)     # id/1
        b = base_id(rid)     # id

        x = encode_seq_4ch(s, args.read_len)  # (4,150)

        X_list.append(x)
        ids.append(b)

        n += 1
        if args.max_reads and n >= args.max_reads:
            break

    X = np.stack(X_list, axis=0)  # (N, 4, 150)

    np.savez_compressed(args.out, X=X)
    with open(args.out_ids, "w") as f:
        for b in ids:
            f.write(b + "\n")

    print(f"[OK] Encoded reads: {X.shape} -> {args.out}")
    print(f"[OK] IDs: {len(ids)} -> {args.out_ids}")

if __name__ == "__main__":
    main()