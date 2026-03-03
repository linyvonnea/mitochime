#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


def open_text(path: str, mode: str = "rt"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode)
    return open(p, mode)


def norm_id(h: str) -> str:
    s = h.strip().split()[0]
    if s.startswith("@"):
        s = s[1:]
    if s.endswith("/1") or s.endswith("/2"):
        s = s[:-2]
    return s


def iter_fastq(path: str) -> Iterator[Tuple[str, str]]:
    """
    Yields (base_read_id, seq) from FASTQ records.
    """
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
            yield norm_id(h), seq


def pad_or_trim_seq(seq: str, L: int) -> str:
    seq = seq.upper()
    if len(seq) >= L:
        return seq[:L]
    return seq + ("N" * (L - len(seq)))


def kmer_to_id(kmer: str) -> int:
    """
    Base-4 encoding A,C,G,T -> 0..(4^k - 1).
    Any k-mer containing non-ACGT -> UNK id (0).
    We reserve 0 as UNK/PAD, and shift real kmers by +1.
    """
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    v = 0
    for ch in kmer:
        if ch not in m:
            return 0  # UNK
        v = v * 4 + m[ch]
    return v + 1  # shift so UNK=0


def seq_to_kmer_tokens(seq: str, k: int, L_kmers: int) -> np.ndarray:
    tokens = []
    n = len(seq)
    if n >= k:
        for i in range(n - k + 1):
            tokens.append(kmer_to_id(seq[i : i + k]))

    if len(tokens) >= L_kmers:
        tokens = tokens[:L_kmers]
    else:
        tokens = tokens + [0] * (L_kmers - len(tokens))

    return np.asarray(tokens, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--out", required=True, help="Output .npz (tokens array)")
    ap.add_argument("--out-ids", required=True, help="Output .ids.txt (base read_id per row)")
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--L-kmers", type=int, default=256)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_ids = Path(args.out_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ids.parent.mkdir(parents=True, exist_ok=True)

    ids = []
    toks = []

    for rid, seq in iter_fastq(args.r1):
        seq = pad_or_trim_seq(seq, args.read_len)
        tok = seq_to_kmer_tokens(seq, k=args.k, L_kmers=args.L_kmers)
        ids.append(rid)
        toks.append(tok)

    if len(toks) == 0:
        raise SystemExit("[ERROR] No records found in R1 FASTQ.")

    X = np.stack(toks, axis=0)  # (N, L_kmers)

    np.savez_compressed(out_path, X=X)
    out_ids.write_text("\n".join(ids) + "\n")

    print(f"[OK] Encoded N={X.shape[0]:,} reads")
    print(f"[OK] tokens shape={X.shape} dtype={X.dtype}")
    print(f"[OK] wrote: {out_path}")
    print(f"[OK] wrote: {out_ids}")


if __name__ == "__main__":
    main()