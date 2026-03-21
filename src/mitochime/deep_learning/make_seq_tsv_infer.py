#!/usr/bin/env python3
"""
Make a sequence TSV for INFERENCE (no labels) from arbitrary FASTQ/FASTQ.GZ.

Output columns:
  read_id   seq

Example:
PYTHONPATH=src python3 -m mitochime.deep_learning.make_seq_tsv_infer \
  --r1 reads_R1.fastq.gz \
  --r2 reads_R2.fastq.gz \
  --L 150 \
  --out data/dl/myrun.seq.tsv
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterable, Tuple


def open_text(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def normalize_header_to_read_id(header: str) -> str:
    """
    Keep the first token, drop '@', keep /1 or /2 if present.
    """
    h = header.strip().split()[0]
    if h.startswith("@"):
        h = h[1:]
    return h


def iter_fastq(path: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (read_id, seq) from FASTQ.
    """
    with open_text(path, "rt") as f:
        while True:
            h = f.readline()
            if not h:
                break
            seq = f.readline()
            plus = f.readline()
            qual = f.readline()
            if not qual:
                break
            rid = normalize_header_to_read_id(h)
            yield rid, seq.strip()


def pad_or_trim(seq: str, L: int) -> str:
    seq = seq.upper()
    if len(seq) >= L:
        return seq[:L]
    return seq + ("N" * (L - len(seq)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--L", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read R1 into dict by base_id to ensure pairing consistency
    # (We only keep bases that exist in BOTH R1 and R2.)
    r1_map = {}
    for rid, seq in iter_fastq(args.r1):
        base = rid[:-2] if rid.endswith("/1") or rid.endswith("/2") else rid
        r1_map[base] = (rid, pad_or_trim(seq, args.L))

    rows = []
    kept_pairs = 0
    scanned_r2 = 0

    for rid2, seq2 in iter_fastq(args.r2):
        scanned_r2 += 1
        base = rid2[:-2] if rid2.endswith("/1") or rid2.endswith("/2") else rid2
        if base not in r1_map:
            continue

        rid1, seq1 = r1_map[base]
        seq2 = pad_or_trim(seq2, args.L)

        # Ensure read_id suffixes exist (force /1 and /2)
        if not rid1.endswith("/1"):
            rid1 = base + "/1"
        if not rid2.endswith("/2"):
            rid2 = base + "/2"

        rows.append((rid1, seq1))
        rows.append((rid2, seq2))
        kept_pairs += 1

    if kept_pairs == 0:
        raise SystemExit("[ERROR] No paired reads found between R1 and R2.")

    # Write TSV
    with out_path.open("w") as out:
        out.write("read_id\tseq\n")
        for rid, seq in rows:
            out.write(f"{rid}\t{seq}\n")

    print(f"[make_seq_tsv_infer] wrote {len(rows):,} reads ({kept_pairs:,} pairs) to {out_path}")
    print(f"[make_seq_tsv_infer] scanned R2 records: {scanned_r2:,}")


if __name__ == "__main__":
    main()