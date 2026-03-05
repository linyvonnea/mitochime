#!/usr/bin/env python3
import argparse, gzip
from typing import Iterator, Tuple

def open_text(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

def norm_id_any(s: str) -> str:
    s = str(s).strip().split()[0]
    if s.startswith("@"):
        s = s[1:]
    if s.endswith("/1") or s.endswith("/2"):
        s = s[:-2]
    return s

def iter_fastq(path: str) -> Iterator[Tuple[str,str,str,str]]:
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
            yield h, seq, plus, qual

def write_fastq(out_path: str, records: Iterator[Tuple[str,str,str,str]]):
    with open_text(out_path, "wt") as out:
        for h, seq, plus, qual in records:
            out.write(h); out.write(seq); out.write(plus); out.write(qual)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--remove-ids", required=True, help="one base read_id per line")
    ap.add_argument("--out-r1", required=True)
    ap.add_argument("--out-r2", required=True)
    args = ap.parse_args()

    remove = set()
    with open_text(args.remove_ids, "rt") as f:
        for line in f:
            x = line.strip()
            if x:
                remove.add(norm_id_any(x))

    r1_keep = {}
    for rec in iter_fastq(args.r1):
        rid = norm_id_any(rec[0])
        if rid not in remove:
            r1_keep[rid] = rec

    def r2_records():
        for rec in iter_fastq(args.r2):
            rid = norm_id_any(rec[0])
            if rid in r1_keep:
                yield rec

    write_fastq(args.out_r1, (r1_keep[rid] for rid in r1_keep.keys()))
    write_fastq(args.out_r2, r2_records())

    print("[OK] Pair-safe filtered outputs:")
    print(f"  {args.out_r1}")
    print(f"  {args.out_r2}")

if __name__ == "__main__":
    main()