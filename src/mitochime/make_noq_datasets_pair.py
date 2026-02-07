#!/usr/bin/env python3
"""
make_noq_datasets_pair.py


Create ablated (NOQ) versions of the *PAIR-level split* TSVs by dropping
"cheaty" metadata columns (e.g., mean_base_quality, ref_start_1based).

This is the SAME idea as make_noq_datasets.py, but:
- it lets you specify custom input/output paths (PAIR_train / PAIR_test)
- it does NOT overwrite your old supervised files

Example
-------
PYTHONPATH=src python3 -m mitochime.make_noq_datasets_pair \
  --train-in  data/processed/PAIR_train.tsv \
  --test-in   data/processed/PAIR_test.tsv \
  --train-out data/processed/PAIR_train_noq.tsv \
  --test-out  data/processed/PAIR_test_noq.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Columns we do NOT want the model to see
DROP_COLS = [
    "mean_base_quality",
    "ref_start_1based",
    # optional future drops:
    # "read_length",
    # "mapq",
]


def make_noq_file(in_path: Path, out_path: Path) -> None:
    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path, sep="\t")

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    print(f"  Dropping columns: {cols_to_drop}")

    df = df.drop(columns=cols_to_drop)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {out_path} with shape {df.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Drop cheaty metadata cols from PAIR_train/PAIR_test TSVs."
    )
    ap.add_argument("--train-in", required=True, help="e.g. data/processed/PAIR_train.tsv")
    ap.add_argument("--test-in", required=True, help="e.g. data/processed/PAIR_test.tsv")
    ap.add_argument("--train-out", required=True, help="e.g. data/processed/PAIR_train_noq.tsv")
    ap.add_argument("--test-out", required=True, help="e.g. data/processed/PAIR_test_noq.tsv")
    args = ap.parse_args()

    make_noq_file(Path(args.train_in), Path(args.train_out))
    make_noq_file(Path(args.test_in), Path(args.test_out))


if __name__ == "__main__":
    main()