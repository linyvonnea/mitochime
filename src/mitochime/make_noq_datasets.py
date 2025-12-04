#!/usr/bin/env python3
"""
make_noq_datasets.py

Create ablated versions of train/test TSVs that drop
"cheaty" read-level metadata like mean_base_quality and ref_start_1based.

Input:
  data/processed/train.tsv
  data/processed/test.tsv

Output:
  data/processed/train_noq.tsv
  data/processed/test_noq.tsv

  Run: python3 -m mitochime.make_noq_datasets
"""

from pathlib import Path
import pandas as pd

# Columns we do NOT want the model to see
DROP_COLS = [
    "mean_base_quality",
    "ref_start_1based",
    # if in the future you also want to drop others, add them here:
    # "read_length",
    # "mapq",
]

def make_noq(split: str) -> None:
    in_path = Path(f"data/processed/{split}.tsv")
    out_path = Path(f"data/processed/{split}_noq.tsv")

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path, sep="\t")

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    print(f"  Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    print(f"  Wrote {out_path} with shape {df.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def main() -> None:
    for split in ["train", "test"]:
        make_noq(split)


if __name__ == "__main__":
    main()