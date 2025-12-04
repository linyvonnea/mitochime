#!/usr/bin/env python3
"""
Create a stratified train/test split from the merged feature table.

Usage:
    python -m mitochime.split_dataset \
        --input data/features/v1/all_features_merged.tsv \
        --outdir data/processed \
        --test-size 0.2 \
        --random-state 42
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_utils import load_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/test split for ML.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to merged feature TSV (e.g., all_features_merged.tsv).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to write train.tsv and test.tsv.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use as test set (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_feature_table(args.input)

    if "label" not in df.columns:
        raise ValueError("Input TSV must contain a 'label' column.")

    y = df["label"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_path = os.path.join(args.outdir, "train.tsv")
    test_path = os.path.join(args.outdir, "test.tsv")

    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    print(f"Wrote train set: {train_path}  (n={len(train_df)})")
    print(f"Wrote test set : {test_path}   (n={len(test_df)})")


if __name__ == "__main__":
    main()