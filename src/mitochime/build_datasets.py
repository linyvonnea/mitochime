#!/usr/bin/env python3
"""
build_datasets.py

Combine clean & chimeric feature tables, do basic cleaning, and
create train/test splits.

Usage
-----
python3 -m mitochime.build_datasets \
  --clean data/features/v1/clean_features_k6.tsv \
  --chim  data/features/v1/chimera_features_k6.tsv \
  --out-all data/processed/all_reads_k6.tsv \
  --train data/processed/train_k6.tsv \
  --test  data/processed/test_k6.tsv \
  --test-size 0.2 \
  --random-state 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined dataset and train/test split."
    )
    parser.add_argument("--clean", required=True, help="TSV with clean features.")
    parser.add_argument("--chim", required=True, help="TSV with chimeric features.")
    parser.add_argument("--out-all", required=True, help="Output TSV for full dataset.")
    parser.add_argument("--train", required=True, help="Output TSV for training set.")
    parser.add_argument("--test", required=True, help="Output TSV for test set.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    clean = pd.read_csv(args.clean, sep="\t")
    chim = pd.read_csv(args.chim,  sep="\t")

    # sanity: ensure label column exists and is correct
    if "label" not in clean.columns:
        clean["label"] = 0
    if "label" not in chim.columns:
        chim["label"] = 1

    df = pd.concat([clean, chim], axis=0, ignore_index=True)

    # shuffle just to be safe
    df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    # save full dataset
    out_all = Path(args.out_all)
    out_all.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_all, sep="\t", index=False)
    print(f"[build_datasets] Wrote full dataset to {out_all} with shape {df.shape}")

    # train-test split (stratified)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train = X_train.copy()
    train["label"] = y_train.to_numpy()

    test = X_test.copy()
    test["label"] = y_test.to_numpy()

    train_path = Path(args.train)
    test_path = Path(args.test)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(train_path, sep="\t", index=False)
    test.to_csv(test_path,   sep="\t", index=False)

    print(f"[build_datasets] Train: {train.shape}, Test: {test.shape}")


if __name__ == "__main__":
    main()