#!/usr/bin/env python3
'''PYTHONPATH=src python3 -m mitochime.build_pair_splits \
  --all data/processed/all_reads.tsv \
  --out-train data/processed/PAIR_train.tsv \
  --out-test  data/processed/PAIR_test.tsv \
  --test-size 0.2 \
  --random-state 42'''
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True, help="all_reads.tsv")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.all, sep="\t")
    if "read_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns: read_id, label")

    # pair-level label: each read_id should have consistent label
    pair = df[["read_id", "label"]].drop_duplicates("read_id")
    pair["label"] = pair["label"].astype(int)

    train_ids, test_ids = train_test_split(
        pair["read_id"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=pair["label"],
    )

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    df_train = df[df["read_id"].isin(train_ids)].copy()
    df_test  = df[df["read_id"].isin(test_ids)].copy()

    # sanity: no overlap
    assert set(df_train["read_id"]).isdisjoint(set(df_test["read_id"]))

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test).parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(args.out_train, sep="\t", index=False)
    df_test.to_csv(args.out_test, sep="\t", index=False)

    print(f"[PAIR split] train rows={len(df_train):,} unique_ids={df_train['read_id'].nunique():,}")
    print(f"[PAIR split] test  rows={len(df_test):,} unique_ids={df_test['read_id'].nunique():,}")

if __name__ == "__main__":
    main()