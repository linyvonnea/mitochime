#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = Path("unsupervised/data/features_merged.tsv")
OUT_PATH = Path("unsupervised/data/features_merged.unique.tsv")

ID_COL = "read_id"
STRAND_COL = "strand"   # 0 = forward, 1 = reverse

df = pd.read_csv(IN_PATH, sep="\t")

# keep original for grouping
if "pair_id" not in df.columns:
    df.insert(0, "pair_id", df[ID_COL].astype(str))
else:
    df["pair_id"] = df["pair_id"].astype(str)

# map strand -> label
strand_map = {0: "F", 1: "R", "0": "F", "1": "R"}
df["_strand_label"] = df[STRAND_COL].map(strand_map)

# if strand has unexpected values, mark as U (unknown)
df["_strand_label"] = df["_strand_label"].fillna("U")

# base unique id using strand
df["read_uid"] = df["pair_id"] + "/" + df["_strand_label"]

# ---- Edge case: still duplicates (e.g., multiple rows for same pair_id+strand)
# add a counter suffix only when needed
dup_mask = df["read_uid"].duplicated(keep=False)
if dup_mask.any():
    df.loc[dup_mask, "read_uid"] = (
        df.loc[dup_mask]
          .groupby("read_uid")
          .cumcount()
          .add(1)
          .astype(str)
          .radd(df.loc[dup_mask, "read_uid"] + "__rep")
    )

# sanity
assert df["read_uid"].is_unique, "read_uid is still not unique!"

# optional: replace read_id with read_uid OR keep both
# Keep both by default:
# df[ID_COL] stays as pair id, df["read_uid"] is row-unique.

df.drop(columns=["_strand_label"], inplace=True)
df.to_csv(OUT_PATH, sep="\t", index=False)

print("Rows:", len(df))
print("Unique pair_id:", df["pair_id"].nunique())
print("Unique read_uid:", df["read_uid"].nunique())
print("Wrote:", OUT_PATH)