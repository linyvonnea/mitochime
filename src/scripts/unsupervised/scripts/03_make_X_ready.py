#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("unsupervised/data/features_merged.unique.tsv")
OUT_DIR = Path("unsupervised/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "read_uid"
LABEL_COL = "y_sim"  # keep for evaluation only (optional)

# From Step 2
DROP_CONST = ["read_length", "sa_diff_contig", "sa_same_strand_count"]

# Columns that are numeric but you may not want as model inputs
# (we exclude strand because you already encoded F/R into read_uid)
EXCLUDE_NUMERIC = ["strand"]

# Features that are heavy-tailed -> log1p helps
LOG1P_COLS = [
    "softclip_left", "softclip_right", "total_clipped_bases",
    "sa_min_delta_pos", "sa_max_delta_pos", "sa_mean_delta_pos",
    "ref_start_1based"
]

def robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR scaling (robust to outliers)."""
    med = df.median(axis=0)
    q1 = df.quantile(0.25, axis=0)
    q3 = df.quantile(0.75, axis=0)
    iqr = (q3 - q1).replace(0, 1.0)  # avoid divide by zero
    return (df - med) / iqr

df = pd.read_csv(IN_PATH, sep="\t")

# --- meta
meta_cols = [c for c in [ID_COL, LABEL_COL] if c in df.columns]
meta = df[meta_cols].copy()

# --- numeric features only
exclude_cols = set(meta_cols + ["pair_id", "read_id"])
candidate_cols = [c for c in df.columns if c not in exclude_cols]
numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

# Drop constants + excluded numeric cols
drop_cols = set(DROP_CONST + EXCLUDE_NUMERIC)
numeric_cols = [c for c in numeric_cols if c not in drop_cols]

X = df[numeric_cols].copy()

# --- log1p transforms (only if present)
for c in LOG1P_COLS:
    if c in X.columns:
        X[c] = np.log1p(X[c].astype(float))

# --- robust scaling
X_scaled = robust_scale(X)

# --- save
X_scaled.to_csv(OUT_DIR / "X_ready.tsv", sep="\t", index=False)
meta.to_csv(OUT_DIR / "meta.tsv", sep="\t", index=False)

print("Saved:", OUT_DIR / "X_ready.tsv")
print("Saved:", OUT_DIR / "meta.tsv")
print("X_ready shape:", X_scaled.shape)
print("Feature columns:", len(X_scaled.columns))
print("Meta columns:", meta_cols)