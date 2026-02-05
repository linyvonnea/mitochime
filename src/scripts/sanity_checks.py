#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("unsupervised/data/features_merged.unique.tsv")  # <-- use the UNIQUE file
ID_COL = "read_uid"
LABEL_COL = "y_sim"  # set to None if you truly don't want labels present

# If you still want "near-constant" flags, keep them as *flags* (not auto-drop)
NEAR_CONST_UNIQUE_FRAC = 0.001     # <=0.1% unique values
NEAR_CONST_DOMINANCE = 0.999       # >=99.9% of values are the same
NEAR_CONST_STD_EPS = 1e-12         # effectively zero std

OUT_DIR = Path("unsupervised/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(DATA_PATH, sep="\t")

print("\n=== Step 2.1: Shape ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Column names:", list(df.columns))

# -----------------------------
# Meta columns
# -----------------------------
meta_cols = [c for c in [ID_COL, LABEL_COL] if c and c in df.columns]

# -----------------------------
# Choose feature columns
#   - For unsupervised, prefer NUMERIC ONLY.
#   - Explicitly exclude known non-feature/string cols if present.
# -----------------------------
exclude_cols = set(meta_cols + ["pair_id", "read_id"])  # keep if present
candidate_cols = [c for c in df.columns if c not in exclude_cols]

numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
non_numeric_cols = [c for c in candidate_cols if c not in numeric_cols]

print("\nNumeric feature cols:", len(numeric_cols))
print("Non-numeric cols excluded:", non_numeric_cols)

# -----------------------------
# Step 2.2: Missing values (numeric features only)
# -----------------------------
print("\n=== Step 2.2: Missing values (numeric only) ===")
missing_counts = df[numeric_cols].isna().sum().sort_values(ascending=False)
missing_pct = (missing_counts / len(df) * 100).round(3)
missing_report = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})

print(missing_report.head(30))
missing_report.to_csv(OUT_DIR / "missing_values_report.csv", index=True)

# -----------------------------
# Step 2.3: Constant / near-constant checks (numeric only)
# -----------------------------
print("\n=== Step 2.3: Constant / near-constant features (numeric only) ===")

const_cols = []
near_const_cols = []
rows = []

for col in numeric_cols:
    s = df[col].dropna()

    if s.empty:
        const_cols.append(col)
        rows.append({
            "feature": col, "n_nonnull": 0, "nunique": 0, "unique_frac": 0.0,
            "top_value": np.nan, "top_value_frac": np.nan, "std": np.nan,
            "flag": "ALL_MISSING"
        })
        continue

    nunique = int(s.nunique())
    unique_frac = float(nunique / len(s))
    vc = s.value_counts(normalize=True, dropna=False)
    top_value = vc.index[0]
    top_value_frac = float(vc.iloc[0])
    std_val = float(s.std(ddof=0))

    flag = ""
    if nunique <= 1:
        const_cols.append(col)
        flag = "CONSTANT"
    else:
        is_near_const = (
            unique_frac <= NEAR_CONST_UNIQUE_FRAC
            or top_value_frac >= NEAR_CONST_DOMINANCE
            or std_val <= NEAR_CONST_STD_EPS
        )
        if is_near_const:
            near_const_cols.append(col)
            flag = "NEAR_CONSTANT"

    rows.append({
        "feature": col,
        "n_nonnull": int(len(s)),
        "nunique": nunique,
        "unique_frac": unique_frac,
        "top_value": top_value,
        "top_value_frac": top_value_frac,
        "std": std_val,
        "flag": flag
    })

variance_report = pd.DataFrame(rows).sort_values(
    by=["flag", "unique_frac", "top_value_frac"],
    ascending=[False, True, False]
)

print("\nConstant features:", const_cols)
print("Near-constant features (flag only):", near_const_cols)

variance_report.to_csv(OUT_DIR / "variance_report_numeric.csv", index=False)

# IMPORTANT RULE:
# For your case, drop ONLY truly constant columns for now.
drop_cols_constant = sorted(const_cols)

print("\n=== Output ===")
print("drop_cols_constant =", drop_cols_constant)
pd.Series(drop_cols_constant).to_csv(OUT_DIR / "drop_cols_constant.txt", index=False, header=False)

# -----------------------------
# Step 2.4: Duplicate checks
# -----------------------------
print("\n=== Step 2.4: Duplicate checks ===")

# 2.4.1 Duplicate IDs (should be 0 for read_uid)
if ID_COL in df.columns:
    dup_id_count = int(df[ID_COL].duplicated().sum())
    print(f"Duplicate {ID_COL} entries:", dup_id_count)
else:
    print(f"ID column '{ID_COL}' not found; skipping duplicate ID check.")

# 2.4.2 Duplicate numeric feature rows (exact duplicates)
dup_feat_count = int(df[numeric_cols].duplicated().sum())
print("Duplicate numeric feature rows:", dup_feat_count)

if dup_feat_count > 0:
    dup_preview = df.loc[df[numeric_cols].duplicated(keep=False), meta_cols + numeric_cols].head(20)
    dup_preview.to_csv(OUT_DIR / "duplicate_numeric_feature_rows_preview.csv", index=False)

print("\nDone. Reports written to:", OUT_DIR)