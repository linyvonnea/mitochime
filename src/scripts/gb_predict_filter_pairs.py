#!/usr/bin/env python3
import argparse
import json
import gzip
import joblib
import numpy as np
import pandas as pd


def open_text(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def norm_id_any(s: str) -> str:
    """Normalize either FASTQ header or TSV read_id to base ID."""
    s = str(s).strip()
    if not s:
        return s
    s = s.split()[0]
    if s.startswith("@"):
        s = s[1:]
    if s.endswith("/1") or s.endswith("/2"):
        s = s[:-2]
    return s


def iter_fastq(path: str):
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


def write_fastq(out_path: str, records):
    with open_text(out_path, "wt") as out:
        for h, seq, plus, qual in records:
            out.write(h)
            out.write(seq)
            out.write(plus)
            out.write(qual)


def normalize_strand_value(x):
    """
    Robust strand normalization.

    Accepts:
    - '+' / '-'
    - 1 / 0
    - 1.0 / 0.0
    - reverse/forward-like strings if ever present
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s in {"+", "1", "1.0", "forward", "fwd", "F", "plus"}:
        return 1.0
    if s in {"-", "0", "0.0", "reverse", "rev", "R", "minus"}:
        return 0.0

    # try numeric fallback
    try:
        v = float(s)
        if v == 1.0:
            return 1.0
        if v == 0.0:
            return 0.0
    except Exception:
        pass

    return np.nan


def prepare_X_like_hparam_script(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Match hyperparam_search_top.py as closely as possible, but NEVER drop a column.

    Steps:
      - normalize strand robustly
      - drop read_id/ref_name/cigar
      - coerce numerics
      - reindex to exact feature_cols order
      - fill NaNs per column using column median if available, else 0.0
    """
    df = df.copy()

    # robust strand normalization
    if "strand" in df.columns:
        df["strand"] = df["strand"].map(normalize_strand_value)

    # drop known non-feature text columns
    drop_cols = ["read_id", "ref_name", "cigar"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # coerce all remaining columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure every expected feature exists
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # exact order
    X_df = df[feature_cols].copy()

    # fill NaNs column by column; never let a column disappear
    for c in X_df.columns:
        med = X_df[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        X_df[c] = X_df[c].fillna(med)

    X = X_df.to_numpy(dtype=float)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--features", required=True, help="TSV from extract_features.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature-cols", required=True, help="JSON list of columns in correct order")
    ap.add_argument("--thresh", type=float, required=True)
    ap.add_argument("--out-r1", required=True)
    ap.add_argument("--out-r2", required=True)
    args = ap.parse_args()

    with open(args.feature_cols) as f:
        feature_cols = json.load(f)

    df = pd.read_csv(args.features, sep="\t")

    if "read_id" not in df.columns:
        raise SystemExit("[ERROR] TSV missing required column: read_id")

    df["read_id"] = df["read_id"].map(norm_id_any)

    X = prepare_X_like_hparam_script(df, feature_cols)
    print(f"[INFO] X shape after preprocessing: {X.shape}")

    model = joblib.load(args.model)

    exp = getattr(model, "n_features_in_", None)
    if exp is not None and X.shape[1] != exp:
        raise SystemExit(f"[ERROR] Model expects {exp} features, but got {X.shape[1]}")

    proba = model.predict_proba(X)[:, 1]
    keep_mask = proba < args.thresh
    keep_ids = set(df.loc[keep_mask, "read_id"].astype(str).tolist())

    r1_keep = {}
    for rec in iter_fastq(args.r1):
        rid = norm_id_any(rec[0])
        if rid in keep_ids:
            r1_keep[rid] = rec

    def r2_records():
        for rec in iter_fastq(args.r2):
            rid = norm_id_any(rec[0])
            if rid in r1_keep:
                yield rec

    write_fastq(args.out_r1, (r1_keep[rid] for rid in r1_keep.keys()))
    write_fastq(args.out_r2, r2_records())

    print("[OK] GB pair-safe filtered outputs:")
    print(f"  {args.out_r1}")
    print(f"  {args.out_r2}")
    print(f"[INFO] kept pairs: {len(r1_keep)}")


if __name__ == "__main__":
    main()