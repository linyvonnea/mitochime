#!/usr/bin/env python3
import argparse
import joblib
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove-ids", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.features, sep="\t")
    if "read_id" not in df.columns:
        raise ValueError("features TSV must contain a read_id column")

    read_ids = df["read_id"].astype(str)

    # Drop non-feature columns commonly present
    drop_cols = [c for c in ["read_id", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Keep only numeric columns (prevents ref_name/cigar breaking inference)
    X = X.select_dtypes(include=[np.number])

    model = joblib.load(args.model)

    # Hard guard: match expected feature count
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            f"Feature mismatch: model expects {expected} features, but input has {X.shape[1]}.\n"
            f"Numeric columns seen: {list(X.columns)}"
        )

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        p = model.predict(X)

    is_chimera = (p >= args.threshold).astype(int)

    out_df = pd.DataFrame({"read_id": read_ids, "p_chimera": p, "is_chimera": is_chimera})
    out_df.to_csv(args.out, sep="\t", index=False)

    remove = out_df.loc[out_df["is_chimera"] == 1, "read_id"]
    remove.to_csv(args.remove_ids, index=False, header=False)

    print(f"[OK] predictions -> {args.out}")
    print(f"[OK] remove_ids_raw -> {args.remove_ids} (n={len(remove)})")

if __name__ == "__main__":
    main()