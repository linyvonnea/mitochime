#!/usr/bin/env python3
import argparse, json, gzip
import joblib
import pandas as pd

def open_text(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

def norm_id_any(s: str) -> str:
    """Normalize either FASTQ header or TSV read_id to base ID."""
    s = str(s).strip()
    if not s:
        return s
    s = s.split()[0]          # drop Illumina trailing fields
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
            out.write(h); out.write(seq); out.write(plus); out.write(qual)

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

    feature_cols = json.load(open(args.feature_cols))
    df = pd.read_csv(args.features, sep="\t")

    if "read_id" not in df.columns:
        raise SystemExit("[ERROR] TSV missing required column: read_id")

    # normalize TSV IDs to base IDs
    df["read_id"] = df["read_id"].map(norm_id_any)

    # if strand is "+/-" in some TSVs, map to numeric; if already 0/1, this won't change it
    if "strand" in df.columns and df["strand"].dtype == object:
        df["strand"] = df["strand"].map({"+": 1.0, "-": 0.0})

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing features in TSV: {missing}")

    X = df[feature_cols].astype(float)
    model = joblib.load(args.model)

    proba = model.predict_proba(X)[:, 1]
    keep_mask = proba < args.thresh
    keep_ids = set(df.loc[keep_mask, "read_id"].astype(str).tolist())

    # Pair-safe: store kept R1 records, write only matching R2
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

if __name__ == "__main__":
    main()