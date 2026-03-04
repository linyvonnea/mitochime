#!/usr/bin/env bash
set -euo pipefail

R1="$1"
R2="$2"
RUN="$3"
THRESH="$4"
THREADS="${5:-8}"
REF="${6:-data/refs/original.fasta}"

[[ -f "$REF" ]] || { echo "[ERROR] Reference not found: $REF" >&2; exit 1; }

MODEL="models_pair_noq_tuned/gradient_boosting_tuned.joblib"
FEATURE_COLS="models_pair_noq_tuned/feature_cols.json"

OUTDIR="data/gb/${RUN}"
mkdir -p "$OUTDIR" "data/filtered_reads/${RUN}_gb"

BAM="${OUTDIR}/${RUN}.sorted.bam"
TSV="${OUTDIR}/${RUN}.features.tsv"

minimap2 -t "$THREADS" -ax sr "$REF" "$R1" "$R2" \
  | samtools view -b - \
  | samtools sort -@ "$THREADS" -o "$BAM" -

samtools index "$BAM"

python3 src/scripts/extract_features.py \
  --bam "$BAM" \
  --out "$TSV" \
  --label -1 \
  --k 6 \
  --micro-window 40

python3 src/scripts/gb_predict_filter.py \
  --r1 "$R1" \
  --r2 "$R2" \
  --features "$TSV" \
  --model "$MODEL" \
  --feature-cols "$FEATURE_COLS" \
  --thresh "$THRESH" \
  --out-r1 "data/filtered_reads/${RUN}_gb/${RUN}_gb.gb.filtered_R1.fastq.gz" \
  --out-r2 "data/filtered_reads/${RUN}_gb/${RUN}_gb.gb.filtered_R2.fastq.gz"