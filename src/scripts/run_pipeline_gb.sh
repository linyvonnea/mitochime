#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_gb.sh <R1> <R2> <RUN_NAME> <THRESH> [THREADS=8] [REF=data/refs/original.fasta]

R1="$1"
R2="$2"
RUN="$3"
THRESH="$4"
THREADS="${5:-8}"
REF="${6:-data/refs/original.fasta}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

command -v minimap2 >/dev/null 2>&1 || { echo "[ERROR] minimap2 not in PATH" >&2; exit 1; }
command -v samtools >/dev/null 2>&1 || { echo "[ERROR] samtools not in PATH" >&2; exit 1; }

[[ -f "$R1" ]] || { echo "[ERROR] R1 not found: $R1" >&2; exit 1; }
[[ -f "$R2" ]] || { echo "[ERROR] R2 not found: $R2" >&2; exit 1; }
[[ -f "$REF" ]] || { echo "[ERROR] Reference not found: $REF" >&2; exit 1; }

# sanity: if .gz, must be real gz
if [[ "$R1" =~ \.gz$ ]] && ! gzip -t "$R1" >/dev/null 2>&1; then
  echo "[ERROR] R1 has .gz extension but is not gzipped: $R1" >&2
  exit 1
fi
if [[ "$R2" =~ \.gz$ ]] && ! gzip -t "$R2" >/dev/null 2>&1; then
  echo "[ERROR] R2 has .gz extension but is not gzipped: $R2" >&2
  exit 1
fi

MODEL="models_noq_tuned/gradient_boosting_tuned.joblib"
FEATURE_COLS="models_noq_tuned/feature_cols.json"

[[ -f "$MODEL" ]] || { echo "[ERROR] Model not found: $MODEL" >&2; exit 1; }
[[ -f "$FEATURE_COLS" ]] || { echo "[ERROR] Feature cols not found: $FEATURE_COLS" >&2; exit 1; }

OUTDIR="data/gb/${RUN}"
FILTDIR="data/filtered_reads/${RUN}"
mkdir -p "$OUTDIR" "$FILTDIR"

BAM="${OUTDIR}/${RUN}.sorted.bam"
TSV="${OUTDIR}/${RUN}.features.tsv"

echo "[INFO] GB pipeline"
echo "[INFO] RUN=$RUN THRESH=$THRESH THREADS=$THREADS"
echo "[INFO] REF=$REF"
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] FILTDIR=$FILTDIR"

echo "[1/3] Map -> sort -> index BAM"
minimap2 -t "$THREADS" -ax sr "$REF" "$R1" "$R2" \
  | samtools view -b - \
  | samtools sort -@ "$THREADS" -o "$BAM" -

samtools index "$BAM"

echo "[2/3] Extract features -> TSV"
python src/scripts/extract_features.py \
  --bam "$BAM" \
  --out "$TSV" \
  --label -1 \
  --k 6 \
  --micro-window 40

echo "[3/3] Predict + pair-safe filter"
python src/scripts/gb_predict_filter_pairs.py \
  --r1 "$R1" \
  --r2 "$R2" \
  --features "$TSV" \
  --model "$MODEL" \
  --feature-cols "$FEATURE_COLS" \
  --thresh "$THRESH" \
  --out-r1 "${FILTDIR}/${RUN}.gb.filtered_R1.fastq.gz" \
  --out-r2 "${FILTDIR}/${RUN}.gb.filtered_R2.fastq.gz"

echo "[DONE] GB outputs:"
echo "  ${FILTDIR}/${RUN}.gb.filtered_R1.fastq.gz"
echo "  ${FILTDIR}/${RUN}.gb.filtered_R2.fastq.gz"