#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_cnn.sh <R1> <R2> <RUN_NAME> [THRESH=0.5]

R1="$1"
R2="$2"
RUN="$3"
THRESH="${4:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="models/deep/cnn_final_L150_seed42/cnn_best.pt"

OUT_ENC="data/dl/${RUN}"
OUT_PRED="data/predictions/${RUN}"
OUT_FILT="data/filtered_reads/${RUN}"
mkdir -p "$OUT_ENC" "$OUT_PRED" "$OUT_FILT"

[[ -f "$MODEL" ]] || { echo "[ERROR] Missing CNN model: $MODEL" >&2; exit 1; }
[[ -f "$R1" ]] || { echo "[ERROR] Missing R1: $R1" >&2; exit 1; }
[[ -f "$R2" ]] || { echo "[ERROR] Missing R2: $R2" >&2; exit 1; }

echo "[1/3] Encode R1 -> NPZ"
python3 "${SCRIPT_DIR}/dl_encode_fastq_R1_L150_4ch.py" \
  --r1 "$R1" \
  --out "${OUT_ENC}/${RUN}.npz" \
  --out-ids "${OUT_ENC}/${RUN}.ids.txt" \
  --read-len 150

echo "[2/3] CNN inference"
PYTHONPATH=src python3 "${SCRIPT_DIR}/dl_predict_cnn.py" \
  --npz "${OUT_ENC}/${RUN}.npz" \
  --ids "${OUT_ENC}/${RUN}.ids.txt" \
  --model "$MODEL" \
  --out "${OUT_PRED}/${RUN}.cnn.pred.tsv" \
  --remove-ids "${OUT_PRED}/${RUN}.cnn.remove_ids_raw.txt" \
  --threshold "$THRESH"

echo "[3/3] Pair-safe filter mates"
python3 "${SCRIPT_DIR}/filter_pairs_by_ids.py" \
  --r1 "$R1" --r2 "$R2" \
  --remove-ids "${OUT_PRED}/${RUN}.cnn.remove_ids_raw.txt" \
  --out-r1 "${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz" \
  --out-r2 "${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz"

echo "[DONE] CNN filtered:"
echo "  ${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz"
echo "  ${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz"