#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_cnn.sh <R1.fastq(.gz)> <R2.fastq(.gz)> <RUN_NAME> [THRESH=0.5]
#
# Example:
#   bash src/scripts/run_pipeline_cnn.sh \
#     data/external_test/ext5_mixed_R1.fastq.gz \
#     data/external_test/ext5_mixed_R2.fastq.gz \
#     ext5_cnn \
#     0.5

R1="$1"
R2="$2"
RUN="$3"
THRESH="${4:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="models_dl_PAIR_L150/cnn_best.pt"

OUT_ENC="data/dl/${RUN}"
OUT_PRED="data/predictions/${RUN}"
OUT_FILT="data/filtered_reads/${RUN}"

mkdir -p "$OUT_ENC" "$OUT_PRED" "$OUT_FILT"

echo "[INFO] ROOT=$ROOT_DIR"
echo "[INFO] MODEL=$MODEL"
echo "[INFO] R1=$R1"
echo "[INFO] R2=$R2"
echo "[INFO] RUN=$RUN"
echo "[INFO] THRESH=$THRESH"

[[ -f "$MODEL" ]] || { echo "[ERROR] Missing CNN model: $MODEL" >&2; exit 1; }
[[ -f "$R1" ]] || { echo "[ERROR] Missing R1: $R1" >&2; exit 1; }
[[ -f "$R2" ]] || { echo "[ERROR] Missing R2: $R2" >&2; exit 1; }

# ---- Step 1: Encode (R1-only, 4ch, L=150) ----
echo "[1/4] Encode R1 -> NPZ (expects output shape N x 4 x 150)"
python "${SCRIPT_DIR}/dl_encode_fastq_R1_L150_4ch.py" \
  --r1 "$R1" \
  --out "${OUT_ENC}/${RUN}.npz" \
  --out-ids "${OUT_ENC}/${RUN}.ids.txt" \
  --read-len 150

# ---- Step 2: CNN inference ----
echo "[2/4] CNN inference"
PYTHONPATH=src python "${SCRIPT_DIR}/dl_predict_cnn.py" \
  --npz "${OUT_ENC}/${RUN}.npz" \
  --ids "${OUT_ENC}/${RUN}.ids.txt" \
  --model "$MODEL" \
  --out "${OUT_PRED}/${RUN}.cnn.pred.tsv" \
  --remove-ids "${OUT_PRED}/${RUN}.cnn.remove_ids_raw.txt" \
  --threshold "$THRESH"

# ---- Step 3: Expand base IDs -> /1 and /2 (to match FASTQ headers) ----
echo "[3/4] Expand IDs to /1 and /2"
python "${SCRIPT_DIR}/make_remove_ids_pairs.py" \
  "${OUT_PRED}/${RUN}.cnn.remove_ids_raw.txt" \
  "${OUT_PRED}/${RUN}.cnn.remove_ids_for_fastq.txt"

# ---- Step 4: Filter both mates ----
echo "[4/4] Filter reads (remove predicted chimeras)"
seqkit grep -v -f "${OUT_PRED}/${RUN}.cnn.remove_ids_for_fastq.txt" \
  "$R1" -o "${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz"

seqkit grep -v -f "${OUT_PRED}/${RUN}.cnn.remove_ids_for_fastq.txt" \
  "$R2" -o "${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz"

echo "[DONE] Outputs:"
echo "  Encoded:   ${OUT_ENC}/${RUN}.npz"
echo "  IDs:       ${OUT_ENC}/${RUN}.ids.txt"
echo "  Pred:      ${OUT_PRED}/${RUN}.cnn.pred.tsv"
echo "  RemoveRaw: ${OUT_PRED}/${RUN}.cnn.remove_ids_raw.txt"
echo "  RemoveFQ:  ${OUT_PRED}/${RUN}.cnn.remove_ids_for_fastq.txt"
echo "  Filtered:  ${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz"
echo "             ${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz"