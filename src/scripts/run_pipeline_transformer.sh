#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_transformer.sh <R1.fastq(.gz)> <R2.fastq(.gz)> <RUN_NAME> [THRESH=0.5]
#
# Example:
#   bash src/scripts/run_pipeline_transformer.sh \
#     data/external_test/ext5_mixed_R1.fastq.gz \
#     data/external_test/ext5_mixed_R2.fastq.gz \
#     ext5_trf \
#     0.5

R1="$1"
R2="$2"
RUN="$3"
THRESH="${4:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

# ---- MODEL (update if your folder differs) ----
MODEL="models_dl_PAIR_L150_transformer/transformer_best.pt"

# ---- Tokenization params (must match training) ----
READ_LEN=150
K=6
L_KMERS=256

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
echo "[INFO] READ_LEN=$READ_LEN K=$K L_KMERS=$L_KMERS"

[[ -f "$MODEL" ]] || { echo "[ERROR] Missing Transformer model: $MODEL" >&2; exit 1; }
[[ -f "$R1" ]] || { echo "[ERROR] Missing R1: $R1" >&2; exit 1; }
[[ -f "$R2" ]] || { echo "[ERROR] Missing R2: $R2" >&2; exit 1; }

# ---- Step 1: Encode (R1-only -> k-mer tokens, T=256) ----
echo "[1/4] Encode R1 -> NPZ (expects output shape N x ${L_KMERS})"
python "${SCRIPT_DIR}/dl_encode_fastq_R1_L150_k6_L256.py" \
  --r1 "$R1" \
  --out "${OUT_ENC}/${RUN}.kmers.npz" \
  --out-ids "${OUT_ENC}/${RUN}.ids.txt" \
  --read-len "$READ_LEN" \
  --k "$K" \
  --L-kmers "$L_KMERS"

# ---- Step 2: Transformer inference ----
echo "[2/4] Transformer inference"
PYTHONPATH=src python "${SCRIPT_DIR}/dl_predict_transformer.py" \
  --npz "${OUT_ENC}/${RUN}.kmers.npz" \
  --ids "${OUT_ENC}/${RUN}.ids.txt" \
  --model "$MODEL" \
  --out "${OUT_PRED}/${RUN}.trf.pred.tsv" \
  --remove-ids "${OUT_PRED}/${RUN}.trf.remove_ids_raw.txt" \
  --threshold "$THRESH" \
  --k "$K" \
  --L-kmers "$L_KMERS"

# ---- Step 3: Expand base IDs -> /1 and /2 ----
echo "[3/4] Expand IDs to /1 and /2"
python "${SCRIPT_DIR}/make_remove_ids_pairs.py" \
  "${OUT_PRED}/${RUN}.trf.remove_ids_raw.txt" \
  "${OUT_PRED}/${RUN}.trf.remove_ids_for_fastq.txt"

# ---- Step 4: Filter both mates ----
echo "[4/4] Filter reads (remove predicted chimeras)"
seqkit grep -v -f "${OUT_PRED}/${RUN}.trf.remove_ids_for_fastq.txt" \
  "$R1" -o "${OUT_FILT}/${RUN}.trf.filtered_R1.fastq.gz"

seqkit grep -v -f "${OUT_PRED}/${RUN}.trf.remove_ids_for_fastq.txt" \
  "$R2" -o "${OUT_FILT}/${RUN}.trf.filtered_R2.fastq.gz"

echo "[DONE] Outputs:"
echo "  Encoded:   ${OUT_ENC}/${RUN}.kmers.npz"
echo "  IDs:       ${OUT_ENC}/${RUN}.ids.txt"
echo "  Pred:      ${OUT_PRED}/${RUN}.trf.pred.tsv"
echo "  RemoveRaw: ${OUT_PRED}/${RUN}.trf.remove_ids_raw.txt"
echo "  RemoveFQ:  ${OUT_PRED}/${RUN}.trf.remove_ids_for_fastq.txt"
echo "  Filtered:  ${OUT_FILT}/${RUN}.trf.filtered_R1.fastq.gz"
echo "             ${OUT_FILT}/${RUN}.trf.filtered_R2.fastq.gz"