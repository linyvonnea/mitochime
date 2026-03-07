#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_cnn.sh <R1> <R2> <RUN_NAME> [THRESH=0.5]
#
# Optional env:
#   CNN_MODEL=/path/to/cnn_final.pt
#   OVERWRITE=1
#   CHECK_SAMPLE=200

R1="$1"
R2="$2"
RUN="$3"
THRESH="${4:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DEFAULT="models/deep/cnn_final_L150_seed42_fixedep25/cnn_final.pt"
MODEL="${CNN_MODEL:-$MODEL_DEFAULT}"

OVERWRITE="${OVERWRITE:-0}"
CHECK_SAMPLE="${CHECK_SAMPLE:-200}"

OUT_ENC="data/dl/${RUN}"
OUT_PRED="data/predictions/${RUN}"
OUT_FILT="data/filtered_reads/${RUN}"

die() { echo "[ERROR] $*" >&2; exit 1; }

[[ -f "$MODEL" ]] || die "Missing CNN model: $MODEL"
[[ -f "$R1" ]] || die "Missing R1: $R1"
[[ -f "$R2" ]] || die "Missing R2: $R2"

if [[ "$OVERWRITE" == "1" ]]; then
  echo "[OVERWRITE] removing old outputs for RUN=$RUN"
  rm -rf "$OUT_ENC" "$OUT_PRED" "$OUT_FILT"
fi

mkdir -p "$OUT_ENC" "$OUT_PRED" "$OUT_FILT"

echo "[INFO] RUN=$RUN"
echo "[INFO] THRESH=$THRESH"
echo "[INFO] MODEL=$MODEL"

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

echo "[CHECK] Counting reads after filtering"
R1N=$(seqkit stats "${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz" | awk 'NR==2{gsub(",","",$4); print $4}')
R2N=$(seqkit stats "${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz" | awk 'NR==2{gsub(",","",$4); print $4}')

if [[ "$R1N" != "$R2N" ]]; then
  die "Filtered pair counts mismatch: R1=$R1N vs R2=$R2N (pairing is broken!)"
fi
echo "[OK] Filtered pairs: $R1N read-pairs kept"

if [[ "$CHECK_SAMPLE" != "0" ]]; then
  echo "[CHECK] Sampling $CHECK_SAMPLE headers to verify pair order (R1 vs R2)"
  H1=$(gzip -cd "${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz" \
    | awk -v n="$CHECK_SAMPLE" 'NR%4==1 {print; c++; if (c>=n) exit}' \
    | sed -E 's/\/1$//; s/\/2$//; s/[[:space:]].*$//')
  H2=$(gzip -cd "${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz" \
    | awk -v n="$CHECK_SAMPLE" 'NR%4==1 {print; c++; if (c>=n) exit}' \
    | sed -E 's/\/1$//; s/\/2$//; s/[[:space:]].*$//')

  if ! diff -q <(echo "$H1") <(echo "$H2") >/dev/null; then
    die "Header pairing check failed: R1 and R2 IDs diverge (order mismatch or filtering bug)."
  fi
  echo "[OK] Header pairing check passed"
fi

echo "[DONE] CNN filtered:"
echo "  ${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz"
echo "  ${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz"