#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_cnn.sh <R1> <R2> <RUN_NAME> [THRESH=0.5]
#
# Optional env:
#   CNN_MODEL=/path/to/cnn_final.pt
#   OVERWRITE=1
#   CHECK_SAMPLE=200
#
# CNN is reference-free.
# It predicts from R1 sequence encoding, then removes both mates pair-safely.

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

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

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

NPZ="${OUT_ENC}/${RUN}.npz"
IDS="${OUT_ENC}/${RUN}.ids.txt"
PRED="${OUT_PRED}/${RUN}.cnn.pred.tsv"
REMOVE_IDS="${OUT_PRED}/${RUN}.cnn.remove_ids_raw.txt"
FILT_R1="${OUT_FILT}/${RUN}.cnn.filtered_R1.fastq.gz"
FILT_R2="${OUT_FILT}/${RUN}.cnn.filtered_R2.fastq.gz"

echo "[1/3] Encode R1 -> NPZ"
python3 "${SCRIPT_DIR}/dl_encode_fastq_R1_L150_4ch.py" \
  --r1 "$R1" \
  --out "$NPZ" \
  --out-ids "$IDS" \
  --read-len 150

echo "[2/3] CNN inference"
PYTHONPATH=src python3 "${SCRIPT_DIR}/dl_predict_cnn.py" \
  --npz "$NPZ" \
  --ids "$IDS" \
  --model "$MODEL" \
  --out "$PRED" \
  --remove-ids "$REMOVE_IDS" \
  --threshold "$THRESH"

echo "[3/3] Pair-safe filter mates"
python3 "${SCRIPT_DIR}/filter_pairs_by_ids.py" \
  --r1 "$R1" \
  --r2 "$R2" \
  --remove-ids "$REMOVE_IDS" \
  --out-r1 "$FILT_R1" \
  --out-r2 "$FILT_R2"

echo "[CHECK] Counting reads after filtering"
R1N=$(seqkit stats "$FILT_R1" | awk 'NR==2{gsub(",","",$4); print $4}')
R2N=$(seqkit stats "$FILT_R2" | awk 'NR==2{gsub(",","",$4); print $4}')

if [[ "$R1N" != "$R2N" ]]; then
  die "Filtered pair counts mismatch: R1=$R1N vs R2=$R2N"
fi

echo "[OK] Filtered pairs: $R1N read-pairs kept"

if [[ "$CHECK_SAMPLE" != "0" ]]; then
  echo "[CHECK] Sampling $CHECK_SAMPLE headers to verify pair order (R1 vs R2)"

  set +e

  H1=$(gzip -cd "$FILT_R1" \
    | awk -v n="$CHECK_SAMPLE" 'NR%4==1 {print; c++; if (c>=n) exit}' \
    | sed -E 's/\/1$//; s/\/2$//; s/[[:space:]].*$//')
  status1=$?

  H2=$(gzip -cd "$FILT_R2" \
    | awk -v n="$CHECK_SAMPLE" 'NR%4==1 {print; c++; if (c>=n) exit}' \
    | sed -E 's/\/1$//; s/\/2$//; s/[[:space:]].*$//')
  status2=$?

  set -e

  if [[ $status1 -ne 0 || $status2 -ne 0 ]]; then
    echo "[WARN] Header sampling check skipped due to pipeline exit status."
  elif ! diff -q <(echo "$H1") <(echo "$H2") >/dev/null; then
    die "Header pairing check failed: R1 and R2 IDs diverge."
  else
    echo "[OK] Header pairing check passed"
  fi
fi

echo "[DONE] CNN filtered:"
echo "  $FILT_R1"
echo "  $FILT_R2"