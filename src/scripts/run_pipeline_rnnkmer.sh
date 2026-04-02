#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/scripts/run_pipeline_rnnkmer.sh <R1> <R2> <RUN_NAME> [THRESH=0.5]
#
# Optional env:
#   RNN_MODEL=/path/to/rnn_kmer_gru_best.pt
#   OVERWRITE=1
#   CHECK_SAMPLE=0

R1="$1"
R2="$2"
RUN="$3"
THRESH="${4:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DEFAULT="models/deep/rnnkmer_bigru_final_L150_seed42/rnn_kmer_gru_best.pt"
MODEL="${RNN_MODEL:-$MODEL_DEFAULT}"

OVERWRITE="${OVERWRITE:-0}"
CHECK_SAMPLE="${CHECK_SAMPLE:-0}"

OUT_ENC="data/dl/${RUN}"
OUT_PRED="data/predictions/${RUN}"
OUT_FILT="data/filtered_reads/${RUN}"

die() { echo "[ERROR] $*" >&2; exit 1; }

[[ -f "$MODEL" ]] || die "Missing BiGRU model: $MODEL"
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

SEQ_TSV="${OUT_ENC}/${RUN}.seq.tsv"

echo "[1/3] Make sequence TSV for inference"
PYTHONPATH=src python3 -m mitochime.deep_learning.make_seq_tsv_infer \
  --r1 "$R1" \
  --r2 "$R2" \
  --L 150 \
  --out "$SEQ_TSV"

echo "[2/3] BiGRU-kmer inference"
PYTHONPATH=src python3 -m mitochime.deep_learning.predict_deep \
  --tsv "$SEQ_TSV" \
  --ckpt "$MODEL" \
  --out "${OUT_PRED}/${RUN}.bigru.pred.tsv" \
  --remove-ids "${OUT_PRED}/${RUN}.bigru.remove_ids_raw.txt" \
  --threshold "$THRESH" \
  --L 150 \
  --k 4 \
  --L-kmers 147 \
  --embed-dim 64 \
  --hidden 256 \
  --rnn-layers 1 \
  --bidirectional \
  --pool last

echo "[3/3] Pair-safe filter mates"
python3 "${SCRIPT_DIR}/filter_pairs_by_ids.py" \
  --r1 "$R1" --r2 "$R2" \
  --remove-ids "${OUT_PRED}/${RUN}.bigru.remove_ids_raw.txt" \
  --out-r1 "${OUT_FILT}/${RUN}.bigru.filtered_R1.fastq.gz" \
  --out-r2 "${OUT_FILT}/${RUN}.bigru.filtered_R2.fastq.gz"

echo "[CHECK] Counting reads after filtering"
R1N=$(seqkit stats "${OUT_FILT}/${RUN}.bigru.filtered_R1.fastq.gz" | awk 'NR==2{gsub(",","",$4); print $4}')
R2N=$(seqkit stats "${OUT_FILT}/${RUN}.bigru.filtered_R2.fastq.gz" | awk 'NR==2{gsub(",","",$4); print $4}')

if [[ "$R1N" != "$R2N" ]]; then
  die "Filtered pair counts mismatch: R1=$R1N vs R2=$R2N"
fi
echo "[OK] Filtered pairs: $R1N read-pairs kept"

if [[ "$CHECK_SAMPLE" != "0" ]]; then
  echo "[CHECK] Sampling $CHECK_SAMPLE headers to verify pair order (R1 vs R2)"
  set +e
  H1=$(gzip -cd "${OUT_FILT}/${RUN}.bigru.filtered_R1.fastq.gz" \
    | awk -v n="$CHECK_SAMPLE" 'NR%4==1 {print; c++; if (c>=n) exit}' \
    | sed -E 's/\/1$//; s/\/2$//; s/[[:space:]].*$//')
  status1=$?

  H2=$(gzip -cd "${OUT_FILT}/${RUN}.bigru.filtered_R2.fastq.gz" \
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

echo "[DONE] BiGRU filtered:"
echo "  ${OUT_FILT}/${RUN}.bigru.filtered_R1.fastq.gz"
echo "  ${OUT_FILT}/${RUN}.bigru.filtered_R2.fastq.gz"