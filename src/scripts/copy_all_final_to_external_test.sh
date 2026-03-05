#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/external_test

PCTS=(5 10 15 50)

copy_set () {
  local N="$1"       # 10K, 20K, 3200
  local SRC="$2"     # Desktop folder
  for PCT in "${PCTS[@]}"; do
    for MATE in 1 2; do
      IN="$HOME/Desktop/${SRC}/${N}_lemuru${PCT}_R${MATE}.fastq.gz"
      OUT="data/external_test/${N}_final_${PCT}_R${MATE}.fastq.gz"
      [[ -f "$IN" ]] || { echo "[ERROR] Missing: $IN" >&2; exit 1; }
      [[ -f "$OUT" ]] && { echo "[ERROR] Already exists (won't overwrite): $OUT" >&2; exit 1; }
      cp -v "$IN" "$OUT"
    done
  done
}

copy_set "10K"  "10K_reads"
copy_set "20K"  "20K_reads"
copy_set "3200" "3200_reads"

echo ""
echo "[DONE] Copied:"
ls -lh data/external_test/*_final_*_R*.fastq.gz | head -n 50
