#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/external_test

copy_final_set () {
  local READCOUNT="$1"   # 10K / 20K / 3200
  local SRC_DIR="$2"     # e.g. /Users/.../Desktop/10K_reads

  for PCT in 5 10 15 50; do
    cp -v "${SRC_DIR}/${READCOUNT}_lemuru${PCT}_R1.fastq.gz" "data/external_test/${READCOUNT}_final_${PCT}_R1.fastq.gz"
    cp -v "${SRC_DIR}/${READCOUNT}_lemuru${PCT}_R2.fastq.gz" "data/external_test/${READCOUNT}_final_${PCT}_R2.fastq.gz"
  done
}

copy_final_set "10K"  "$HOME/Desktop/10K_reads"
copy_final_set "20K"  "$HOME/Desktop/20K_reads"
copy_final_set "3200" "$HOME/Desktop/3200_reads"

echo ""
echo "[OK] Copied final datasets:"
ls -lh data/external_test/*_final_*_R*.fastq.gz | head -n 60
