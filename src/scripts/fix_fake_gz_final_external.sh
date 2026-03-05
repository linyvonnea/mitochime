#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

for f in data/external_test/*_final_*_R*.fastq.gz; do
  if gzip -t "$f" >/dev/null 2>&1; then
    continue
  fi

  echo "[FIX] $f is not gzipped -> converting to real .gz"
  tmp="${f}.tmp"
  mv "$f" "$tmp"
  gzip -c "$tmp" > "$f"
  rm -f "$tmp"
done

echo "[DONE] Conversion complete."
