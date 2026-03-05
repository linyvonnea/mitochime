#!/usr/bin/env bash
set -euo pipefail

# Usage: bash src/scripts/run_spades.sh <R1> <R2> <OUTDIR> [THREADS=8] [ENV=""]
R1="$1"
R2="$2"
OUTDIR="$3"
THREADS="${4:-8}"
ENVNAME="${5:-}"

if [[ -z "$ENVNAME" ]]; then
  command -v spades.py >/dev/null 2>&1 || { echo "[ERROR] spades.py not in PATH" >&2; exit 1; }
else
  command -v conda >/dev/null 2>&1 || { echo "[ERROR] conda not in PATH (needed for conda run)" >&2; exit 1; }
fi

[[ -f "$R1" ]] || { echo "[ERROR] R1 not found: $R1" >&2; exit 1; }
[[ -f "$R2" ]] || { echo "[ERROR] R2 not found: $R2" >&2; exit 1; }

if [[ -d "$OUTDIR" ]]; then rm -rf "$OUTDIR"; fi
mkdir -p "$(dirname "$OUTDIR")"

echo "[INFO] SPAdes -> $OUTDIR"

if [[ -n "$ENVNAME" ]]; then
  conda run -n "$ENVNAME" spades.py -1 "$R1" -2 "$R2" -o "$OUTDIR" -t "$THREADS" --careful -k 21,33,55,77,99,115
else
  spades.py -1 "$R1" -2 "$R2" -o "$OUTDIR" -t "$THREADS" --careful -k 21,33,55,77,99,115
fi

echo "[OK] Outputs:"
echo "  $OUTDIR/contigs.fasta"
echo "  $OUTDIR/assembly_graph.gfa"