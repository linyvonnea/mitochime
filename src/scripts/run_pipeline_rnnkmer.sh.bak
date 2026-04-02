#!/usr/bin/env bash
set -euo pipefail

# src/scripts/run_mitochime_cli.sh
#
# Interactive CLI to run MitoChime filtering pipelines:
#   1) GB (reference-guided)
#   2) CNN (reference-free)
#   3) BOTH (GB + CNN)
#   4) BiGRU-kmer (reference-free)
#   5) ALL (GB + CNN + BiGRU-kmer)
#
# Optional: run SPAdes assemblies on unfiltered + filtered outputs.
#
# Assumes these scripts exist:
#   src/scripts/run_pipeline_gb.sh
#   src/scripts/run_pipeline_cnn.sh
#   src/scripts/run_pipeline_rnnkmer.sh
#   src/scripts/run_spades.sh (optional if you choose SPAdes)

prompt() {
  local var="$1" msg="$2" def="${3:-}" ans
  if [[ -n "$def" ]]; then
    read -r -p "${msg} [${def}]: " ans
    ans="${ans:-$def}"
  else
    read -r -p "${msg}: " ans
  fi
  printf -v "$var" "%s" "$ans"
}

require_file(){ [[ -f "$1" ]] || { echo "[ERROR] File not found: $1" >&2; exit 1; }; }
require_cmd(){ command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }; }
to_abs(){ python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

echo "=== MitoChime CLI (GB + CNN + BiGRU-kmer) ==="

require_cmd python3
require_cmd bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

# --- collect inputs
prompt R1 "Enter R1 FASTQ/FASTQ.GZ path"
prompt R2 "Enter R2 FASTQ/FASTQ.GZ path"
R1="$(to_abs "$R1")"
R2="$(to_abs "$R2")"
require_file "$R1"
require_file "$R2"

prompt RUN "Base run name" "run1"
prompt THRESH "Chimera threshold (0-1)" "0.5"
prompt THREADS "Threads (GB mapping/assembly)" "8"

echo ""
echo "Mode:"
echo "  1) GB only"
echo "  2) CNN only"
echo "  3) BOTH (GB + CNN on original reads)"
echo "  4) BiGRU-kmer only"
echo "  5) ALL (GB + CNN + BiGRU-kmer)"
prompt MODE "Enter 1/2/3/4/5" "3"

REF=""
if [[ "$MODE" == "1" || "$MODE" == "3" || "$MODE" == "5" ]]; then
  prompt REF "GB reference FASTA" "data/refs/original.fasta"
  REF="$(to_abs "$REF")"
  require_file "$REF"
fi

prompt DO_SPADES "Run SPAdes assembly (y/n)" "n"
prompt SPADES_ENV "SPAdes conda env name (blank if in PATH)" ""

# --- scripts
GB_SCRIPT="src/scripts/run_pipeline_gb.sh"
CNN_SCRIPT="src/scripts/run_pipeline_cnn.sh"
RNNKMER_SCRIPT="src/scripts/run_pipeline_rnnkmer.sh"
SPADES_SCRIPT="src/scripts/run_spades.sh"

require_file "$GB_SCRIPT"
require_file "$CNN_SCRIPT"
require_file "$RNNKMER_SCRIPT"
chmod +x "$GB_SCRIPT" "$CNN_SCRIPT" "$RNNKMER_SCRIPT" >/dev/null 2>&1 || true
[[ -f "$SPADES_SCRIPT" ]] && chmod +x "$SPADES_SCRIPT" >/dev/null 2>&1 || true

# --- derived run names
GB_RUN="${RUN}_gb"
CNN_RUN="${RUN}_cnn"
RNN_RUN="${RUN}_bigru"

# --- expected outputs
GB_R1_OUT="data/filtered_reads/${GB_RUN}/${GB_RUN}.gb.filtered_R1.fastq.gz"
GB_R2_OUT="data/filtered_reads/${GB_RUN}/${GB_RUN}.gb.filtered_R2.fastq.gz"

CNN_R1_OUT="data/filtered_reads/${CNN_RUN}/${CNN_RUN}.cnn.filtered_R1.fastq.gz"
CNN_R2_OUT="data/filtered_reads/${CNN_RUN}/${CNN_RUN}.cnn.filtered_R2.fastq.gz"

RNN_R1_OUT="data/filtered_reads/${RNN_RUN}/${RNN_RUN}.bigru.filtered_R1.fastq.gz"
RNN_R2_OUT="data/filtered_reads/${RNN_RUN}/${RNN_RUN}.bigru.filtered_R2.fastq.gz"

# --- run pipelines
if [[ "$MODE" == "1" || "$MODE" == "3" || "$MODE" == "5" ]]; then
  echo ""
  echo "[GB] Running..."
  bash "$GB_SCRIPT" "$R1" "$R2" "$GB_RUN" "$THRESH" "$THREADS" "$REF"
  require_file "$GB_R1_OUT"
  require_file "$GB_R2_OUT"
fi

if [[ "$MODE" == "2" || "$MODE" == "3" || "$MODE" == "5" ]]; then
  echo ""
  echo "[CNN] Running..."
  bash "$CNN_SCRIPT" "$R1" "$R2" "$CNN_RUN" "$THRESH"
  require_file "$CNN_R1_OUT"
  require_file "$CNN_R2_OUT"
fi

if [[ "$MODE" == "4" || "$MODE" == "5" ]]; then
  echo ""
  echo "[BiGRU-kmer] Running..."
  bash "$RNNKMER_SCRIPT" "$R1" "$R2" "$RNN_RUN" "$THRESH"
  require_file "$RNN_R1_OUT"
  require_file "$RNN_R2_OUT"
fi

echo ""
echo "[DONE] Filtered outputs:"
if [[ -f "$GB_R1_OUT" ]]; then
  echo "  GB     R1: $GB_R1_OUT"
  echo "  GB     R2: $GB_R2_OUT"
fi
if [[ -f "$CNN_R1_OUT" ]]; then
  echo "  CNN    R1: $CNN_R1_OUT"
  echo "  CNN    R2: $CNN_R2_OUT"
fi
if [[ -f "$RNN_R1_OUT" ]]; then
  echo "  BiGRU  R1: $RNN_R1_OUT"
  echo "  BiGRU  R2: $RNN_R2_OUT"
fi

# --- optional SPAdes
if [[ "$DO_SPADES" =~ ^[Yy]$ ]]; then
  [[ -f "$SPADES_SCRIPT" ]] || { echo "[ERROR] Missing $SPADES_SCRIPT" >&2; exit 1; }

  echo ""
  echo "[SPAdes] UNFILTERED..."
  bash "$SPADES_SCRIPT" "$R1" "$R2" "data/assemblies_spades/${RUN}_unfiltered/spades" "$THREADS" "$SPADES_ENV"

  if [[ -f "$GB_R1_OUT" ]]; then
    echo ""
    echo "[SPAdes] GB FILTERED..."
    bash "$SPADES_SCRIPT" "$GB_R1_OUT" "$GB_R2_OUT" "data/assemblies_spades/${RUN}_gb_filtered/spades" "$THREADS" "$SPADES_ENV"
  fi

  if [[ -f "$CNN_R1_OUT" ]]; then
    echo ""
    echo "[SPAdes] CNN FILTERED..."
    bash "$SPADES_SCRIPT" "$CNN_R1_OUT" "$CNN_R2_OUT" "data/assemblies_spades/${RUN}_cnn_filtered/spades" "$THREADS" "$SPADES_ENV"
  fi

  if [[ -f "$RNN_R1_OUT" ]]; then
    echo ""
    echo "[SPAdes] BiGRU-kmer FILTERED..."
    bash "$SPADES_SCRIPT" "$RNN_R1_OUT" "$RNN_R2_OUT" "data/assemblies_spades/${RUN}_bigru_filtered/spades" "$THREADS" "$SPADES_ENV"
  fi
fi

echo ""
echo "[OK] Pipeline finished."