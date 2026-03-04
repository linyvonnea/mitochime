#!/usr/bin/env bash
set -euo pipefail

# =========================
# Helpers
# =========================
prompt() {
  local var="$1"
  local msg="$2"
  local def="${3:-}"
  local ans
  if [[ -n "${def}" ]]; then
    read -r -p "${msg} [${def}]: " ans
    ans="${ans:-$def}"
  else
    read -r -p "${msg}: " ans
  fi
  printf -v "$var" "%s" "$ans"
}

require_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "[ERROR] File not found: $f" >&2; exit 1; }
}

require_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || { echo "[ERROR] Missing command in PATH: $c" >&2; exit 1; }
}

to_abs() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# -------- FASTQ gzip helper (optional) --------
is_gz() { [[ "$1" =~ \.gz$ ]]; }

ensure_gz_fastq() {
  local in="$1"
  require_file "$in"
  if is_gz "$in"; then
    echo "$in"
    return 0
  fi
  echo "[INFO] Input is not .gz, compressing: $in"
  local out="${in}.gz"
  gzip -c "$in" > "$out"
  echo "$out"
}

# -------- Assembly (SPAdes) --------
run_spades() {
  local r1="$1"
  local r2="$2"
  local outdir="$3"
  local threads="$4"
  local klist="$5"   # e.g. "21,33,55,77" or "115"
  local env_spades="${6:-}"

  mkdir -p "$outdir"

  if [[ -n "$env_spades" ]]; then
    conda run -n "$env_spades" spades.py \
      -1 "$r1" -2 "$r2" -o "$outdir" -t "$threads" -k "$klist"
  else
    spades.py -1 "$r1" -2 "$r2" -o "$outdir" -t "$threads" -k "$klist"
  fi
}

# =========================
# CLI
# =========================
echo "=== MitoChime CLI Pipeline ==="

prompt R1 "Enter R1 FASTQ/FASTQ.GZ path"
prompt R2 "Enter R2 FASTQ/FASTQ.GZ path"
R1="$(to_abs "$R1")"
R2="$(to_abs "$R2")"
require_file "$R1"
require_file "$R2"

prompt AUTO_GZ "If inputs are not .gz, auto-gzip them? (y/n)" "y"
if [[ "$AUTO_GZ" =~ ^[Yy]$ ]]; then
  R1="$(ensure_gz_fastq "$R1")"
  R2="$(ensure_gz_fastq "$R2")"
fi

prompt RUN "Run name (used for output folders)" "run1"
prompt THRESH "Chimera threshold (0-1; higher = stricter removal)" "0.5"
prompt THREADS "Threads" "8"

echo ""
echo "Choose filtering mode:"
echo "  1) GB only"
echo "  2) CNN only"
echo "  3) BOTH independent (GB + CNN on original reads)"
echo "  4) GB -> CNN sequential (CNN runs on GB-filtered reads)"
prompt CHOICE "Enter 1/2/3/4" "1"

prompt DO_ASSEMBLY "Run SPAdes assemblies too? (y/n)" "y"
prompt SPADES_K "SPAdes k list (comma-separated). Example: 21,33,55,77 or single like 115" "115"
prompt SPADES_ENV "If SPAdes is in a separate conda env, enter env name; else leave blank" ""

# =========================
# Reference selection (only needed when GB is involved)
# =========================
DEFAULT_REF="data/refs/original.fasta"
REF="$DEFAULT_REF"

needs_gb="n"
case "$CHOICE" in
  1|3|4) needs_gb="y" ;;
esac

if [[ "$needs_gb" == "y" ]]; then
  echo ""
  echo "GB model requires mapping to a reference genome (minimap2)."
  echo "Default reference (trained/validated): $DEFAULT_REF"
  echo ""
  echo "[WARNING] Using a different reference/species can shift alignment/SA/clipping features"
  echo "and may degrade performance. Ideally retrain/validate per reference/species."
  echo ""

  prompt USE_DEFAULT_REF "Use default Sardinella lemuru reference? (y/n)" "y"
  if [[ "$USE_DEFAULT_REF" =~ ^[Yy]$ ]]; then
    REF="$DEFAULT_REF"
  else
    prompt REF_IN "Enter path/to/ref.fasta"
    REF="$(to_abs "$REF_IN")"
  fi
  require_file "$REF"
fi

# =========================
# Scripts
# =========================
GB_SCRIPT="src/scripts/run_pipeline_gb.sh"
CNN_SCRIPT="src/scripts/run_pipeline_cnn.sh"

GB_OUTDIR="data/filtered_reads/${RUN}_gb"
CNN_OUTDIR="data/filtered_reads/${RUN}_cnn"
GBCNN_OUTDIR="data/filtered_reads/${RUN}_gbcnn"

GB_R1_OUT="${GB_OUTDIR}/${RUN}_gb.gb.filtered_R1.fastq.gz"
GB_R2_OUT="${GB_OUTDIR}/${RUN}_gb.gb.filtered_R2.fastq.gz"

CNN_R1_OUT="${CNN_OUTDIR}/${RUN}_cnn.cnn.filtered_R1.fastq.gz"
CNN_R2_OUT="${CNN_OUTDIR}/${RUN}_cnn.cnn.filtered_R2.fastq.gz"

GBCNN_R1_OUT="${GBCNN_OUTDIR}/${RUN}_gbcnn.cnn.filtered_R1.fastq.gz"
GBCNN_R2_OUT="${GBCNN_OUTDIR}/${RUN}_gbcnn.cnn.filtered_R2.fastq.gz"

run_gb() {
  local in1="$1"
  local in2="$2"
  local runname="$3"
  require_file "$GB_SCRIPT"
  chmod +x "$GB_SCRIPT" >/dev/null 2>&1 || true
  echo ""
  echo "[STEP] Running GB filter..."
  # args: R1 R2 RUN THRESH THREADS REF
  bash "$GB_SCRIPT" "$in1" "$in2" "$runname" "$THRESH" "$THREADS" "$REF"
}

run_cnn() {
  local in1="$1"
  local in2="$2"
  local runname="$3"
  require_file "$CNN_SCRIPT"
  chmod +x "$CNN_SCRIPT" >/dev/null 2>&1 || true
  echo ""
  echo "[STEP] Running CNN filter..."
  bash "$CNN_SCRIPT" "$in1" "$in2" "$runname" "$THRESH"
}

# =========================
# Run filtering
# =========================
FINAL_R1=""
FINAL_R2=""

case "$CHOICE" in
  1)
    run_gb "$R1" "$R2" "$RUN"
    require_file "$GB_R1_OUT"; require_file "$GB_R2_OUT"
    FINAL_R1="$GB_R1_OUT"; FINAL_R2="$GB_R2_OUT"
    ;;

  2)
    run_cnn "$R1" "$R2" "${RUN}_cnn"
    require_file "$CNN_R1_OUT"; require_file "$CNN_R2_OUT"
    FINAL_R1="$CNN_R1_OUT"; FINAL_R2="$CNN_R2_OUT"
    ;;

  3)
    # BOTH independent
    run_gb "$R1" "$R2" "$RUN"
    require_file "$GB_R1_OUT"; require_file "$GB_R2_OUT"

    run_cnn "$R1" "$R2" "${RUN}_cnn"
    require_file "$CNN_R1_OUT"; require_file "$CNN_R2_OUT"

    echo ""
    echo "[DONE] Independent filtered outputs created:"
    echo "  GB   R1: $GB_R1_OUT"
    echo "  GB   R2: $GB_R2_OUT"
    echo "  CNN  R1: $CNN_R1_OUT"
    echo "  CNN  R2: $CNN_R2_OUT"

    # FINAL is not meaningful here; keep GB for display only
    FINAL_R1="$GB_R1_OUT"; FINAL_R2="$GB_R2_OUT"
    ;;

  4)
    # sequential GB -> CNN
    run_gb "$R1" "$R2" "$RUN"
    require_file "$GB_R1_OUT"; require_file "$GB_R2_OUT"

    run_cnn "$GB_R1_OUT" "$GB_R2_OUT" "${RUN}_gbcnn"
    require_file "$GBCNN_R1_OUT"; require_file "$GBCNN_R2_OUT"

    FINAL_R1="$GBCNN_R1_OUT"; FINAL_R2="$GBCNN_R2_OUT"
    ;;

  *)
    echo "[ERROR] Invalid choice: $CHOICE" >&2
    exit 1
    ;;
esac

echo ""
echo "[FINAL] Filtered FASTQs (single target shown):"
echo "  R1: $FINAL_R1"
echo "  R2: $FINAL_R2"

# =========================
# Assembly (SPAdes)
# =========================
if [[ "$DO_ASSEMBLY" =~ ^[Yy]$ ]]; then
  require_cmd spades.py

  echo ""
  echo "[ASSEMBLY] SPAdes UNFILTERED..."
  UNF_OUT="data/assemblies_spades/${RUN}_unfiltered"
  run_spades "$R1" "$R2" "$UNF_OUT" "$THREADS" "$SPADES_K" "$SPADES_ENV"

  if [[ "$CHOICE" == "3" ]]; then
    # ✅ ALWAYS assemble BOTH GB and CNN filtered
    echo ""
    echo "[ASSEMBLY] SPAdes FILTERED (GB)..."
    GB_ASM_OUT="data/assemblies_spades/${RUN}_gb_filtered"
    run_spades "$GB_R1_OUT" "$GB_R2_OUT" "$GB_ASM_OUT" "$THREADS" "$SPADES_K" "$SPADES_ENV"

    echo ""
    echo "[ASSEMBLY] SPAdes FILTERED (CNN)..."
    CNN_ASM_OUT="data/assemblies_spades/${RUN}_cnn_filtered"
    run_spades "$CNN_R1_OUT" "$CNN_R2_OUT" "$CNN_ASM_OUT" "$THREADS" "$SPADES_K" "$SPADES_ENV"

    echo ""
    echo "[SPAdes outputs]"
    echo "UNFILTERED:"
    echo "  ${UNF_OUT}/contigs.fasta"
    echo "  ${UNF_OUT}/scaffolds.fasta"
    echo "GB FILTERED:"
    echo "  ${GB_ASM_OUT}/contigs.fasta"
    echo "  ${GB_ASM_OUT}/scaffolds.fasta"
    echo "CNN FILTERED:"
    echo "  ${CNN_ASM_OUT}/contigs.fasta"
    echo "  ${CNN_ASM_OUT}/scaffolds.fasta"

  else
    echo ""
    echo "[ASSEMBLY] SPAdes FILTERED..."
    FIL_OUT="data/assemblies_spades/${RUN}_filtered"
    run_spades "$FINAL_R1" "$FINAL_R2" "$FIL_OUT" "$THREADS" "$SPADES_K" "$SPADES_ENV"

    echo ""
    echo "[SPAdes outputs]"
    echo "UNFILTERED:"
    echo "  ${UNF_OUT}/contigs.fasta"
    echo "  ${UNF_OUT}/scaffolds.fasta"
    echo "FILTERED:"
    echo "  ${FIL_OUT}/contigs.fasta"
    echo "  ${FIL_OUT}/scaffolds.fasta"
  fi
fi

echo ""
echo "[DONE] Pipeline finished."