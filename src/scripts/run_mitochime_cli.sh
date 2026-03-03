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

run_getorganelle() {
  local r1="$1"
  local r2="$2"
  local outdir="$3"
  local threads="$4"
  local env_getorg="${5:-}"

  mkdir -p "$(dirname "$outdir")"

  if [[ -n "$env_getorg" ]]; then
    conda run -n "$env_getorg" get_organelle_from_reads.py \
      -1 "$r1" -2 "$r2" -o "$outdir" -F animal_mt -t "$threads" --overwrite
  else
    get_organelle_from_reads.py \
      -1 "$r1" -2 "$r2" -o "$outdir" -F animal_mt -t "$threads" --overwrite
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

prompt RUN "Run name (used for output folders)" "run1"
prompt THRESH "Chimera threshold (0-1; higher = stricter removal)" "0.5"
prompt THREADS "Threads" "8"

echo ""
echo "Choose filter model:"
echo "  1) GB only"
echo "  2) CNN only"
echo "  3) Transformer only"
echo "  4) ALL independent (GB, CNN, Transformer) on original reads"
echo "  5) GB -> CNN sequential (CNN runs on GB-filtered reads)"
echo "  6) GB -> Transformer sequential (Transformer runs on GB-filtered reads)"
prompt CHOICE "Enter 1/2/3/4/5/6" "1"

prompt DO_ASSEMBLY "Run GetOrganelle assemblies too? (y/n)" "y"
prompt GO_ENV "If GetOrganelle is in a separate conda env, enter env name; else leave blank" ""

# label used for naming filtered assembly folders in single-mode runs
FILTER_LABEL="filtered"
case "$CHOICE" in
  1) FILTER_LABEL="gb" ;;
  2) FILTER_LABEL="cnn" ;;
  3) FILTER_LABEL="trf" ;;
  5) FILTER_LABEL="gbcnn" ;;
  6) FILTER_LABEL="gbtrf" ;;
  4) FILTER_LABEL="multi" ;;
esac

# =========================
# Scripts + expected outputs
# =========================
GB_SCRIPT="src/scripts/run_pipeline_gb.sh"
CNN_SCRIPT="src/scripts/run_pipeline_cnn.sh"
TRF_SCRIPT="src/scripts/run_pipeline_transformer.sh"

GB_OUTDIR="data/filtered_reads/${RUN}_gb"
CNN_OUTDIR="data/filtered_reads/${RUN}_cnn"
TRF_OUTDIR="data/filtered_reads/${RUN}_trf"
GBCNN_OUTDIR="data/filtered_reads/${RUN}_gbcnn"
GBTRF_OUTDIR="data/filtered_reads/${RUN}_gbtrf"

GB_R1_OUT="${GB_OUTDIR}/${RUN}_gb.gb.filtered_R1.fastq.gz"
GB_R2_OUT="${GB_OUTDIR}/${RUN}_gb.gb.filtered_R2.fastq.gz"

CNN_R1_OUT="${CNN_OUTDIR}/${RUN}_cnn.cnn.filtered_R1.fastq.gz"
CNN_R2_OUT="${CNN_OUTDIR}/${RUN}_cnn.cnn.filtered_R2.fastq.gz"

TRF_R1_OUT="${TRF_OUTDIR}/${RUN}_trf.trf.filtered_R1.fastq.gz"
TRF_R2_OUT="${TRF_OUTDIR}/${RUN}_trf.trf.filtered_R2.fastq.gz"

GBCNN_R1_OUT="${GBCNN_OUTDIR}/${RUN}_gbcnn.cnn.filtered_R1.fastq.gz"
GBCNN_R2_OUT="${GBCNN_OUTDIR}/${RUN}_gbcnn.cnn.filtered_R2.fastq.gz"

GBTRF_R1_OUT="${GBTRF_OUTDIR}/${RUN}_gbtrf.trf.filtered_R1.fastq.gz"
GBTRF_R2_OUT="${GBTRF_OUTDIR}/${RUN}_gbtrf.trf.filtered_R2.fastq.gz"

run_gb() {
  local in1="$1"
  local in2="$2"
  local runname="$3"
  require_file "$GB_SCRIPT"
  chmod +x "$GB_SCRIPT" >/dev/null 2>&1 || true
  echo ""
  echo "[STEP] Running GB filter..."
  bash "$GB_SCRIPT" "$in1" "$in2" "$runname" "$THRESH"
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

run_trf() {
  local in1="$1"
  local in2="$2"
  local runname="$3"
  require_file "$TRF_SCRIPT"
  chmod +x "$TRF_SCRIPT" >/dev/null 2>&1 || true
  echo ""
  echo "[STEP] Running Transformer filter..."
  bash "$TRF_SCRIPT" "$in1" "$in2" "$runname" "$THRESH"
}

# =========================
# Run filtering
# =========================
FINAL_R1=""
FINAL_R2=""

ASSEMBLE_GB="n"
ASSEMBLE_CNN="n"
ASSEMBLE_TRF="n"

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
    run_trf "$R1" "$R2" "${RUN}_trf"
    require_file "$TRF_R1_OUT"; require_file "$TRF_R2_OUT"
    FINAL_R1="$TRF_R1_OUT"; FINAL_R2="$TRF_R2_OUT"
    ;;

  4)
    # ALL independent
    run_gb "$R1" "$R2" "$RUN"
    require_file "$GB_R1_OUT"; require_file "$GB_R2_OUT"

    run_cnn "$R1" "$R2" "${RUN}_cnn"
    require_file "$CNN_R1_OUT"; require_file "$CNN_R2_OUT"

    run_trf "$R1" "$R2" "${RUN}_trf"
    require_file "$TRF_R1_OUT"; require_file "$TRF_R2_OUT"

    echo ""
    echo "[DONE] Independent filtered outputs created:"
    echo "  GB   R1: $GB_R1_OUT"
    echo "  GB   R2: $GB_R2_OUT"
    echo "  CNN  R1: $CNN_R1_OUT"
    echo "  CNN  R2: $CNN_R2_OUT"
    echo "  TRF  R1: $TRF_R1_OUT"
    echo "  TRF  R2: $TRF_R2_OUT"

    echo ""
    echo "For the FILTERED assembly step, what do you want?"
    echo "  1) assemble GB filtered only"
    echo "  2) assemble CNN filtered only"
    echo "  3) assemble Transformer filtered only"
    echo "  4) assemble ALL (GB + CNN + TRF)"
    prompt ASSEMBLE_PICK "Enter 1/2/3/4" "4"

    if [[ "$ASSEMBLE_PICK" == "1" ]]; then
      FINAL_R1="$GB_R1_OUT"; FINAL_R2="$GB_R2_OUT"
      ASSEMBLE_GB="y"
    elif [[ "$ASSEMBLE_PICK" == "2" ]]; then
      FINAL_R1="$CNN_R1_OUT"; FINAL_R2="$CNN_R2_OUT"
      ASSEMBLE_CNN="y"
    elif [[ "$ASSEMBLE_PICK" == "3" ]]; then
      FINAL_R1="$TRF_R1_OUT"; FINAL_R2="$TRF_R2_OUT"
      ASSEMBLE_TRF="y"
    else
      ASSEMBLE_GB="y"; ASSEMBLE_CNN="y"; ASSEMBLE_TRF="y"
      FINAL_R1="$GB_R1_OUT"; FINAL_R2="$GB_R2_OUT"
    fi
    ;;

  5)
    # sequential GB -> CNN
    run_gb "$R1" "$R2" "$RUN"
    require_file "$GB_R1_OUT"; require_file "$GB_R2_OUT"

    run_cnn "$GB_R1_OUT" "$GB_R2_OUT" "${RUN}_gbcnn"
    require_file "$GBCNN_R1_OUT"; require_file "$GBCNN_R2_OUT"

    FINAL_R1="$GBCNN_R1_OUT"; FINAL_R2="$GBCNN_R2_OUT"
    ;;

  6)
    # sequential GB -> Transformer
    run_gb "$R1" "$R2" "$RUN"
    require_file "$GB_R1_OUT"; require_file "$GB_R2_OUT"

    run_trf "$GB_R1_OUT" "$GB_R2_OUT" "${RUN}_gbtrf"
    require_file "$GBTRF_R1_OUT"; require_file "$GBTRF_R2_OUT"

    FINAL_R1="$GBTRF_R1_OUT"; FINAL_R2="$GBTRF_R2_OUT"
    ;;

  *)
    echo "[ERROR] Invalid choice: $CHOICE" >&2
    exit 1
    ;;
esac

echo ""
echo "[FINAL] Filtered FASTQs (for a single filtered assembly target):"
echo "  R1: $FINAL_R1"
echo "  R2: $FINAL_R2"

# =========================
# Assembly
# =========================
if [[ "$DO_ASSEMBLY" =~ ^[Yy]$ ]]; then
  require_cmd python3
  require_cmd bash

  echo ""
  echo "[ASSEMBLY] GetOrganelle UNFILTERED..."
  UNF_OUT="data/assemblies/${RUN}_unfiltered_cli/getorganelle_unfiltered"
  run_getorganelle "$R1" "$R2" "$UNF_OUT" "$THREADS" "$GO_ENV"

  if [[ "$CHOICE" == "4" ]]; then
    if [[ "$ASSEMBLE_GB" == "y" ]]; then
      echo ""
      echo "[ASSEMBLY] GetOrganelle FILTERED (GB)..."
      GB_ASM_OUT="data/assemblies/${RUN}_gb_filtered_cli/getorganelle_filtered_gb"
      run_getorganelle "$GB_R1_OUT" "$GB_R2_OUT" "$GB_ASM_OUT" "$THREADS" "$GO_ENV"
    fi

    if [[ "$ASSEMBLE_CNN" == "y" ]]; then
      echo ""
      echo "[ASSEMBLY] GetOrganelle FILTERED (CNN)..."
      CNN_ASM_OUT="data/assemblies/${RUN}_cnn_filtered_cli/getorganelle_filtered_cnn"
      run_getorganelle "$CNN_R1_OUT" "$CNN_R2_OUT" "$CNN_ASM_OUT" "$THREADS" "$GO_ENV"
    fi

    if [[ "$ASSEMBLE_TRF" == "y" ]]; then
      echo ""
      echo "[ASSEMBLY] GetOrganelle FILTERED (Transformer)..."
      TRF_ASM_OUT="data/assemblies/${RUN}_trf_filtered_cli/getorganelle_filtered_trf"
      run_getorganelle "$TRF_R1_OUT" "$TRF_R2_OUT" "$TRF_ASM_OUT" "$THREADS" "$GO_ENV"
    fi

    echo ""
    echo "[Bandage files]"
    echo "UNFILTERED:"
    echo "  ${UNF_OUT}/extended_spades/K*/assembly_graph.fastg.extend-animal_mt.fastg"
    echo "  ${UNF_OUT}/animal_mt.K*.contigs.graph1.selected_graph.gfa"
    if [[ "$ASSEMBLE_GB" == "y" ]]; then
      echo "GB FILTERED:"
      echo "  ${GB_ASM_OUT}/extended_spades/K*/assembly_graph.fastg.extend-animal_mt.fastg"
      echo "  ${GB_ASM_OUT}/animal_mt.K*.contigs.graph1.selected_graph.gfa"
    fi
    if [[ "$ASSEMBLE_CNN" == "y" ]]; then
      echo "CNN FILTERED:"
      echo "  ${CNN_ASM_OUT}/extended_spades/K*/assembly_graph.fastg.extend-animal_mt.fastg"
      echo "  ${CNN_ASM_OUT}/animal_mt.K*.contigs.graph1.selected_graph.gfa"
    fi
    if [[ "$ASSEMBLE_TRF" == "y" ]]; then
      echo "TRF FILTERED:"
      echo "  ${TRF_ASM_OUT}/extended_spades/K*/assembly_graph.fastg.extend-animal_mt.fastg"
      echo "  ${TRF_ASM_OUT}/animal_mt.K*.contigs.graph1.selected_graph.gfa"
    fi

  else
    # SINGLE filtered output: make folder model-specific
    echo ""
    echo "[ASSEMBLY] GetOrganelle FILTERED (${FILTER_LABEL})..."
    FIL_OUT="data/assemblies/${RUN}_${FILTER_LABEL}_filtered_cli/getorganelle_filtered_${FILTER_LABEL}"
    run_getorganelle "$FINAL_R1" "$FINAL_R2" "$FIL_OUT" "$THREADS" "$GO_ENV"

    echo ""
    echo "[Bandage files]"
    echo "UNFILTERED:"
    echo "  ${UNF_OUT}/extended_spades/K*/assembly_graph.fastg.extend-animal_mt.fastg"
    echo "  ${UNF_OUT}/animal_mt.K*.contigs.graph1.selected_graph.gfa"
    echo "FILTERED (${FILTER_LABEL}):"
    echo "  ${FIL_OUT}/extended_spades/K*/assembly_graph.fastg.extend-animal_mt.fastg"
    echo "  ${FIL_OUT}/animal_mt.K*.contigs.graph1.selected_graph.gfa"
  fi
fi

echo ""
echo "[DONE] Pipeline finished."