#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch Lemuru: UNFILTERED + (GB + CNN) filtered across thresholds + SPAdes multi-k
#
# Assumes:
#  - GB script writes to:  data/filtered_reads/${RUN}_gb/${RUN}_gb.gb.filtered_R{1,2}.fastq.gz
#  - CNN script writes to: data/filtered_reads/${RUN}/${RUN}.cnn.filtered_R{1,2}.fastq.gz
# ============================================================

# ---------- Only lemuru datasets (base names WITHOUT _R1/_R2) ----------
DATASETS=(
  "3200_lemuru5"
  "3200_lemuru10"
  "3200_lemuru15"
  "3200_lemuru50"
  "10K_lemuru5"
  "10K_lemuru10"
  "10K_lemuru15"
  "10K_lemuru50"
)

RAW_DIR="data/external_test"

# thresholds to test
THRESHOLDS=(0.3 0.5 0.7)

# SPAdes k-lists to test (you said “all k values” — best interpreted as multiple k-lists)
# NOTE: 127 may fail on some builds/data; keep a second list without 127.
KLISTS=(
  "21,33,55,77,99,127"
  "21,33,55,77,99,115"
)

THREADS="${THREADS:-8}"
OVERWRITE="${OVERWRITE:-0}"

REF_DEFAULT="data/refs/original.fasta"

GB_SCRIPT="src/scripts/run_pipeline_gb.sh"
CNN_SCRIPT="src/scripts/run_pipeline_cnn.sh"

ASM_ROOT="data/assemblies_spades"
SPADES_ENV="${SPADES_ENV:-}"   # optional env name for spades.py

# ---------------- Helpers ----------------
die() { echo "[ERROR] $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "File not found: $1"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command in PATH: $1"; }

maybe_run() {
  # maybe_run "<target_file>" "<command...>"
  local target="$1"; shift
  if [[ "$OVERWRITE" == "1" ]]; then
    echo "[RUN] $*"
    eval "$@"
    return 0
  fi
  if [[ -f "$target" ]]; then
    echo "[SKIP] exists: $target"
    return 0
  fi
  echo "[RUN] $*"
  eval "$@"
}

run_spades() {
  local r1="$1"
  local r2="$2"
  local outdir="$3"
  local klist="$4"

  mkdir -p "$outdir"

  if [[ -n "$SPADES_ENV" ]]; then
    conda run -n "$SPADES_ENV" spades.py -1 "$r1" -2 "$r2" -o "$outdir" -t "$THREADS" -k "$klist"
  else
    spades.py -1 "$r1" -2 "$r2" -o "$outdir" -t "$THREADS" -k "$klist"
  fi
}

thr_token() {
  # 0.3 -> t0p3 ; 0.05 -> t0p05 ; 0.50 -> t0p5
  python3 - <<PY
t=float("$1")
s=("{:g}".format(t)).replace(".","p")
print("t"+s)
PY
}

# ---------------- Checks ----------------
need_cmd bash
need_cmd python3
need_file "$GB_SCRIPT"
need_file "$CNN_SCRIPT"
need_file "$REF_DEFAULT"

if [[ -z "$SPADES_ENV" ]]; then
  need_cmd spades.py
fi

chmod +x "$GB_SCRIPT" "$CNN_SCRIPT" >/dev/null 2>&1 || true
mkdir -p "$ASM_ROOT"

echo "=== Batch Lemuru All ==="
echo "THREADS=$THREADS"
echo "THRESHOLDS=${THRESHOLDS[*]}"
echo "KLISTS=${KLISTS[*]}"
echo "REF=$REF_DEFAULT"
echo "OVERWRITE=$OVERWRITE"
[[ -n "$SPADES_ENV" ]] && echo "SPADES_ENV=$SPADES_ENV"
echo ""

# ---------------- Main loop ----------------
for ds in "${DATASETS[@]}"; do
  R1="${RAW_DIR}/${ds}_R1.fastq.gz"
  R2="${RAW_DIR}/${ds}_R2.fastq.gz"
  need_file "$R1"
  need_file "$R2"

  echo "============================================================"
  echo "[DATASET] $ds"

  # 1) UNFILTERED assemblies (for each klist)
  for k in "${KLISTS[@]}"; do
    ktag="k$(echo "$k" | tr ',' '_')"
    UNF_DIR="${ASM_ROOT}/${ds}_unfiltered/${ktag}"
    UNF_CONTIGS="${UNF_DIR}/contigs.fasta"
    maybe_run "$UNF_CONTIGS" "run_spades \"$R1\" \"$R2\" \"$UNF_DIR\" \"$k\""
  done

  # 2) Filter + assemble for each threshold (GB and CNN)
  for thr in "${THRESHOLDS[@]}"; do
    ttag="$(thr_token "$thr")"

    # ---------- GB ----------
    GB_RUN="${ds}_gb_${ttag}"
    # IMPORTANT: GB script appends _gb to output folder and filenames
    GB_OUTDIR="data/filtered_reads/${GB_RUN}_gb"
    GB_R1_OUT="${GB_OUTDIR}/${GB_RUN}_gb.gb.filtered_R1.fastq.gz"
    GB_R2_OUT="${GB_OUTDIR}/${GB_RUN}_gb.gb.filtered_R2.fastq.gz"

    maybe_run "$GB_R1_OUT" \
      "bash \"$GB_SCRIPT\" \"$R1\" \"$R2\" \"$GB_RUN\" \"$thr\" \"$THREADS\" \"$REF_DEFAULT\""

    for k in "${KLISTS[@]}"; do
      ktag="k$(echo "$k" | tr ',' '_')"
      GB_ASM_DIR="${ASM_ROOT}/${ds}_gb_${ttag}_filtered/${ktag}"
      GB_ASM_CONTIGS="${GB_ASM_DIR}/contigs.fasta"
      maybe_run "$GB_ASM_CONTIGS" "run_spades \"$GB_R1_OUT\" \"$GB_R2_OUT\" \"$GB_ASM_DIR\" \"$k\""
    done

    # ---------- CNN ----------
    CNN_RUN="${ds}_cnn_${ttag}"
    # CNN script writes to folder RUN (no _cnn suffix)
    CNN_OUTDIR="data/filtered_reads/${CNN_RUN}"
    CNN_R1_OUT="${CNN_OUTDIR}/${CNN_RUN}.cnn.filtered_R1.fastq.gz"
    CNN_R2_OUT="${CNN_OUTDIR}/${CNN_RUN}.cnn.filtered_R2.fastq.gz"

    maybe_run "$CNN_R1_OUT" \
      "bash \"$CNN_SCRIPT\" \"$R1\" \"$R2\" \"$CNN_RUN\" \"$thr\""

    for k in "${KLISTS[@]}"; do
      ktag="k$(echo "$k" | tr ',' '_')"
      CNN_ASM_DIR="${ASM_ROOT}/${ds}_cnn_${ttag}_filtered/${ktag}"
      CNN_ASM_CONTIGS="${CNN_ASM_DIR}/contigs.fasta"
      maybe_run "$CNN_ASM_CONTIGS" "run_spades \"$CNN_R1_OUT\" \"$CNN_R2_OUT\" \"$CNN_ASM_DIR\" \"$k\""
    done

  done

  echo "[DONE] $ds"
  echo ""
done

echo "=== ALL DONE ==="
echo "Assemblies under: $ASM_ROOT"
echo "Tip: OVERWRITE=1 THREADS=8 bash src/scripts/run_batch_lemuru_all.sh"