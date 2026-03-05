#!/usr/bin/env bash
set -euo pipefail

# ===== CONFIG =====
REF="data/refs/original.fasta"
THREADS=8
SPADES_ENV=""   # set if needed, e.g. "spades_env"

GB_THRESHES=("0.3" "0.5")
CNN_THRESHES=("0.3" "0.5" "0.7")

DATASETS=(
  "3200_lemuru15 data/external_test/3200_lemuru15_R1.fastq.gz data/external_test/3200_lemuru15_R2.fastq.gz"
  "3200_lemuru50 data/external_test/3200_lemuru50_R1.fastq.gz data/external_test/3200_lemuru50_R2.fastq.gz"
  "10K_lemuru15  data/external_test/10K_lemuru15_R1.fastq.gz  data/external_test/10K_lemuru15_R2.fastq.gz"
  "10K_lemuru50  data/external_test/10K_lemuru50_R1.fastq.gz  data/external_test/10K_lemuru50_R2.fastq.gz"
)

mkdir -p reports

# ===== PRECHECKS =====
command -v seqkit >/dev/null 2>&1 || { echo "[ERROR] seqkit not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not found"; exit 1; }
[[ -f "$REF" ]] || { echo "[ERROR] Missing REF: $REF"; exit 1; }
[[ -f src/scripts/run_pipeline_gb.sh ]] || { echo "[ERROR] Missing src/scripts/run_pipeline_gb.sh"; exit 1; }
[[ -f src/scripts/run_pipeline_cnn.sh ]] || { echo "[ERROR] Missing src/scripts/run_pipeline_cnn.sh"; exit 1; }
[[ -f src/scripts/run_spades.sh ]] || { echo "[ERROR] Missing src/scripts/run_spades.sh"; exit 1; }

# ===== HELPERS =====
num_reads() {  # FASTQ reads in file
  seqkit stats "$1" | awk 'NR==2{gsub(",","",$4); print $4}'
}

contig_stats() { # contigs.fasta -> num_seqs total_len max_len
  if [[ ! -f "$1" ]]; then
    echo "NA NA NA"
    return
  fi
  seqkit stats "$1" | awk 'NR==2{gsub(",","",$4); gsub(",","",$5); gsub(",","",$8); print $4" "$5" "$8}'
}

# ===== OUTPUT TABLE =====
OUT="reports/eval_summary.tsv"
echo -e "dataset\tmodel\tthresh\treads_in\treads_out\tpct_kept\tspades_contigs\tspades_total_len\tspades_max_len\tcontigs_path" > "$OUT"

# ===== RUN =====
for entry in "${DATASETS[@]}"; do
  set -- $entry
  BASE="$1"; R1="$2"; R2="$3"

  [[ -f "$R1" ]] || { echo "[ERROR] Missing $R1"; exit 1; }
  [[ -f "$R2" ]] || { echo "[ERROR] Missing $R2"; exit 1; }

  echo ""
  echo "=============================="
  echo "[DATASET] $BASE"
  echo "=============================="

  IN_READS=$(num_reads "$R1")

  # ---- SPAdes on UNFILTERED ----
  UNF_ASM="data/assemblies_spades/${BASE}_unfiltered/spades"
  if [[ ! -f "${UNF_ASM}/contigs.fasta" ]]; then
    echo "[SPAdes] unfiltered -> $UNF_ASM"
    bash src/scripts/run_spades.sh "$R1" "$R2" "$UNF_ASM" "$THREADS" "$SPADES_ENV"
  else
    echo "[SPAdes] unfiltered exists -> $UNF_ASM"
  fi

  # ---- GB runs ----
  for T in "${GB_THRESHES[@]}"; do
    TAG="t${T/./p}"                 # 0.3 -> t0p3
    RUN="${BASE}_gb_${TAG}"

    echo ""
    echo "[GB] RUN=$RUN thresh=$T"
    bash src/scripts/run_pipeline_gb.sh "$R1" "$R2" "$RUN" "$T" "$THREADS" "$REF"

    GB_R1="data/filtered_reads/${RUN}/${RUN}.gb.filtered_R1.fastq.gz"
    [[ -f "$GB_R1" ]] || { echo "[ERROR] Missing GB output $GB_R1"; exit 1; }

    OUT_READS=$(num_reads "$GB_R1")
    PCT=$(python3 - <<PY
inr=$IN_READS; outr=$OUT_READS
print(f"{(outr/inr)*100:.3f}")
PY
)

    # SPAdes on GB-filtered
    ASM="data/assemblies_spades/${RUN}/spades"
    echo "[SPAdes] GB-filtered -> $ASM"
    bash src/scripts/run_spades.sh \
      "data/filtered_reads/${RUN}/${RUN}.gb.filtered_R1.fastq.gz" \
      "data/filtered_reads/${RUN}/${RUN}.gb.filtered_R2.fastq.gz" \
      "$ASM" "$THREADS" "$SPADES_ENV"

    read CNUM CTOT CMAX <<<"$(contig_stats "$ASM/contigs.fasta")"
    echo -e "${BASE}\tGB\t${T}\t${IN_READS}\t${OUT_READS}\t${PCT}\t${CNUM}\t${CTOT}\t${CMAX}\t${ASM}/contigs.fasta" >> "$OUT"
  done

  # ---- CNN runs ----
  for T in "${CNN_THRESHES[@]}"; do
    TAG="t${T/./p}"
    RUN="${BASE}_cnn_${TAG}"

    echo ""
    echo "[CNN] RUN=$RUN thresh=$T"
    bash src/scripts/run_pipeline_cnn.sh "$R1" "$R2" "$RUN" "$T"

    CNN_R1="data/filtered_reads/${RUN}/${RUN}.cnn.filtered_R1.fastq.gz"
    [[ -f "$CNN_R1" ]] || { echo "[ERROR] Missing CNN output $CNN_R1"; exit 1; }

    OUT_READS=$(num_reads "$CNN_R1")
    PCT=$(python3 - <<PY
inr=$IN_READS; outr=$OUT_READS
print(f"{(outr/inr)*100:.3f}")
PY
)

    # SPAdes on CNN-filtered
    ASM="data/assemblies_spades/${RUN}/spades"
    echo "[SPAdes] CNN-filtered -> $ASM"
    bash src/scripts/run_spades.sh \
      "data/filtered_reads/${RUN}/${RUN}.cnn.filtered_R1.fastq.gz" \
      "data/filtered_reads/${RUN}/${RUN}.cnn.filtered_R2.fastq.gz" \
      "$ASM" "$THREADS" "$SPADES_ENV"

    read CNUM CTOT CMAX <<<"$(contig_stats "$ASM/contigs.fasta")"
    echo -e "${BASE}\tCNN\t${T}\t${IN_READS}\t${OUT_READS}\t${PCT}\t${CNUM}\t${CTOT}\t${CMAX}\t${ASM}/contigs.fasta" >> "$OUT"
  done

done

echo ""
echo "[DONE] Wrote summary table -> $OUT"
echo "Preview:"
column -t -s $'\t' "$OUT" | head -n 20
