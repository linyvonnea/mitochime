#!/usr/bin/env bash
set -euo pipefail
REF="data/refs/original.fasta"
THREADS=8
T="0.5"
TAG="t0p5"

DATASETS=(
  "10K_initial5_mixed   data/external_test/10K_initial5_mixed_R1.fastq.gz   data/external_test/10K_initial5_mixed_R2.fastq.gz"
  "10K_initial10_mixed  data/external_test/10K_initial10_mixed_R1.fastq.gz  data/external_test/10K_initial10_mixed_R2.fastq.gz"
  "10K_initial20_mixed  data/external_test/10K_initial20_mixed_R1.fastq.gz  data/external_test/10K_initial20_mixed_R2.fastq.gz"
  "10K_initial50_mixed  data/external_test/10K_initial50_mixed_R1.fastq.gz  data/external_test/10K_initial50_mixed_R2.fastq.gz"
  "3200_initial5_mixed  data/external_test/3200_initial5_mixed_R1.fastq.gz  data/external_test/3200_initial5_mixed_R2.fastq.gz"
  "3200_initial10_mixed data/external_test/3200_initial10_mixed_R1.fastq.gz data/external_test/3200_initial10_mixed_R2.fastq.gz"
  "3200_initial20_mixed data/external_test/3200_initial20_mixed_R1.fastq.gz data/external_test/3200_initial20_mixed_R2.fastq.gz"
  "3200_initial50_mixed data/external_test/3200_initial50_mixed_R1.fastq.gz data/external_test/3200_initial50_mixed_R2.fastq.gz"
)

mkdir -p reports
OUT="reports/eval_initial_t0p5.tsv"
echo -e "dataset\tmodel\tthresh\treads_in\treads_out\tpct_kept\tspades_contigs\tspades_total_len\tspades_max_len\tcontigs_path" > "$OUT"

num_reads() { seqkit stats "$1" | awk 'NR==2{gsub(",","",$4); print $4}'; }
contig_stats() { seqkit stats "$1" | awk 'NR==2{gsub(",","",$4); gsub(",","",$5); gsub(",","",$8); print $4" "$5" "$8}'; }

for entry in "${DATASETS[@]}"; do
  set -- $entry
  BASE="$1"; R1="$2"; R2="$3"
  echo ""
  echo "=== $BASE ==="

  IN_READS=$(num_reads "$R1")

  # UNFILTERED assembly
  UNF_ASM="data/assemblies_spades/${BASE}_unfiltered/spades"
  bash src/scripts/run_spades.sh "$R1" "$R2" "$UNF_ASM" "$THREADS" ""
  read CNUM CTOT CMAX <<<"$(contig_stats "$UNF_ASM/contigs.fasta")"
  echo -e "${BASE}\tUNFILTERED\tNA\t${IN_READS}\t${IN_READS}\t100.000\t${CNUM}\t${CTOT}\t${CMAX}\t${UNF_ASM}/contigs.fasta" >> "$OUT"

  # GB filter + assembly
  GB_RUN="${BASE}_gb_${TAG}"
  bash src/scripts/run_pipeline_gb.sh "$R1" "$R2" "$GB_RUN" "$T" "$THREADS" "$REF"
  GB_R1="data/filtered_reads/${GB_RUN}/${GB_RUN}.gb.filtered_R1.fastq.gz"
  GB_R2="data/filtered_reads/${GB_RUN}/${GB_RUN}.gb.filtered_R2.fastq.gz"
  OUT_READS=$(num_reads "$GB_R1")
  PCT=$(python3 - <<PY
inr=$IN_READS; outr=$OUT_READS
print(f"{(outr/inr)*100:.3f}")
PY
)
  GB_ASM="data/assemblies_spades/${GB_RUN}/spades"
  bash src/scripts/run_spades.sh "$GB_R1" "$GB_R2" "$GB_ASM" "$THREADS" ""
  read CNUM CTOT CMAX <<<"$(contig_stats "$GB_ASM/contigs.fasta")"
  echo -e "${BASE}\tGB\t${T}\t${IN_READS}\t${OUT_READS}\t${PCT}\t${CNUM}\t${CTOT}\t${CMAX}\t${GB_ASM}/contigs.fasta" >> "$OUT"

  # CNN filter + assembly
  CNN_RUN="${BASE}_cnn_${TAG}"
  bash src/scripts/run_pipeline_cnn.sh "$R1" "$R2" "$CNN_RUN" "$T"
  CNN_R1="data/filtered_reads/${CNN_RUN}/${CNN_RUN}.cnn.filtered_R1.fastq.gz"
  CNN_R2="data/filtered_reads/${CNN_RUN}/${CNN_RUN}.cnn.filtered_R2.fastq.gz"
  OUT_READS=$(num_reads "$CNN_R1")
  PCT=$(python3 - <<PY
inr=$IN_READS; outr=$OUT_READS
print(f"{(outr/inr)*100:.3f}")
PY
)
  CNN_ASM="data/assemblies_spades/${CNN_RUN}/spades"
  bash src/scripts/run_spades.sh "$CNN_R1" "$CNN_R2" "$CNN_ASM" "$THREADS" ""
  read CNUM CTOT CMAX <<<"$(contig_stats "$CNN_ASM/contigs.fasta")"
  echo -e "${BASE}\tCNN\t${T}\t${IN_READS}\t${OUT_READS}\t${PCT}\t${CNUM}\t${CTOT}\t${CMAX}\t${CNN_ASM}/contigs.fasta" >> "$OUT"

done

echo ""
echo "[DONE] -> $OUT"
column -t -s $'\t' "$OUT" | head -n 40
