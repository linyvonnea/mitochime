#!/usr/bin/env bash
set -euo pipefail

REF="data/refs/original.fasta"
THREADS=8
T="0.5"
TAG="t0p5"

CNN_MODEL="models/deep/cnn_final_L150_seed42_fixedep25/cnn_final.pt"
export CNN_MODEL

DATASETS=(
  "10K_final_5   data/external_test/10K_final_5_R1.fastq.gz   data/external_test/10K_final_5_R2.fastq.gz"
  "10K_final_10  data/external_test/10K_final_10_R1.fastq.gz  data/external_test/10K_final_10_R2.fastq.gz"
  "10K_final_15  data/external_test/10K_final_15_R1.fastq.gz  data/external_test/10K_final_15_R2.fastq.gz"
  "10K_final_50  data/external_test/10K_final_50_R1.fastq.gz  data/external_test/10K_final_50_R2.fastq.gz"

  "20K_final_5   data/external_test/20K_final_5_R1.fastq.gz   data/external_test/20K_final_5_R2.fastq.gz"
  "20K_final_10  data/external_test/20K_final_10_R1.fastq.gz  data/external_test/20K_final_10_R2.fastq.gz"
  "20K_final_15  data/external_test/20K_final_15_R1.fastq.gz  data/external_test/20K_final_15_R2.fastq.gz"
  "20K_final_50  data/external_test/20K_final_50_R1.fastq.gz  data/external_test/20K_final_50_R2.fastq.gz"

  "3200_final_5   data/external_test/3200_final_5_R1.fastq.gz   data/external_test/3200_final_5_R2.fastq.gz"
  "3200_final_10  data/external_test/3200_final_10_R1.fastq.gz  data/external_test/3200_final_10_R2.fastq.gz"
  "3200_final_15  data/external_test/3200_final_15_R1.fastq.gz  data/external_test/3200_final_15_R2.fastq.gz"
  "3200_final_50  data/external_test/3200_final_50_R1.fastq.gz  data/external_test/3200_final_50_R2.fastq.gz"
)

mkdir -p reports
OUT="reports/eval_final_all_${TAG}.tsv"
echo -e "dataset\tmodel\tthresh\treads_in\treads_out\tpct_kept\tspades_contigs\tspades_total_len\tspades_max_len\tcontigs_path" > "$OUT"

num_reads() {
  seqkit stats "$1" | awk 'NR==2{gsub(",","",$4); print $4}'
}

contig_stats() {
  seqkit stats "$1" | awk 'NR==2{gsub(",","",$4); gsub(",","",$5); gsub(",","",$8); print $4" "$5" "$8}'
}

for entry in "${DATASETS[@]}"; do
  set -- $entry
  BASE="$1"
  R1="$2"
  R2="$3"

  echo ""
  echo "=== $BASE ==="

  IN_READS=$(num_reads "$R1")

  # UNFILTERED
  UNF_ASM="data/assemblies_spades/${BASE}_unfiltered/spades"
  bash src/scripts/run_spades.sh "$R1" "$R2" "$UNF_ASM" "$THREADS" ""
  read CNUM CTOT CMAX <<<"$(contig_stats "$UNF_ASM/contigs.fasta")"
  echo -e "${BASE}\tUNFILTERED\tNA\t${IN_READS}\t${IN_READS}\t100.000\t${CNUM}\t${CTOT}\t${CMAX}\t${UNF_ASM}/contigs.fasta" >> "$OUT"

  # GB
  GB_RUN="${BASE}_gb_${TAG}"
  bash src/scripts/run_pipeline_gb.sh "$R1" "$R2" "$GB_RUN" "$T" "$THREADS" "$REF"
  GB_R1="data/filtered_reads/${GB_RUN}/${GB_RUN}.gb.filtered_R1.fastq.gz"
  GB_R2="data/filtered_reads/${GB_RUN}/${GB_RUN}.gb.filtered_R2.fastq.gz"
  OUT_READS=$(num_reads "$GB_R1")
  PCT=$(python3 - <<PY
inr=$IN_READS
outr=$OUT_READS
print(f"{(outr/inr)*100:.3f}")
PY
)
  GB_ASM="data/assemblies_spades/${GB_RUN}/spades"
  bash src/scripts/run_spades.sh "$GB_R1" "$GB_R2" "$GB_ASM" "$THREADS" ""
  read CNUM CTOT CMAX <<<"$(contig_stats "$GB_ASM/contigs.fasta")"
  echo -e "${BASE}\tGB\t${T}\t${IN_READS}\t${OUT_READS}\t${PCT}\t${CNUM}\t${CTOT}\t${CMAX}\t${GB_ASM}/contigs.fasta" >> "$OUT"

  # CNN (new fixed epoch 25 model)
  CNN_RUN="${BASE}_cnn_fixedep25_${TAG}"
  CHECK_SAMPLE=0 bash src/scripts/run_pipeline_cnn.sh "$R1" "$R2" "$CNN_RUN" "$T"
  CNN_R1="data/filtered_reads/${CNN_RUN}/${CNN_RUN}.cnn.filtered_R1.fastq.gz"
  CNN_R2="data/filtered_reads/${CNN_RUN}/${CNN_RUN}.cnn.filtered_R2.fastq.gz"
  OUT_READS=$(num_reads "$CNN_R1")
  PCT=$(python3 - <<PY
inr=$IN_READS
outr=$OUT_READS
print(f"{(outr/inr)*100:.3f}")
PY
)
  CNN_ASM="data/assemblies_spades/${CNN_RUN}/spades"
  bash src/scripts/run_spades.sh "$CNN_R1" "$CNN_R2" "$CNN_ASM" "$THREADS" ""
  read CNUM CTOT CMAX <<<"$(contig_stats "$CNN_ASM/contigs.fasta")"
  echo -e "${BASE}\tCNN_fixedep25\t${T}\t${IN_READS}\t${OUT_READS}\t${PCT}\t${CNUM}\t${CTOT}\t${CMAX}\t${CNN_ASM}/contigs.fasta" >> "$OUT"
done

echo ""
echo "[DONE] -> $OUT"
column -t -s $'\t' "$OUT" | head -n 60