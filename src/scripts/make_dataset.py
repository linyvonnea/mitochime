#!/usr/bin/env python3
"""
make_dataset.py

Extract per-read features from:
  - clean.sorted.bam      (label = 0, clean)
  - chimeric.sorted.bam   (label = 1, chimeric)

and write a single TSV file (dataset.tsv) with columns:

  read_id, label,
  n_align, n_supp, n_secondary,
  max_mapq, min_nm,
  max_softclip_left, max_softclip_right

You can later add sequence-based features (k-mers, GC, etc.) by
extending extract_alignment_features() or adding additional passes.

Requirements:
  pip install pysam
"""

import csv
from collections import defaultdict
import pysam

OUT_PATH = "dataset.tsv"


def init_feature_record():
    """
    Initialize an empty feature record for a single read_id.
    Aggregates information from possibly multiple alignments
    (primary + supplementary).
    """
    return {
        "n_align": 0,
        "n_supp": 0,
        "n_secondary": 0,
        "max_mapq": 0,
        "min_nm": None,
        "max_softclip_left": 0,
        "max_softclip_right": 0,
    }


def update_features(rec, aln):
    """
    Update per-read aggregated features from a single alignment record.
    rec: dict returned by init_feature_record()
    aln: pysam.AlignedSegment
    """
    rec["n_align"] += 1

    if aln.is_supplementary:
        rec["n_supp"] += 1
    if aln.is_secondary:
        rec["n_secondary"] += 1

    # mapping quality
    if aln.mapping_quality > rec["max_mapq"]:
        rec["max_mapq"] = aln.mapping_quality

    # NM tag (edit distance / mismatches) if present
    nm = None
    try:
        nm = aln.get_tag("NM")
    except KeyError:
        pass

    if nm is not None:
        if rec["min_nm"] is None:
            rec["min_nm"] = nm
        else:
            rec["min_nm"] = min(rec["min_nm"], nm)

    # Soft-clip lengths at left and right ends
    # CIGAR operation 4 = soft clip
    softclip_left = 0
    softclip_right = 0
    if aln.cigartuples:
        if aln.cigartuples[0][0] == 4:
            softclip_left = aln.cigartuples[0][1]
        if aln.cigartuples[-1][0] == 4:
            softclip_right = aln.cigartuples[-1][1]

    if softclip_left > rec["max_softclip_left"]:
        rec["max_softclip_left"] = softclip_left
    if softclip_right > rec["max_softclip_right"]:
        rec["max_softclip_right"] = softclip_right


def process_bam(bam_path, label, writer):
    """
    Process one BAM file and write per-read feature rows.
    label: 0 for clean, 1 for chimeric
    """
    print(f"[INFO] Processing {bam_path} with label={label} ...")
    bam = pysam.AlignmentFile(bam_path, "rb")

    # Aggregate by read_id
    features_by_read = defaultdict(init_feature_record)

    for aln in bam.fetch():
        if aln.is_unmapped:
            continue
        # Optional: if you want to ignore secondary alignments altogether:
        # if aln.is_secondary:
        #     continue

        read_id = aln.query_name
        rec = features_by_read[read_id]
        update_features(rec, aln)

    bam.close()

    # Write out one row per read_id
    n_rows = 0
    for read_id, feats in features_by_read.items():
        # If min_nm was never set (no NM tag), set to -1 as placeholder
        if feats["min_nm"] is None:
            feats["min_nm"] = -1

        row = {
            "read_id": read_id,
            "label": label,
            **feats,
        }
        writer.writerow(row)
        n_rows += 1

    print(f"[INFO] Wrote {n_rows} rows for {bam_path}")


def main():
    fieldnames = [
        "read_id",
        "label",
        "n_align",
        "n_supp",
        "n_secondary",
        "max_mapq",
        "min_nm",
        "max_softclip_left",
        "max_softclip_right",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        # label 0 = clean
        process_bam("clean.sorted.bam", 0, writer)
        # label 1 = chimeric
        process_bam("chimeric.sorted.bam", 1, writer)

    print(f"[DONE] Feature dataset written to {OUT_PATH}")


if __name__ == "__main__":
    main()