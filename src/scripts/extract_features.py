#!/usr/bin/env python3
"""
extract_features.py

Extract per-read features from a BAM file:

  - read-level metadata:
      read_length, mean_base_quality
      ref_name, ref_start_1based, strand, mapq, cigar

  - SA-based split-alignment structure:
      has_sa, sa_count, num_segments,
      sa_diff_contig,
      sa_min_delta_pos, sa_max_delta_pos, sa_mean_delta_pos,
      sa_same_strand_count, sa_opp_strand_count,
      sa_max_mapq, sa_mean_mapq,
      sa_min_nm, sa_mean_nm

  - clipping:
      softclip_left, softclip_right, total_clipped_bases

  - breakpoint (read coordinate):
      breakpoint_read_pos  (now inferred from clipping / alignment)

  - k-mer composition jump (left vs right halves):
      kmer_cosine_diff, kmer_js_divergence

  - micro-homology around inferred breakpoint:
      microhomology_length, microhomology_gc

Usage
-----

Example: clean vs chimeric BAMs

    python scripts/extract_features.py \
        --bam data/aligned/clean.sorted.bam \
        --out data/features/clean_features.tsv \
        --label 0

    python scripts/extract_features.py \
        --bam data/aligned/chimeric.sorted.bam \
        --out data/features/chim_features.tsv \
        --label 1

Requirements: pysam, numpy
"""

import argparse
import csv

import numpy as np
import pysam


# ============================
# K-MER UTILITIES
# ============================


def kmer_profile(seq: str, k: int = 5) -> dict:
    """
    Compute raw k-mer counts for a single sequence (no normalization).
    Ignores k-mers containing N.
    """
    seq = seq.upper()
    n = len(seq)
    counts: dict[str, int] = {}
    if n < k:
        return counts
    for i in range(n - k + 1):
        kmer = seq[i : i + k]
        if "N" in kmer:
            continue
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def normalize_counts(counts: dict) -> dict:
    """
    Convert raw counts to probabilities that sum to 1.
    """
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def js_divergence(p: dict, q: dict) -> float:
    """
    Jensen–Shannon divergence between two discrete distributions
    given as dicts of probabilities. Returns value in [0, ~1].
    """
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0

    P = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    Q = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    M = 0.5 * (P + Q)

    def _kl_div(a, b):
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js = 0.5 * _kl_div(P, M) + 0.5 * _kl_div(Q, M)
    return js


def cosine_difference(p: dict, q: dict) -> float:
    """
    1 - cosine similarity between two probability dicts.
    """
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0

    P = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    Q = np.array([q.get(k, 0.0) for k in keys], dtype=float)

    if np.all(P == 0) or np.all(Q == 0):
        return 0.0

    dot = float(np.dot(P, Q))
    normP = float(np.linalg.norm(P))
    normQ = float(np.linalg.norm(Q))
    if normP == 0.0 or normQ == 0.0:
        return 0.0

    cos_sim = dot / (normP * normQ)
    cos_sim = max(min(cos_sim, 1.0), -1.0)  # clamp
    return 1.0 - cos_sim


# ============================
# MICRO-HOMOLOGY
# ============================


def longest_suffix_prefix_overlap(left: str, right: str, max_len: int = 30) -> int:
    """
    Find longest exact suffix-prefix overlap length between left and right,
    up to max_len. Returns overlap length only.
    """
    left = left.upper()
    right = right.upper()
    max_possible = min(len(left), len(right), max_len)

    for L in range(max_possible, 0, -1):
        if left[-L:] == right[:L]:
            return L
    return 0


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


# ============================
# ALIGNMENT / CIGAR FEATURES
# ============================


def soft_clips(read: pysam.AlignedSegment) -> tuple[int, int]:
    """
    Return (softclip_left, softclip_right) from CIGAR (operation 4).
    """
    left = right = 0
    if read.cigartuples:
        first_op, first_len = read.cigartuples[0]
        last_op, last_len = read.cigartuples[-1]
        if first_op == 4:
            left = first_len
        if last_op == 4:
            right = last_len
    return left, right


def total_clipped(read: pysam.AlignedSegment) -> int:
    """
    Total clipped bases (soft + hard; ops 4 and 5).
    """
    if not read.cigartuples:
        return 0
    total = 0
    for op, length in read.cigartuples:
        if op in (4, 5):
            total += length
    return total


def infer_breakpoint(read: pysam.AlignedSegment, read_len: int) -> int:
    """
    Infer a likely breakpoint position on the read (0-based index).

    Priority:
      1) If there is soft clipping, assume the junction is at the edge
         of the aligned block (left or right, whichever has larger clip).
      2) If no clipping but SA tag exists, use the midpoint of the
         aligned portion (query_alignment_start/end).
      3) Fallback: midpoint of the read.

    This is more biologically reasonable than always using the strict
    read midpoint, because many chimeric events manifest as truncated
    alignments with clipped ends.
    """
    left_clip, right_clip = soft_clips(read)

    # Case 1: soft clipping present -> junction at edge of aligned region
    if left_clip > 0 or right_clip > 0:
        # If both sides clipped, use the side with larger clip
        if left_clip >= right_clip:
            # Alignment starts at read position left_clip
            return left_clip
        else:
            # Alignment ends at read position read_len - right_clip
            return max(0, read_len - right_clip)

    # Case 2: no clipping, but split alignment hinted by SA tag
    if read.has_tag("SA"):
        q_start = read.query_alignment_start
        q_end = read.query_alignment_end
        if q_start is not None and q_end is not None and q_end > q_start:
            return (q_start + q_end) // 2

    # Case 3: fallback to simple midpoint
    return read_len // 2


# ============================
# SA TAG FEATURES
# ============================


def parse_sa_tag(read: pysam.AlignedSegment) -> list[dict]:
    """
    Parse SA:Z tag into list of dicts:
    [{rname, pos, strand, cigar, mapq, nm}, ...]
    """
    if not read.has_tag("SA"):
        return []

    sa_raw = read.get_tag("SA")
    entries = [e for e in sa_raw.split(";") if e]

    segments: list[dict] = []
    for e in entries:
        rname, pos, strand, cigar, mapq, nm = e.split(",")
        segments.append(
            {
                "rname": rname,
                "pos": int(pos),  # 1-based pos
                "strand": strand,
                "cigar": cigar,
                "mapq": int(mapq),
                "nm": int(nm),
            }
        )
    return segments


def sa_feature_stats(
    primary_rname: str,
    primary_pos: int,
    primary_strand: str,
    sa_segments: list[dict],
) -> dict:
    """
    Compute SA-based numeric features.
    """
    has_sa = 1 if sa_segments else 0
    sa_count = len(sa_segments)
    num_segments = 1 + sa_count

    deltas_same: list[int] = []
    diff_contig = 0
    same_strand = 0
    opp_strand = 0
    sa_mapqs: list[int] = []
    sa_nms: list[int] = []

    for seg in sa_segments:
        sa_mapqs.append(seg["mapq"])
        sa_nms.append(seg["nm"])

        if seg["rname"] != primary_rname:
            diff_contig = 1
        else:
            delta = abs(seg["pos"] - primary_pos)
            deltas_same.append(delta)

        if seg["strand"] == primary_strand:
            same_strand += 1
        else:
            opp_strand += 1

    if deltas_same:
        sa_min_delta = min(deltas_same)
        sa_max_delta = max(deltas_same)
        sa_mean_delta = sum(deltas_same) / len(deltas_same)
    else:
        sa_min_delta = 0
        sa_max_delta = 0
        sa_mean_delta = 0.0

    if sa_mapqs:
        sa_max_mapq = max(sa_mapqs)
        sa_mean_mapq = sum(sa_mapqs) / len(sa_mapqs)
    else:
        sa_max_mapq = 0
        sa_mean_mapq = 0.0

    if sa_nms:
        sa_min_nm = min(sa_nms)
        sa_mean_nm = sum(sa_nms) / len(sa_nms)
    else:
        sa_min_nm = 0
        sa_mean_nm = 0.0

    return {
        "has_sa": has_sa,
        "sa_count": sa_count,
        "num_segments": num_segments,
        "sa_diff_contig": diff_contig,
        "sa_min_delta_pos": sa_min_delta,
        "sa_max_delta_pos": sa_max_delta,
        "sa_mean_delta_pos": sa_mean_delta,
        "sa_same_strand_count": same_strand,
        "sa_opp_strand_count": opp_strand,
        "sa_max_mapq": sa_max_mapq,
        "sa_mean_mapq": sa_mean_mapq,
        "sa_min_nm": sa_min_nm,
        "sa_mean_nm": sa_mean_nm,
    }


# ============================
# MAIN FEATURE EXTRACTION
# ============================


def extract_features(
    bam_path: str,
    out_tsv: str,
    label: int,
    k: int = 5,
    micro_window: int = 30,
) -> None:
    """
    Main feature extraction loop.
    One row per primary mapped read.
    """
    bam = pysam.AlignmentFile(bam_path, "rb")

    with open(out_tsv, "w", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")

        # Header (sequence NOT included as feature, only read_id)
        writer.writerow(
            [
                "read_id",
                "label",
                "read_length",
                "mean_base_quality",
                "ref_name",
                "ref_start_1based",
                "strand",
                "mapq",
                "cigar",
                # SA features
                "has_sa",
                "sa_count",
                "num_segments",
                "sa_diff_contig",
                "sa_min_delta_pos",
                "sa_max_delta_pos",
                "sa_mean_delta_pos",
                "sa_same_strand_count",
                "sa_opp_strand_count",
                "sa_max_mapq",
                "sa_mean_mapq",
                "sa_min_nm",
                "sa_mean_nm",
                # clipping
                "softclip_left",
                "softclip_right",
                "total_clipped_bases",
                # breakpoint
                "breakpoint_read_pos",
                # k-mer features
                "kmer_cosine_diff",
                "kmer_js_divergence",
                # micro-homology
                "microhomology_length",
                "microhomology_gc",
            ]
        )

        for read in bam.fetch(until_eof=True):
            # Only use primary mapped reads
            if read.is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue

            seq = read.query_sequence
            quals = read.query_qualities

            if seq is None or len(seq) == 0:
                continue

            read_len = len(seq)
            if quals is not None and len(quals) > 0:
                mean_q = float(np.mean(quals))
            else:
                mean_q = 0.0

            ref_name = bam.get_reference_name(read.reference_id)
            ref_start_1based = read.reference_start + 1
            strand = 1 if read.is_reverse else 0

            mapq = read.mapping_quality
            cigar = read.cigarstring if read.cigarstring is not None else "*"

            # SA features
            sa_segments = parse_sa_tag(read)
            sa_feats = sa_feature_stats(
                primary_rname=ref_name,
                primary_pos=ref_start_1based,
                primary_strand=strand,
                sa_segments=sa_segments,
            )

            # Clipping
            soft_left, soft_right = soft_clips(read)
            clipped_total = total_clipped(read)

            # Breakpoint: inferred from clipping / alignment
            breakpoint = infer_breakpoint(read, read_len)

            # Guard: ensure in range
            if breakpoint < 0:
                breakpoint = 0
            elif breakpoint > read_len:
                breakpoint = read_len

            # K-mer composition difference between left/right halves
            left_seq = seq[:breakpoint]
            right_seq = seq[breakpoint:]

            left_counts = kmer_profile(left_seq, k=k)
            right_counts = kmer_profile(right_seq, k=k)
            left_probs = normalize_counts(left_counts)
            right_probs = normalize_counts(right_counts)

            kmer_cos_diff = cosine_difference(left_probs, right_probs)
            kmer_js = js_divergence(left_probs, right_probs)

            # Micro-homology in ±micro_window around inferred breakpoint
            left_window_start = max(0, breakpoint - micro_window)
            left_window = seq[left_window_start:breakpoint]

            right_window_end = min(read_len, breakpoint + micro_window)
            right_window = seq[breakpoint:right_window_end]

            mh_len = longest_suffix_prefix_overlap(
                left_window, right_window, max_len=micro_window
            )

            if mh_len > 0:
                mh_seq = left_window[-mh_len:]
                mh_gc = gc_content(mh_seq)
            else:
                mh_gc = 0.0

            writer.writerow(
                [
                    read.query_name,
                    label,
                    read_len,
                    f"{mean_q:.3f}",
                    ref_name,
                    ref_start_1based,
                    strand,
                    mapq,
                    cigar,
                    sa_feats["has_sa"],
                    sa_feats["sa_count"],
                    sa_feats["num_segments"],
                    sa_feats["sa_diff_contig"],
                    sa_feats["sa_min_delta_pos"],
                    sa_feats["sa_max_delta_pos"],
                    f"{sa_feats['sa_mean_delta_pos']:.3f}",
                    sa_feats["sa_same_strand_count"],
                    sa_feats["sa_opp_strand_count"],
                    sa_feats["sa_max_mapq"],
                    f"{sa_feats['sa_mean_mapq']:.3f}",
                    sa_feats["sa_min_nm"],
                    f"{sa_feats['sa_mean_nm']:.3f}",
                    soft_left,
                    soft_right,
                    clipped_total,
                    breakpoint,
                    f"{kmer_cos_diff:.6f}",
                    f"{kmer_js:.6f}",
                    mh_len,
                    f"{mh_gc:.3f}",
                ]
            )

    bam.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract alignment, SA, k-mer, and micro-homology features from a BAM file."
    )
    parser.add_argument("--bam", required=True, help="Input BAM file (sorted, indexed).")
    parser.add_argument("--out", required=True, help="Output TSV file.")
    parser.add_argument(
        "--label",
        type=int,
        required=True,
        help="Class label for these reads (e.g., 0 = clean, 1 = chimeric).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k-mer size for composition features (default: 5).",
    )
    parser.add_argument(
        "--micro-window",
        type=int,
        default=30,
        help="Window size around breakpoint for micro-homology (default: 30).",
    )

    args = parser.parse_args()

    extract_features(
        bam_path=args.bam,
        out_tsv=args.out,
        label=args.label,
        k=args.k,
        micro_window=args.micro_window,
    )


if __name__ == "_main_":
    main()