#!/usr/bin/env python3
"""
make_chimeric_templates.py

Generate chimeric templates from a mitochondrial reference by
concatenating two non-adjacent segments with optional microhomology.

Features:
- Picks two segments (A and B) from the same genome that are
  separated by a configurable distance range.
- Tries to enforce a short microhomology at the junction (suffix of A
  equals prefix of B) between min_mh and max_mh bases.
- If no microhomology is found after a number of attempts, it relaxes
  the constraint and allows junctions without microhomology.
- Randomly splits the template length into two segment lengths so the
  junction position varies within the template.

Usage (manual example):
    python make_chimeric_templates.py \
        --ref /path/to/NC_039553.1_original.fasta \
        --out /path/to/NC_039553.1_chimera.fasta \
        --num-templates 1000 \
        --template-len 300 \
        --min-distance 500 \
        --max-distance 8000 \
        --min-mh 3 \
        --max-mh 8

This script is also meant to be called automatically by simulation.py.
"""

import argparse
import random
from pathlib import Path
from textwrap import wrap


def read_fasta_single(path: Path):
    """Read a single-sequence FASTA and return (header, sequence_str)."""
    header = None
    seq_chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is None:
                    header = line[1:].strip()
                else:
                    # assume single-entry FASTA; ignore extra entries
                    break
            else:
                seq_chunks.append(line)
    if header is None:
        raise ValueError(f"No FASTA header found in {path}")
    return header, "".join(seq_chunks).upper()


def write_fasta_record(handle, name: str, seq: str):
    """Write one FASTA record with wrapped sequence."""
    handle.write(f">{name}\n")
    for chunk in wrap(seq, 80):
        handle.write(chunk + "\n")


def find_microhomology(seg1: str, seg2: str, min_mh: int, max_mh: int) -> int:
    """
    Find the longest microhomology (suffix of seg1 == prefix of seg2)
    up to max_mh, at least min_mh. Returns mh_len (0 if none).
    """
    max_possible = min(len(seg1), len(seg2), max_mh)
    for k in range(max_possible, min_mh - 1, -1):
        if seg1[-k:] == seg2[:k]:
            return k
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate chimeric templates from a mitochondrial reference."
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Path to original mito FASTA (e.g. NC_039553.1_original.fasta)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output FASTA path for chimeric templates",
    )
    parser.add_argument(
        "--num-templates",
        type=int,
        default=1000,
        help="Number of chimeric templates to generate",
    )
    parser.add_argument(
        "--template-len",
        type=int,
        default=300,
        help="Length of each chimeric template (approx; "
             "actual length may be slightly shorter due to overlap)",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=500,
        help="Minimum distance between segment midpoints in reference",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=8000,
        help="Maximum distance between segment midpoints in reference",
    )
    parser.add_argument(
        "--min-mh",
        type=int,
        default=3,
        help="Minimum microhomology length",
    )
    parser.add_argument(
        "--max-mh",
        type=int,
        default=20,
        help="Maximum microhomology length",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=100,
        help="Max random attempts per template before relaxing constraints",
    )
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference FASTA not found: {ref_path}")

    header, genome = read_fasta_single(ref_path)
    L = len(genome)
    print(f"[INFO] Loaded reference {header} (length {L} bp)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)  # deterministic

    min_seg_len = 50
    if args.template_len < 2 * min_seg_len:
        raise ValueError("template-len too small for min_seg_len=50 on both sides")

    num_created = 0

    with open(out_path, "w") as out_f:
        while num_created < args.num_templates:
            # First try with microhomology required, then relax if needed
            for mh_required in (True, False):
                success = False

                for _ in range(args.max_tries):
                    # Randomly split template length into two segment lengths
                    len1 = rng.randint(min_seg_len, args.template_len - min_seg_len)
                    len2 = args.template_len - len1

                    # Choose random start positions
                    start1 = rng.randint(0, L - len1)
                    start2 = rng.randint(0, L - len2)

                    # Distance between midpoints (linear approximation)
                    mid1 = start1 + len1 // 2
                    mid2 = start2 + len2 // 2
                    dist = abs(mid2 - mid1)

                    if dist < args.min_distance or dist > args.max_distance:
                        continue

                    seg1 = genome[start1:start1 + len1]
                    seg2 = genome[start2:start2 + len2]

                    # Microhomology if required
                    mh_len = 0
                    if mh_required:
                        mh_len = find_microhomology(seg1, seg2, args.min_mh, args.max_mh)
                        if mh_len == 0:
                            continue  # retry if MH required but not found

                    # Build chimera sequence
                    if mh_len > 0:
                        chimera_seq = seg1 + seg2[mh_len:]
                        print(f"[INFO] Microhomology found: {mh_len} bp ({seg1[-mh_len:]} == {seg2[:mh_len]}) "
                              f"for template {num_created+1}")
                    else:
                        chimera_seq = seg1 + seg2
                        print(f"[INFO] No microhomology for template {num_created+1}")

                    # If it's too long due to small overlap, trim to template_len
                    if len(chimera_seq) > args.template_len:
                        chimera_seq = chimera_seq[:args.template_len]

                    name = (
                        f"chimera_{num_created+1}"
                        f"_A{start1+1}-{start1+len1}"
                        f"_B{start2+1}-{start2+len2}"
                        f"_MH{mh_len}"
                    )
                    write_fasta_record(out_f, name, chimera_seq)
                    num_created += 1

                    if num_created % 50 == 0:
                        print(f"[INFO] Created {num_created} chimeric templates...")

                    success = True
                    break  # out of max_tries loop

                if success:
                    break  # out of mh_required loop

                # If still failed after trying without microhomology
                if not mh_required and not success:
                    raise RuntimeError(
                        f"Failed to generate chimera {num_created+1} even without "
                        f"microhomology after {args.max_tries} tries. "
                        "Consider relaxing distance or length constraints."
                    )

    print(f"[DONE] Wrote {num_created} chimeric templates to {out_path}")


if __name__ == "__main__":
    main()
