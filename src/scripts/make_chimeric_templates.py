#!/usr/bin/env python3
"""
make_chimeric_templates.py

Generate chimeric templates from a mitochondrial reference by
concatenating two non-adjacent segments with optional microhomology.
"""

import argparse
import random
from pathlib import Path
from textwrap import wrap

def read_fasta_single(path):
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

def write_fasta_record(handle, name, seq):
    """Write one FASTA record with wrapped sequence."""
    handle.write(f">{name}\n")
    for chunk in wrap(seq, 80):
        handle.write(chunk + "\n")

def find_microhomology(seg1, seg2, min_mh, max_mh):
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
    parser.add_argument("--ref", required=True,
                        help="Path to original mito FASTA (NC_039553.1_original.fasta)")
    parser.add_argument("--out", required=True,
                        help="Output FASTA path for chimeric templates")
    parser.add_argument("--num-templates", type=int, default=1000,
                        help="Number of chimeric templates to generate")
    parser.add_argument("--template-len", type=int, default=300,
                        help="Length of each chimeric template (approx; "
                             "actual length may be slightly shorter due to overlap)")
    parser.add_argument("--min-distance", type=int, default=500,
                        help="Minimum distance between segment midpoints in reference")
    parser.add_argument("--max-distance", type=int, default=8000,
                        help="Maximum distance between segment midpoints in reference")
    parser.add_argument("--min-mh", type=int, default=3,
                        help="Minimum microhomology length")
    parser.add_argument("--max-mh", type=int, default=8,
                        help="Maximum microhomology length")
    parser.add_argument("--max-tries", type=int, default=100,
                        help="Max random attempts per template before relaxing constraints")
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
                    len1 = rng.randint(min_seg_len,
                                       args.template_len - min_seg_len)
                    len2 = args.template_len - len1

                    # Choose random start positions
                    start1 = rng.randint(0, L - len1)
                    start2 = rng.randint(0, L - len2)

                    # Distance between midpoints
                    mid1 = start1 + len1 // 2
                    mid2 = start2 + len2 // 2
                    dist = abs(mid2 - mid1)

                    if dist < args.min_distance or dist > args.max_distance:
                        continue

                    seg1 = genome[start1:start1 + len1]
                    seg2 = genome[start2:start2 + len2]

                    # Microhomology if required
                    if mh_required:
                        mh_len = find_microhomology(
                            seg1, seg2, args.min_mh, args.max_mh
                        )
                        if mh_len == 0:
                            continue
                    else:
                        mh_len = 0

                    # Build chimera sequence
                    if mh_len > 0:
                        chimera_seq = seg1 + seg2[mh_len:]
                    else:
                        chimera_seq = seg1 + seg2

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
                    break  # out of tries loop

                if success:
                    break  # out of mh_required loop

                # If we get here: this mh_required setting failed after max_tries
                if mh_required:
                    print(
                        f"[WARN] No microhomology ≥{args.min_mh} bp found "
                        f"for template {num_created+1} after {args.max_tries} tries; "
                        "retrying without microhomology."
                    )
                else:
                    raise RuntimeError(
                        f"Failed to generate chimera {num_created+1} even without "
                        f"microhomology after {args.max_tries} tries. "
                        "Consider relaxing distance or length constraints."
                    )

    print(f"[DONE] Wrote {num_created} chimeric templates to {out_path}")

if __name__ == "__main__":
    main()