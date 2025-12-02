#!/usr/bin/env python3
"""
simulation.py

Simulate CLEAN and CHIMERIC Illumina paired-end reads from a mitochondrial
reference, then map them back using minimap2 and process with samtools.

This script is PIPELINED with `make_chimeric_templates.py`:

1. You provide a directory that contains at least ONE FASTA file whose
   filename includes the word "original" (e.g. `NC_039553.1_original.fasta`).
   This should be your true mitochondrial reference (e.g. NC_039553.1).

2. The script looks for a FASTA whose filename includes the word "chimera".
   - If found, it uses that as the chimeric template reference.
   - If NOT found, it automatically calls `make_chimeric_templates.py` to
     generate a chimera FASTA in the same directory (e.g. `NC_039553.1_chimera.fasta`).

3. It then CLEANs the chimera FASTA using awk to:
   - remove whitespace and non-ATCGNatcgn characters,
   - join sequence lines into a single continuous sequence line.

   Command used (conceptually):
       awk 'BEGIN{print_header=1}
            /^>/ {print $0; next}
            {gsub(/[^ATCGNatcgn]/, "", $0); printf "%s", $0}
            END{printf "\\n"}' old.fasta > clean_chimera.fasta

4. Using wgsim, it simulates:
   - CLEAN reads from the ORIGINAL reference   -> `ref1.fastq`, `ref2.fastq`
   - CHIMERIC reads from the CLEANED chimera   -> `chime1.fastq`, `chime2.fastq`

5. It builds a minimap2 index (`ref.mmi`) of the ORIGINAL reference and maps:
   - CLEAN reads     -> `clean.sam`
   - CHIMERIC reads  -> `chimeric.sam`

6. It converts SAM -> BAM, sorts:
   - `clean.sorted.bam`
   - `chimeric.sorted.bam`
   and writes `chimeric.sorted.sam` for inspection.

REQUIREMENTS:
- Conda environments (or equivalent) named:
    - "wgsim"    with `wgsim` installed
    - "minimap2" with `minimap2` installed
    - "samtools" with `samtools` installed
- `awk` available in your shell (standard on Linux/macOS).
- `make_chimeric_templates.py` must be in the SAME directory as this script.

USAGE:
    1. Put your mito reference FASTA in some folder, e.g.:

           /path/to/refs/NC_039553.1_original.fasta

       (Important: filename MUST contain the word "original")

    2. From the folder containing `simulation.py` and `make_chimeric_templates.py`, run:

           python simulation.py

    3. When prompted:

           === ENTER REFERENCES DIRECTORY ===
           Paste the directory path containing your reference FASTA files:

       Paste the path to your refs folder, e.g.:

           /path/to/refs

    4. The script will:
       - auto-generate the chimera FASTA if missing,
       - clean the chimera FASTA with awk,
       - simulate reads,
       - map and sort BAMs,
       - leaving you with BAM/SAM/FASTQ files ready for feature extraction.
"""

import os
import sys
import subprocess
from pathlib import Path

NUM_READS = 10000  # Number of reads to simulate (read pairs)

## ATTENTION !!!! EME
## INCLUDE the word "original" in original reference (from ncbi) name file
## INCLUDE the word "chimera" in chimera reference (generated or manual) name file :)


def find_reference(ref_dir, keyword):
    """Find the first FASTA file containing the keyword in its name."""
    ref_dir = Path(ref_dir)
    fasta_files = list(ref_dir.glob(f"*{keyword}*.fasta")) + list(
        ref_dir.glob(f"*{keyword}*.fa")
    )
    return str(fasta_files[0]) if fasta_files else None


def run_cmd(cmd, env_name=None):
    """Run a shell command sequentially and check for errors."""
    if env_name:
        cmd = f"conda run -n {env_name} {cmd}"
    print(f"\n[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed:\n{result.stderr}")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


# Convert SAM to BAM safely by removing blank lines first
def sam_to_bam_safe(sam_file, bam_file, env_name=None):
    fixed_sam = sam_file.replace(".sam", ".fixed.sam")
    # Remove blank lines
    with open(sam_file, "r") as f_in, open(fixed_sam, "w") as f_out:
        for line in f_in:
            if line.strip():  # skip empty lines
                f_out.write(line)
    # Convert to BAM
    run_cmd(f"samtools view -bS {fixed_sam} -o {bam_file}", env_name=env_name)
    return bam_file


def main():
    # --------------------------
    # Get reference directory
    # --------------------------
    print("\n=== ENTER REFERENCES DIRECTORY ===\n")
    ref_dir = input("Paste the directory path containing your reference FASTA files: ").strip()
    if not os.path.isdir(ref_dir):
        print(f"[ERROR] Directory not found: {ref_dir}")
        sys.exit(1)

    # --------------------------
    # Find original reference
    # --------------------------
    original_ref = find_reference(ref_dir, "original")
    if not original_ref:
        print("[ERROR] Could not find an ORIGINAL reference FASTA (filename must contain 'original').")
        sys.exit(1)

    print(f"[INFO] Found original reference: {original_ref}")

    # --------------------------
    # Find or generate chimera reference
    # --------------------------
    chimera_ref = find_reference(ref_dir, "chimera")

    if not chimera_ref:
        print("[INFO] No chimera reference found. Generating chimeric templates now...")

        ref_dir_path = Path(ref_dir)
        original_name = Path(original_ref).name

        # Default chimera output name based on original
        default_chim_name = original_name.replace("original", "chimera")
        if default_chim_name == original_name:
            # if user didn't put 'original' in name, just prepend
            default_chim_name = f"chimera_{original_name}"

        chimera_path = ref_dir_path / default_chim_name

        # Path to make_chimeric_templates.py (assumed to be in same folder as this script)
        script_dir = Path(__file__).resolve().parent
        generator_script = script_dir / "make_chimeric_templates.py"

        if not generator_script.is_file():
            print(f"[ERROR] Could not find make_chimeric_templates.py at {generator_script}")
            sys.exit(1)

        # Call the generator with sensible defaults
        gen_cmd = (
            f"python {generator_script} "
            f"--ref {original_ref} "
            f"--out {chimera_path} "
            f"--num-templates 1000 "
            f"--template-len 300 "
            f"--min-distance 500 "
            f"--max-distance 8000 "
            f"--min-mh 3 "
            f"--max-mh 8"
        )
        run_cmd(gen_cmd, env_name=None)

        chimera_ref = str(chimera_path)

    print(f"[INFO] Using chimera reference: {chimera_ref}")

    # --------------------------
    # Clean chimera FASTA with awk
    # --------------------------
    print("\n=== Cleaning chimera FASTA (awk) ===")
    chimera_path = Path(chimera_ref)
    chimera_clean_path = chimera_path.with_name(chimera_path.stem + "_clean.fasta")

    awk_cmd = (
        "awk 'BEGIN{print_header=1}"
        " /^>/ {print $0; next}"
        " {gsub(/[^ATCGNatcgn]/, \"\", $0); printf \"%s\", $0}"
        " END{printf \"\\n\"}' "
        f"\"{chimera_ref}\" > \"{chimera_clean_path}\""
    )
    # awk is usually in the base shell, so no conda env_name
    run_cmd(awk_cmd, env_name=None)

    # Use the cleaned chimera FASTA from this point on
    chimera_ref = str(chimera_clean_path)
    print(f"[INFO] Using cleaned chimera reference: {chimera_ref}")

    # --------------------------
    # Generate WGSIM reads
    # --------------------------
    print("\n=== Generating reads with wgsim ===")

    run_cmd(
        f"wgsim -1 150 -2 150 -r 0 -R 0 -X 0 -e 0.05 -N {NUM_READS} "
        f"{original_ref} ref1.fastq ref2.fastq",
        env_name="wgsim",
    )
    run_cmd(
        f"wgsim -1 150 -2 150 -r 0 -R 0 -X 0 -e 0 -N {NUM_READS} "
        f"{chimera_ref} chime1.fastq chime2.fastq",
        env_name="wgsim",
    )

    # --------------------------
    # Build minimap2 index
    # --------------------------
    print("\n=== Building minimap2 index ===")
    run_cmd(f"minimap2 -d ref.mmi {original_ref}", env_name="minimap2")

    # --------------------------
    # Map reads sequentially
    # --------------------------
    print("\n=== Mapping CLEAN reads ===")
    run_cmd(
        "minimap2 -ax sr -t 8 ref.mmi ref1.fastq ref2.fastq > clean.sam",
        env_name="minimap2",
    )

    print("\n=== Mapping CHIMERIC reads ===")
    run_cmd(
        "minimap2 -ax sr -t 8 ref.mmi chime1.fastq chime2.fastq > chimeric.sam",
        env_name="minimap2",
    )

    # --------------------------
    # Convert SAM to BAM and sort
    # --------------------------
    print("\n=== Processing CLEAN alignments ===")
    print("Converting clean.sam to BAM safely...")
    sam_to_bam_safe("clean.sam", "clean.bam", env_name="samtools")
    print("Sorting clean.bam...")
    run_cmd("samtools sort clean.bam -o clean.sorted.bam", env_name="samtools")

    print("\n=== Processing CHIMERIC alignments ===")
    print("Converting chimeric.sam to BAM safely...")
    sam_to_bam_safe("chimeric.sam", "chimeric.bam", env_name="samtools")
    print("Sorting chimeric.bam...")
    run_cmd("samtools sort chimeric.bam -o chimeric.sorted.bam", env_name="samtools")

    # Export SAM from sorted BAM (for inspection)
    run_cmd(
        "samtools view chimeric.sorted.bam -o chimeric.sorted.sam",
        env_name="samtools",
    ) 

    print("\nPipeline complete! All FASTQ, SAM, and BAM files are ready.\n")


if __name__ == "__main__":
    main()