import os
import sys
import subprocess
from pathlib import Path

NUM_READS = 50  # Number of reads to simulate

## ATTENTION !!!! EME
## INCLUDE the word "original" in original reference (from ncbi) name file
## INCLUDE the word "chimera" in chimera reference (manually created by you) name file

def find_reference(ref_dir, keyword):
    """Find the first FASTA file containing the keyword in its name."""
    fasta_files = list(Path(ref_dir).glob(f"*{keyword}*.fasta")) + list(Path(ref_dir).glob(f"*{keyword}*.fa"))
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

    original_ref = find_reference(ref_dir, "original")
    chimera_ref = find_reference(ref_dir, "chimera")

    if not original_ref or not chimera_ref:
        print("[ERROR] Could not find both reference files.")
        sys.exit(1)

    print(f"\nDetected references:\n  Original: {original_ref}\n  Chimera:  {chimera_ref}")

    # --------------------------
    # Generate WGSIM reads
    # --------------------------
    run_cmd(f"wgsim -1 150 -2 150 -r 0 -R 0 -X 0 -e 0 -N {NUM_READS} {original_ref} ref1.fastq ref2.fastq", env_name="wgsim")
    run_cmd(f"wgsim -1 150 -2 150 -r 0 -R 0 -X 0 -e 0 -N {NUM_READS} {chimera_ref} chime1.fastq chime2.fastq", env_name="wgsim")

    # --------------------------
    # Build minimap2 index
    # --------------------------
    run_cmd(f"minimap2 -d ref.mmi {original_ref}", env_name="minimap2")

    # --------------------------
    # Map reads sequentially
    # --------------------------
    run_cmd(f"minimap2 -ax sr -t 8 ref.mmi ref1.fastq ref2.fastq > clean.sam", env_name="minimap2")
    run_cmd(f"minimap2 -ax sr -t 8 ref.mmi chime1.fastq chime2.fastq > chimeric.sam", env_name="minimap2")

    # --------------------------
    # Convert SAM to BAM and sort
    # --------------------------
    print("Converting clean.sam to BAM safely...")
    sam_to_bam_safe("clean.sam", "clean.bam", env_name="samtools")
    print("Sorting clean.bam...")
    run_cmd("samtools sort clean.bam -o clean.sorted.bam", env_name="samtools")

    print("Converting chimeric.sam to BAM safely...")
    sam_to_bam_safe("chimeric.sam", "chimeric.bam", env_name="samtools")
    print("Sorting chimeric.bam...")
    run_cmd("samtools sort chimeric.bam -o chimeric.sorted.bam", env_name="samtools")


    run_cmd("samtools view chimeric.sorted.bam -o chimeric.sorted.sam", env_name="samtools")

    print("\nPipeline complete! All FASTQ, SAM, and BAM files are ready.\n")

if __name__ == "__main__":
    main()
