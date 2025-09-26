# MitoChime

Machine-learning pipeline to detect PCR-induced chimeric reads in mitochondrial sequencing.

## Environments

- **Linux/full**: `environment.yml` (includes bwa/jellyfish/getorganelle, etc.)
- **macOS ARM (M1/M2)**: `environment.arm.yml` (ARM-safe core tools)

Create one:

```bash
# Pick one:
conda env create -f environment.yml        # Linux / CI
conda env create -f environment.arm.yml    # macOS ARM

conda activate mitochime
pip install -e .
pre-commit install
git lfs install
