# MitoChime Local Computer Lab Setup and Run Guide

This guide assumes you will run everything from the **project root** after cloning the repo.

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPO_URL> mitochime
cd mitochime
```

If your repo is private, use your authorized GitHub account or PAT.

## 2. Create the environment

### Option A: conda or miniconda
```bash
conda create -n mitochime python=3.10 -y
conda activate mitochime
```

### Option B: venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install system or bioinformatics tools

If the lab already has these installed, skip this section.

### With conda
```bash
conda install -c conda-forge -c bioconda -y   minimap2 samtools seqkit fastp
```

These are mainly for the full pipeline.  
For the **local retraining notebook** below, the key Python packages matter more than the external tools.

## 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib joblib pysam biopython
pip install torch torchvision
pip install xgboost lightgbm catboost
```

If your lab machines already have some of these, pip will just confirm or update them.

## 5. Confirm project layout

From the project root, these should exist:

```bash
ls src/mitochime
ls src/mitochime/deep_learning
ls data/processed
```

Important files expected:
- `src/mitochime/train_all_models.py`
- `src/mitochime/model_zoo.py`
- `src/mitochime/deep_learning/train_deep.py`
- `src/mitochime/deep_learning/dl_cnn.py`
- `src/mitochime/deep_learning/dl_rnn_kmer.py`
- `data/processed/PAIR_train_noq.tsv`
- `data/processed/PAIR_test_noq.tsv`
- `data/processed/PAIR_train_seq_L150.tsv`
- `data/processed/PAIR_test_seq_L150.tsv`
- `data/processed/cv_seq_L150_seed42/fold0_train_seq.tsv` to `fold4_val_seq.tsv`

## 6. Important local code note

Your current `src/mitochime/deep_learning/train_deep.py` still contains an old import:

```python
from .dl_rnn import RNNClassifier
```

but `dl_rnn.py` no longer exists in the project.  
For local runs, clean this first.

### Edit `src/mitochime/deep_learning/train_deep.py`
Comment out:
```python
from .dl_rnn import RNNClassifier
```

Also remove or disable the old `rnn_lstm` / `rnn_gru` branch so only the working modes remain:
- `cnn`
- `rnn_kmer_gru`
- optionally transformer if you still want it later

A safe replacement is:

```python
elif args.mode in {"rnn_lstm", "rnn_gru"}:
    raise ValueError("Old rnn_lstm/rnn_gru modes are disabled; use rnn_kmer_gru instead")
```

## 7. Run the notebook

From the project root:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open the notebook file:
- `mitochime_local_lab_runs.ipynb`

## 8. What the notebook covers

The notebook includes:
- environment check
- path verification
- Gradient Boosting CV + held-out test
- CNN 5-fold CV + held-out test
- BiGRU 5-fold CV + held-out test
- timing capture
- summary tables
- zip-ready outputs folder

## 9. If you prefer direct terminal commands instead of the notebook

### Gradient Boosting and full model zoo script
```bash
PYTHONPATH=src python3 -m mitochime.train_all_models   --train data/processed/PAIR_train_noq.tsv   --test data/processed/PAIR_test_noq.tsv   --models-dir models_PAIR   --reports-dir reports/metrics_PAIR
```

### CNN held-out test
```bash
PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep   --mode cnn   --train-tsv data/processed/PAIR_train_seq_L150.tsv   --test-tsv data/processed/PAIR_test_seq_L150.tsv   --L 150   --epochs 30   --batch 128   --lr 0.001   --seed 42   --select-best-by f1   --weight-decay 1e-4   --out-dir models/deep/cnn_holdout30_L150_seed42   --reports-dir reports/deep/cnn_holdout30_L150_seed42   --save-predictions
```

### BiGRU held-out test
```bash
PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep   --mode rnn_kmer_gru   --train-tsv data/processed/PAIR_train_seq_L150.tsv   --test-tsv data/processed/PAIR_test_seq_L150.tsv   --L 150   --k 4   --L-kmers 147   --embed-dim 64   --hidden 256   --rnn-layers 1   --bidirectional   --pool last   --epochs 30   --batch 128   --lr 0.001   --seed 42   --select-best-by f1   --weight-decay 1e-4   --out-dir models/deep/bigru_holdout30_L150_seed42   --reports-dir reports/deep/bigru_holdout30_L150_seed42   --save-predictions
```

## 10. Notes for reporting

For the thesis or class report, separate:
- **CV results** on the training set
- **held-out test results** on the final train/test split

For CNN, if your study selected **epoch 25** even though the best numeric epoch was later, keep that clearly stated in the writeup.

## 11. Outputs to collect

After runs, archive:
- `outputs_local/`
- `reports/`
- `models/`
- notebook screenshots or exported HTML if needed

## 12. Troubleshooting

### `ModuleNotFoundError: mitochime`
Run from project root and use:
```bash
PYTHONPATH=src
```

### `No module named xgboost/lightgbm/catboost`
Install them with:
```bash
pip install xgboost lightgbm catboost
```

### `No module named dl_rnn`
That means `train_deep.py` was not patched yet.

### Torch CUDA issues
If the lab machine has no working GPU, the notebook still works on CPU, but deep learning will be slower.
