# src/mitochime/deep_learning/dl_data.py
'''PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep \
  --mode cnn \
  --train-tsv data/processed/PAIR_train_seq_L300.tsv \
  --test-tsv  data/processed/PAIR_test_seq_L300.tsv \
  --L 300 \
  --epochs 15 \
  --batch 128
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ----------------------------
# Config
# ----------------------------

@dataclass
class SeqConfig:
    L: int = 300              # length for CNN
    use_qual: bool = False    # kept for compatibility; not used in seq-tsv mode
    k: int = 6                # k for transformer k-mers
    L_kmers: int = 256        # max k-mer tokens length


# ----------------------------
# TSV helpers
# ----------------------------

def load_split_tsv(path: str) -> Dict[str, int]:
    """
    Loads a split TSV (train_noq.tsv/test_noq.tsv) and returns {read_id: label}.
    Ensures uniqueness by read_id.
    """
    df = pd.read_csv(path, sep="\t")
    if "read_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns read_id and label")

    df = df[["read_id", "label"]].copy()
    df["read_id"] = df["read_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["read_id"], keep="first")

    return dict(zip(df["read_id"], df["label"]))


def load_seq_tsv(path: str, L: int) -> pd.DataFrame:
    """
    Loads the sequence TSV created by make_seq_tsv:
      read_id, label, seq
    Ensures seq length is exactly L (trim/pad just in case).
    """
    df = pd.read_csv(path, sep="\t")
    for col in ["read_id", "label", "seq"]:
        if col not in df.columns:
            raise ValueError(f"{path} must contain columns: read_id, label, seq")

    df = df[["read_id", "label", "seq"]].copy()
    df["read_id"] = df["read_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["seq"] = df["seq"].astype(str).str.upper()

    # safety: enforce exact length L
    df["seq"] = df["seq"].apply(lambda s: (s[:L] + ("N" * max(0, L - len(s)))) if len(s) != L else s)

    # unique read_id (should already be unique)
    df = df.drop_duplicates(subset=["read_id"], keep="first").reset_index(drop=True)
    return df


# ----------------------------
# Encoding for CNN (5 channels)
# ----------------------------

_BASE_TO_CH = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,  # catch-all
}

def one_hot_5ch(seq: str) -> np.ndarray:
    """
    Returns (5, L) float32 one-hot for A,C,G,T,N.
    Unknown bases map to N.
    """
    L = len(seq)
    x = np.zeros((5, L), dtype=np.float32)
    for i, b in enumerate(seq):
        ch = _BASE_TO_CH.get(b, 4)
        x[ch, i] = 1.0
    return x


# ----------------------------
# Encoding for Transformer (k-mers)
# ----------------------------

def _kmer_to_id(kmer: str) -> int:
    """
    Base-4 encoding A,C,G,T -> 0..(4^k - 1).
    Any k-mer containing non-ACGT -> UNK id (0).
    We reserve 0 as UNK, and shift real kmers by +1.
    """
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    v = 0
    for ch in kmer:
        if ch not in m:
            return 0  # UNK
        v = v * 4 + m[ch]
    return v + 1  # shift by 1 so UNK=0


def seq_to_kmer_tokens(seq: str, k: int, L_kmers: int) -> np.ndarray:
    """
    Convert sequence into a fixed-length token array of length L_kmers.
    Pads with 0 (UNK) or truncates.
    """
    tokens = []
    n = len(seq)
    if n >= k:
        for i in range(n - k + 1):
            tokens.append(_kmer_to_id(seq[i : i + k]))
    else:
        tokens = []

    if len(tokens) >= L_kmers:
        tokens = tokens[:L_kmers]
    else:
        tokens = tokens + [0] * (L_kmers - len(tokens))

    return np.array(tokens, dtype=np.int64)


# ----------------------------
# Dataset
# ----------------------------

class ReadSeqDataset(Dataset):
    """
    Deep learning dataset that reads from the seq TSV (train_seq_L300.tsv etc.)
    and returns:
      - cnn: (5, L) float tensor
      - transformer: (L_kmers,) int64 token ids
    """

    def __init__(
        self,
        seq_tsv_path: str,
        mode: str,
        cfg: SeqConfig,
    ):
        if mode not in {"cnn", "transformer"}:
            raise ValueError("mode must be 'cnn' or 'transformer'")

        self.mode = mode
        self.cfg = cfg

        df = load_seq_tsv(seq_tsv_path, L=cfg.L)
        self.read_ids = df["read_id"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.seqs = df["seq"].tolist()

        print(
            f"[ReadSeqDataset] loaded={len(df):,} from {seq_tsv_path} | "
            f"mode={mode} | L={cfg.L} | k={cfg.k} | L_kmers={cfg.L_kmers}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        seq = self.seqs[idx]

        if self.mode == "cnn":
            x_np = one_hot_5ch(seq)              # (5, L)
            x = torch.from_numpy(x_np)           # float32
            return x, y

        # transformer
        tok = seq_to_kmer_tokens(seq, k=self.cfg.k, L_kmers=self.cfg.L_kmers)  # (L_kmers,)
        x = torch.from_numpy(tok)  # int64
        return x, y