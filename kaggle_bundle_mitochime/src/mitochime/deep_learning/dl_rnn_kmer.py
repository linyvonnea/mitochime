# src/mitochime/deep_learning/dl_rnn_kmer.py

""" 

BIGRU CV:
for k in 0 1 2 3 4; do
  PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep \
    --mode rnn_kmer_gru \
    --train-tsv data/processed/cv_seq_L150_seed42/fold${k}_train_seq.tsv \
    --test-tsv  data/processed/cv_seq_L150_seed42/fold${k}_val_seq.tsv \
    --L 150 --k 4 --L-kmers 147 \
    --embed-dim 64 --hidden 256 --rnn-layers 1 --bidirectional --pool last \
    --epochs 30 --batch 128 --lr 1e-3 \
    --seed 42 --select-best-by f1 --weight-decay 1e-4 \
    --out-dir models/deep/rnnkmer_bigru_cv30_L150_seed42/fold${k} \
    --reports-dir reports/deep/rnnkmer_bigru_cv30_L150_seed42/fold${k} \
    --save-predictions
done

BIGRU HELDOUT:
PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep \
  --mode rnn_kmer_gru \
  --train-tsv data/processed/PAIR_train_seq_L150.tsv \
  --test-tsv  data/processed/PAIR_test_seq_L150.tsv \
  --L 150 --k 4 --L-kmers 147 \
  --embed-dim 64 --hidden 256 --rnn-layers 1 --bidirectional --pool last \
  --epochs 30 --batch 128 --lr 1e-3 \
  --seed 42 --select-best-by f1 --weight-decay 1e-4 \
  --out-dir models/deep/rnnkmer_bigru_final_L150_seed42 \
  --reports-dir reports/deep/rnnkmer_bigru_final_L150_seed42 \
  --save-predictions


"""
from __future__ import annotations

import torch
import torch.nn as nn


class RNNKmerClassifier(nn.Module):
    """
    RNN over k-mer token IDs (Embedding -> GRU/LSTM -> pool -> head)

    Input:  x = (B, L_kmers) int64 tokens (0..vocab_size-1)
    Output: logits (B, 2)
    """

    def __init__(
        self,
        rnn_type: str,               # "lstm" or "gru"
        vocab_size: int,             # (4^k)+1  (UNK=0)
        embed_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pool: str = "last",          # "last" | "mean" | "max"
        pad_idx: int = 0,
    ):
        super().__init__()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if pool not in {"last", "mean", "max"}:
            raise ValueError("pool must be 'last', 'mean', or 'max'")

        self.rnn_type = rnn_type
        self.pool = pool
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,         # (B, T, C)
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) tokens
        emb = self.embed(x)          # (B, T, E)

        if self.rnn_type == "lstm":
            out, (h_n, _) = self.rnn(emb)
        else:
            out, h_n = self.rnn(emb)

        if self.pool == "mean":
            feat = out.mean(dim=1)
        elif self.pool == "max":
            feat = out.max(dim=1).values
        else:
            # last hidden state(s)
            if self.bidirectional:
                h_f = h_n[-2]        # (B, H)
                h_b = h_n[-1]        # (B, H)
                feat = torch.cat([h_f, h_b], dim=1)  # (B, 2H)
            else:
                feat = h_n[-1]       # (B, H)

        return self.head(feat)