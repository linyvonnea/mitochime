# src/mitochime/deep_learning/dl_transformer.py

# PYTHONPATH=src python3 -m mitochime.deep_learning.train_deep \
#   --mode transformer \
#   --train-tsv data/processed/PAIR_train_seq_L150.tsv \
#   --test-tsv  data/processed/PAIR_test_seq_L150.tsv \
#   --L 150 \
#   --k 6 \
#   --L-kmers 256 \
#   --d-model 128 \
#   --layers 4 \
#   --heads 4 \
#   --epochs 15 \
#   --batch 128 \
#   --lr 1e-3 \
#   --out-dir models_dl_PAIR \
#   --reports-dir reports/metrics_dl_PAIR \
#   --save-predictions
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sine/cos positional encoding.

    Adds position information to token embeddings so the model knows order.
    Input/Output: (B, T, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class KmerTransformer(nn.Module):
    """
    Transformer encoder over k-mer tokens.

    Data (your setup):
      - raw sequence length L = 150 bases (from make_seq_tsv / SeqConfig.L)
      - tokenize into k-mers (default k=6) -> T = L - k + 1 = 145 tokens
      - then pad/truncate to fixed length T = L_kmers (default 256)

    Input:  tok (B, T) int64 token ids, where:
      - 0 is UNK/PAD (any k-mer with non-ACGT OR padding)
      - 1..4^k are real A/C/G/T k-mers

    Output: logits (B, 2)
    """

    def __init__(
        self,
        vocab_size: int,          # (4**k) + 1
        d_model: int = 128,       # embedding / hidden width
        nhead: int = 4,           # attention heads
        num_layers: int = 4,      # encoder blocks
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,       # should match L_kmers
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        # Token embedding: (B,T) -> (B,T,d_model)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Add positional encoding so token order is represented
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        # Encoder block = multi-head self-attention + MLP (FFN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        # Stack encoder blocks
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 2)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        # tok: (B, T)
        x = self.emb(tok)          # (B, T, d_model)
        x = self.pos(x)            # (B, T, d_model)

        # Padding mask: True where token == pad_id (0)
        key_padding_mask = (tok == self.pad_id)  # (B, T)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)

        # Mean pool over NON-pad positions
        nonpad = (~key_padding_mask).float()  # (B, T)
        denom = nonpad.sum(dim=1, keepdim=True).clamp_min(1.0)      # (B, 1)
        pooled = (x * nonpad.unsqueeze(-1)).sum(dim=1) / denom      # (B, d_model)

        pooled = self.norm(pooled)
        logits = self.cls(pooled)  # (B, 2)
        return logits