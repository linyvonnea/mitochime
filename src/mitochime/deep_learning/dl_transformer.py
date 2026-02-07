# src/mitochime/deep_learning/dl_transformer.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sine/cos positional encoding.
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
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class KmerTransformer(nn.Module):
    """
    Transformer encoder over k-mer tokens.

    Input:  (B, T) token ids (int64)
    Output: (B, 2)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 2)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        # tok: (B, T)
        x = self.emb(tok)                 # (B, T, d_model)
        x = self.pos(x)

        # mask padding tokens
        key_padding_mask = (tok == self.pad_id)  # True where pad

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)

        # mean pool over non-pad positions
        nonpad = (~key_padding_mask).float()  # (B, T)
        denom = nonpad.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        pooled = (x * nonpad.unsqueeze(-1)).sum(dim=1) / denom  # (B, d_model)

        pooled = self.norm(pooled)
        logits = self.cls(pooled)
        return logits