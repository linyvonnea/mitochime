# src/mitochime/deep_learning/dl_cnn.py
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D CNN for one-hot DNA (5, L).
    Input:  (B, 5, L)
    Output: (B, 2)
    """

    def __init__(self, in_ch: int = 5, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # -> (B, 256, 1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),            # -> (B, 256)
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x