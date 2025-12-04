#!/usr/bin/env python3
"""
Utility functions for loading the per-read feature table
and preparing X (features) and y (labels) for ML.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


NON_FEATURE_COLS = [
    "read_id",
    "label",
    "ref_name",   # drop textual contig name for now
    "cigar",      # CIGAR string is textual; we keep numeric alignment stats instead
]


def load_feature_table(path: str) -> pd.DataFrame:
    """
    Load the merged TSV created by extract_features.py.
    """
    df = pd.read_csv(path, sep="\t")
    return df


def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Turn the raw feature DataFrame into (X, y).

    - Maps strand '+'/'-' to 1/0.
    - Drops non-numeric / non-feature columns.
    """
    df = df.copy()

    # Ensure label is int (0 = clean, 1 = chimeric)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the feature table.")
    df["label"] = df["label"].astype(int)

    # Encode strand as numeric
    if "strand" in df.columns:
        df["strand"] = df["strand"].map({"+": 1, "-": 0}).astype("float32")

    # Drop clearly non-feature columns
    X = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    y = df["label"]

    return X, y