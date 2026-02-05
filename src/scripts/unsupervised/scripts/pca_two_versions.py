#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# -----------------------------
# Config
# -----------------------------
X_PATH = Path("../../unsupervised/data/processed/X_ready.tsv")
META_PATH = Path("../../unsupervised/data/processed/meta.tsv")

OUT_DIR = Path("unsupervised/reports/pca")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "y_sim"              # post-hoc only
N_COMPONENTS = 10

# Features to always drop if present (your constant list from sanity checks)
DROP_CONSTANTS = [
    "read_length",
    "sa_diff_contig",
    "sa_same_strand_count",
]

# The hijacker feature to test removing (from your loadings)
DROP_HIJACKER = ["breakpoint_read_pos"]

RANDOM_STATE = 42

# -----------------------------
# Helpers
# -----------------------------
def run_pca_and_save(X_df: pd.DataFrame, meta_df: pd.DataFrame, tag: str):
    """Run PCA, save variance/points/loadings/plots under OUT_DIR/tag."""
    tag_dir = OUT_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    # PCA
    pca = PCA(n_components=min(N_COMPONENTS, X_df.shape[1]), random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_df.values)

    pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    pc_df = pd.DataFrame(pcs, columns=pc_cols)

    # Explained variance
    explained = pd.DataFrame({
        "pc": pc_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_ratio_cum": np.cumsum(pca.explained_variance_ratio_)
    })
    explained.to_csv(tag_dir / "pca_explained_variance.csv", index=False)

    # Points
    out_points = pd.concat([meta_df.reset_index(drop=True), pc_df], axis=1)
    out_points.to_csv(tag_dir / "pca_points.tsv", sep="\t", index=False)

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_df.columns,
        columns=pc_cols
    )
    loadings.to_csv(tag_dir / "pca_loadings_all.csv")

    # Top loadings for PC1/PC2
    for pc in ["PC1", "PC2"]:
        if pc in loadings.columns:
            top = loadings[pc].abs().sort_values(ascending=False).head(20)
            top_df = pd.DataFrame({
                "feature": top.index,
                "loading": loadings.loc[top.index, pc].values,
                "abs_loading": top.values
            })
            top_df.to_csv(tag_dir / f"pca_loadings_top20_{pc}.csv", index=False)

    # Plot: PC1 vs PC2
    plt.figure()
    plt.scatter(out_points["PC1"], out_points["PC2"], s=4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA scatter (PC1 vs PC2) [{tag}]")
    plt.tight_layout()
    plt.savefig(tag_dir / "pca_scatter_pc1_pc2.png", dpi=200)
    plt.close()

    # Plot: colored by label (post-hoc)
    if LABEL_COL in out_points.columns:
        plt.figure()
        y = out_points[LABEL_COL].astype(int)
        for cls in sorted(y.unique()):
            mask = (y == cls)
            plt.scatter(out_points.loc[mask, "PC1"], out_points.loc[mask, "PC2"], s=4, label=f"y={cls}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA scatter (PC1 vs PC2), colored by y_sim [{tag}]")
        plt.legend(markerscale=3)
        plt.tight_layout()
        plt.savefig(tag_dir / "pca_scatter_pc1_pc2_by_label.png", dpi=200)
        plt.close()

    # Console summary
    print(f"\n=== {tag} ===")
    print("X shape:", X_df.shape)
    print(explained.head(10).to_string(index=False))

    return explained, loadings


# -----------------------------
# Load
# -----------------------------
X = pd.read_csv(X_PATH, sep="\t")
meta = pd.read_csv(META_PATH, sep="\t")

assert len(X) == len(meta), "X and meta row counts do not match!"

print("Loaded X:", X.shape)
print("Loaded meta:", meta.shape)

if LABEL_COL in meta.columns:
    print("\nLabel counts (post-hoc):")
    print(meta[LABEL_COL].value_counts(dropna=False))

# -----------------------------
# Version A: drop constants only
# -----------------------------
drop_A = [c for c in DROP_CONSTANTS if c in X.columns]
X_A = X.drop(columns=drop_A, errors="ignore")
run_pca_and_save(X_A, meta, tag="A_drop_constants_only")

# -----------------------------
# Version B: drop constants + breakpoint_read_pos
# -----------------------------
drop_B = [c for c in (DROP_CONSTANTS + DROP_HIJACKER) if c in X.columns]
X_B = X.drop(columns=drop_B, errors="ignore")
run_pca_and_save(X_B, meta, tag="B_drop_constants_plus_breakpoint")

print("\nDone. PCA outputs written to:", OUT_DIR)