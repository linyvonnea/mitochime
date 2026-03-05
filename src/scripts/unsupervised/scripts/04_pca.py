#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
X_PATH = Path("unsupervised/data/processed/X_ready.tsv")
META_PATH = Path("unsupervised/data/processed/meta.tsv")

OUT_DIR = Path("unsupervised/reports/pca")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "read_uid"
LABEL_COL = "y_sim"   # optional (post-hoc only)

N_COMPONENTS = 10     # compute first 10 PCs for reporting (even if plotting only first 2)

# -----------------------------
# Load
# -----------------------------
X = pd.read_csv(X_PATH, sep="\t")
meta = pd.read_csv(META_PATH, sep="\t")

assert len(X) == len(meta), "X and meta row counts do not match!"

print("Loaded X:", X.shape)
print("Loaded meta:", meta.shape)

# -----------------------------
# PCA
# -----------------------------
# X_ready is already robust-scaled. PCA can run directly.
# (Optional) If you want classical PCA behavior, you can standardize again.
# We'll keep it OFF by default since you already scaled with robust scaling.
X_mat = X.values

pca = PCA(n_components=min(N_COMPONENTS, X.shape[1]), random_state=42)
pcs = pca.fit_transform(X_mat)

# -----------------------------
# Save explained variance
# -----------------------------
explained = pd.DataFrame({
    "pc": [f"PC{i+1}" for i in range(pca.n_components_)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "explained_variance_ratio_cum": np.cumsum(pca.explained_variance_ratio_)
})
explained.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False)

# -----------------------------
# Save PC coordinates
# -----------------------------
pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
pc_df = pd.DataFrame(pcs, columns=pc_cols)

out_points = pd.concat([meta.reset_index(drop=True), pc_df], axis=1)
out_points.to_csv(OUT_DIR / "pca_points.tsv", sep="\t", index=False)

# -----------------------------
# Save loadings (feature contributions)
# -----------------------------
# loadings = components_.T : (n_features x n_components)
loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=pc_cols
)

# Top 20 absolute loadings for PC1 and PC2
for pc in ["PC1", "PC2"]:
    if pc in loadings.columns:
        top = loadings[pc].abs().sort_values(ascending=False).head(20)
        top_df = pd.DataFrame({
            "feature": top.index,
            "loading": loadings.loc[top.index, pc].values,
            "abs_loading": top.values
        })
        top_df.to_csv(OUT_DIR / f"pca_loadings_top20_{pc}.csv", index=False)

# -----------------------------
# Plot PC1 vs PC2 (no label)
# -----------------------------
plt.figure()
plt.scatter(out_points["PC1"], out_points["PC2"], s=4)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA scatter (PC1 vs PC2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_scatter_pc1_pc2.png", dpi=200)
plt.close()

# -----------------------------
# Plot PC1 vs PC2 colored by label (if available)
# -----------------------------
if LABEL_COL in out_points.columns:
    plt.figure()
    y = out_points[LABEL_COL].astype(int)

    # plot class 0 and 1 separately (no custom colors, default matplotlib cycle)
    for cls in sorted(y.unique()):
        mask = (y == cls)
        plt.scatter(out_points.loc[mask, "PC1"], out_points.loc[mask, "PC2"], s=4, label=f"y={cls}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA scatter (PC1 vs PC2), colored by y_sim (post-hoc)")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_scatter_pc1_pc2_by_label.png", dpi=200)
    plt.close()

print("Wrote PCA outputs to:", OUT_DIR)
print(explained.head(10).to_string(index=False))

print(df["y_sim"].value_counts(dropna=False))