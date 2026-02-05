#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------
# Config
# -----------------------------
X_PATH = Path("X_ready.tsv")
META_PATH = Path("meta.tsv")

OUT_DIR = Path("unsupervised/reports/isoforest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
# contamination = expected outlier fraction; you can sweep later
CONTAMINATION = 0.05

# Precision@K settings
TOP_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10]  # 0.5%, 1%, 2%, 5%, 10%

# -----------------------------
# Load
# -----------------------------
X = pd.read_csv(X_PATH, sep="\t")
meta = pd.read_csv(META_PATH, sep="\t")

assert len(X) == len(meta), "X and meta row counts do not match!"
assert "read_uid" in meta.columns, "meta.tsv must include read_uid"
assert "y_sim" in meta.columns, "meta.tsv must include y_sim for post-hoc eval"

y = meta["y_sim"].astype(int).values

# -----------------------------
# Fit Isolation Forest
# -----------------------------
iso = IsolationForest(
    n_estimators=400,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

iso.fit(X.values)

# sklearn returns:
#   decision_function: higher = more normal
#   score_samples: higher = more normal
# We want anomaly_score: higher = more anomalous
normality = iso.score_samples(X.values)
anomaly_score = -normality

# predicted labels: -1 outlier, +1 inlier
pred = iso.predict(X.values)

# -----------------------------
# Save per-read results
# -----------------------------
out = meta.copy()
out["anomaly_score"] = anomaly_score
out["iso_label"] = pred  # -1 outlier, +1 inlier
out.to_csv(OUT_DIR / "isoforest_scores.tsv", sep="\t", index=False)

# -----------------------------
# Post-hoc evaluation
# -----------------------------
auroc = roc_auc_score(y, anomaly_score)
auprc = average_precision_score(y, anomaly_score)

print("=== Isolation Forest (post-hoc eval using y_sim) ===")
print("contamination:", CONTAMINATION)
print("AUROC:", round(auroc, 4))
print("AUPRC:", round(auprc, 4))

# Precision@K
rank = np.argsort(-anomaly_score)  # descending anomaly
for frac in TOP_FRACS:
    k = max(1, int(len(y) * frac))
    topk = y[rank[:k]]
    prec = topk.mean()  # since y=1 is chimera
    print(f"Precision@top {frac*100:.1f}% (k={k}): {prec:.4f}")

# Save summary
summary = {
    "contamination": CONTAMINATION,
    "AUROC": auroc,
    "AUPRC": auprc,
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "isoforest_summary.csv", index=False)

print("Wrote:", OUT_DIR / "isoforest_scores.tsv")
print("Wrote:", OUT_DIR / "isoforest_summary.csv")