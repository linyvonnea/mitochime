#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DEFAULT_SELECTED4 = [
    "total_clipped_bases",
    "kmer_js_divergence",
    "kmer_cosine_diff",
    "softclip_left",
]

LABEL_CANDIDATES = ["label", "y", "target", "class"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["clean", "chimeric"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
        "n_neg": int(len(y_true) - np.sum(y_true)),
    }


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p, sep="\t")


def resolve_feature_list(args) -> List[str]:
    if args.features is not None and len(args.features) > 0:
        return list(args.features)
    return list(DEFAULT_SELECTED4)


def resolve_label_col(df: pd.DataFrame, explicit: Optional[str]) -> str:
    """
    If user provided --label-col, use it (but validate).
    Otherwise auto-detect among LABEL_CANDIDATES.
    """
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Label column '{explicit}' not found. Columns seen: {list(df.columns)}"
            )
        return explicit

    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand

    raise ValueError(
        f"Could not auto-detect label column. Expected one of: {LABEL_CANDIDATES}. "
        f"Columns seen: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser(description="Train a GradientBoosting model using selected 4 features.")
    ap.add_argument("--train-tsv", required=True, help="TSV containing features + labels.")
    ap.add_argument("--test-tsv", required=True, help="TSV containing features + labels.")
    ap.add_argument("--out-model", required=True, help="Output joblib path (model bundle).")
    ap.add_argument("--out-report", default=None, help="Optional: write JSON report here.")

    ap.add_argument("--id-col", default="read_id", help="Read ID column name (default: read_id).")

    # IMPORTANT: default is None -> auto-detect label/ y / target / class
    ap.add_argument(
        "--label-col",
        default=None,
        help="Label column name. If omitted, auto-detects one of: label, y, target, class.",
    )

    ap.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Override feature list. If omitted, uses default selected4 list.",
    )

    # GB params
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--random-state", type=int, default=42)

    # reporting threshold
    ap.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()

    features = resolve_feature_list(args)

    train_df = load_table(args.train_tsv)
    test_df = load_table(args.test_tsv)

    # Resolve label column (auto-detect unless explicitly provided)
    label_col_train = resolve_label_col(train_df, args.label_col)
    label_col_test = resolve_label_col(test_df, args.label_col)

    # Enforce same name in both (usually true)
    if label_col_train != label_col_test:
        raise ValueError(
            f"Label column mismatch between train and test: "
            f"train='{label_col_train}' vs test='{label_col_test}'."
        )
    label_col = label_col_train

    # Feature checks
    missing_train = [c for c in features if c not in train_df.columns]
    missing_test = [c for c in features if c not in test_df.columns]
    if missing_train:
        raise ValueError(f"Train TSV missing feature columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test TSV missing feature columns: {missing_test}")

    X_train = train_df[features].copy()
    y_train = train_df[label_col].astype(int).to_numpy()

    X_test = test_df[features].copy()
    y_test = test_df[label_col].astype(int).to_numpy()

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    # Evaluate
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= float(args.threshold)).astype(int)
    metrics = compute_metrics(y_test, pred, prob)

    bundle = {
        "model": model,
        "features": features,  # contract saved here
        "id_col": args.id_col,
        "label_col": label_col,
        "threshold_default": float(args.threshold),
        "train_path": str(Path(args.train_tsv)),
        "test_path": str(Path(args.test_tsv)),
        "gb_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "random_state": args.random_state,
        },
        "test_metrics": metrics,
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_model)

    print("[OK] Saved model bundle:", out_model)
    print("[OK] Label col:", label_col)
    print("[OK] Features used:", features)
    print(
        f"[OK] Test metrics: f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} "
        f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} acc={metrics['accuracy']:.4f}"
    )

    if args.out_report:
        out_report = Path(args.out_report)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(bundle["test_metrics"], indent=2))
        print("[OK] Wrote report:", out_report)


if __name__ == "__main__":
    main()