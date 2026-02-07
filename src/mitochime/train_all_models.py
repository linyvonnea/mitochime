#!/usr/bin/env python3
"""
Train the full model zoo on the per-read features.

Workflow:
  1. Load train.tsv and test.tsv (already stratified 80/20).
  2. Prepare X, y using data_utils.prepare_X_y().
  3. For each model in model_zoo:
        - build a sklearn Pipeline:
              [SimpleImputer -> StandardScaler -> classifier]
        - 5-fold Stratified CV on training set (accuracy, F1).
        - fit on full training set.
        - evaluate on holdout test set (accuracy, precision, recall, F1, ROC AUC).
        - save the fitted pipeline to models/<model_name>.joblib
        - record metrics to reports/metrics/metrics_summary.tsv


Run:
python3 -m mitochime.train_all_models \
  --train data/processed/train_noq.tsv \
  --test  data/processed/test_noq.tsv \
  --models-dir  models_noq \
  --reports-dir reports/metrics_noq

  ORRRR FOR PAIR


  PYTHONPATH=src python3 -m mitochime.train_all_models \
  --train data/processed/PAIR_train_noq.tsv \
  --test  data/processed/PAIR_test_noq.tsv \
  --models-dir  models_PAIR \
  --reports-dir reports/metrics_PAIR
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import prepare_X_y
from .model_zoo import get_model_zoo


RANDOM_STATE = 42


def _safe_roc_auc(model, X, y) -> float:
    """
    Compute ROC-AUC using predict_proba or decision_function.
    Returns NaN if neither is available.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim > 1:
            scores = scores[:, 1]
    else:
        return float("nan")

    return roc_auc_score(y, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models on feature table.")
    parser.add_argument(
        "--train",
        required=True,
        help="Path to train.tsv created by split_dataset.py",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to test.tsv created by split_dataset.py",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save fitted model pipelines (default: models/).",
    )
    parser.add_argument(
        "--reports-dir",
        default=os.path.join("reports", "metrics"),
        help="Directory to save metrics summaries (default: reports/metrics/).",
    )

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # --- Load data ---
    train_df = pd.read_csv(args.train, sep="\t")
    test_df = pd.read_csv(args.test, sep="\t")

    X_train, y_train = prepare_X_y(train_df)
    X_test, y_test = prepare_X_y(test_df)

    print(f"Train size: {X_train.shape}, positive={y_train.sum()}")
    print(f"Test  size: {X_test.shape}, positive={y_test.sum()}")

    # --- CV setup ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_metrics: List[Dict[str, Any]] = []

    model_zoo = get_model_zoo(random_state=RANDOM_STATE)

    for name, clf in model_zoo.items():
        print("\n" + "=" * 80)
        print(f"Training model: {name}")
        print("=" * 80)

        # Common preprocessing: impute + scale
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

        # --- 5-fold CV on training set ---
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1"],
            return_train_score=False,
            n_jobs=-1,
        )

        cv_acc_mean = float(np.mean(cv_results["test_accuracy"]))
        cv_acc_std = float(np.std(cv_results["test_accuracy"]))
        cv_f1_mean = float(np.mean(cv_results["test_f1"]))
        cv_f1_std = float(np.std(cv_results["test_f1"]))

        print(
            f"[CV] accuracy = {cv_acc_mean:.4f} ± {cv_acc_std:.4f}, "
            f"F1 = {cv_f1_mean:.4f} ± {cv_f1_std:.4f}"
        )

        # --- Fit on full training set ---
        pipeline.fit(X_train, y_train)

        # --- Evaluate on test set ---
        y_pred = pipeline.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_roc_auc = _safe_roc_auc(pipeline, X_test, y_test)

        print(
            f"[TEST] acc={test_acc:.4f}, prec={test_prec:.4f}, "
            f"rec={test_rec:.4f}, F1={test_f1:.4f}, ROC-AUC={test_roc_auc:.4f}"
        )

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix (rows: true, cols: pred):")
        print(cm)

        # --- Save model ---
        model_path = os.path.join(args.models_dir, f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"Saved trained model to {model_path}")

        # --- Record metrics ---
        metrics_row = {
            "model": name,
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "test_accuracy": float(test_acc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
            "test_roc_auc": float(test_roc_auc),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        all_metrics.append(metrics_row)

        # also save per-model metrics as JSON
        per_model_json = os.path.join(args.reports_dir, f"{name}_metrics.json")
        with open(per_model_json, "w") as f_json:
            json.dump(metrics_row, f_json, indent=2)

    # --- Save summary TSV across all models ---
    summary_df = pd.DataFrame(all_metrics)
    summary_tsv = os.path.join(args.reports_dir, "metrics_summary.tsv")
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    print(f"\nWrote summary metrics table: {summary_tsv}")


if __name__ == "__main__":
    main()