#!/usr/bin/env python3
"""
Hyperparameter search for ensemble / non-linear models on chimeric-read classification.

Models tuned here
-----------------
- RandomForest
- ExtraTrees
- GradientBoosting
- XGBoost
- LightGBM
- CatBoost
- Bagging (DecisionTree)
- MLP (small feed-forward NN)

Usage
-----
python3 -m mitochime.hyperparam_search_ensembles \
  --train data/processed/train_noq.tsv \
  --test  data/processed/test_noq.tsv \
  --models-dir  models_noq_tuned \
  --reports-dir reports/hparam_tuning_noq_ensembles \
  --n-iter 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Data loader (numeric only)
# ----------------------------------------------------------------------
def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load TSV dataset and return:
        X: numeric feature matrix (np.ndarray, float)
        y: labels (np.ndarray, int)
        feature_names: list of feature column names

    We deliberately:
      - keep ONLY numeric columns
      - drop 'label' from features
      - ignore string / categorical cols like 'read_id', 'ref_name', 'cigar', 'strand'
    """
    df = pd.read_csv(path, sep="\t")

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "label"]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    return X, y, feature_cols


# ----------------------------------------------------------------------
# Evaluation helper
# ----------------------------------------------------------------------
def evaluate_best(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models_dir: Path,
    reports_dir: Path,
) -> dict:
    """
    Fit model on full train, evaluate on held-out test,
    save classification_report and tuned model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # for ROC-AUC: try predict_proba, else decision_function, else fallback
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_scores is not None:
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = 0.5

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["clean", "chimeric"],
        digits=4,
        zero_division=0,
    )
    (reports_dir / f"{name}_tuned_classification_report.txt").write_text(report_text)

    joblib.dump(model, models_dir / f"{name}_tuned.joblib")

    return {
        "model": f"{name}_tuned",
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for ensemble / non-linear models."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="RandomizedSearchCV iterations per model (default: 25).",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, feature_names = load_dataset(args.train)
    X_test, y_test, _ = load_dataset(args.test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # Define models + search spaces
    # ------------------------------------------------------------------
    model_spaces: Dict[str, Tuple[object, Dict[str, list]]] = {}

    # 1) RandomForest
    model_spaces["random_forest"] = (
        RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
    )

    # 2) ExtraTrees
    model_spaces["extra_trees"] = (
        ExtraTreesClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
    )

    # 3) GradientBoosting
    model_spaces["gradient_boosting"] = (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
    )

    # 4) XGBoost
    model_spaces["xgboost"] = (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        ),
        {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "min_child_weight": [1, 5, 10],
        },
    )

    # 5) LightGBM
    model_spaces["lightgbm"] = (
        LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.05, 0.1, 0.2],
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 10, 20],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "min_child_samples": [10, 20, 40],
        },
    )

    # 6) CatBoost
    model_spaces["catboost"] = (
        CatBoostClassifier(
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1, 0.2],
            "iterations": [200, 400, 800],
            "l2_leaf_reg": [1, 3, 5],
        },
    )

    # 7) Bagging (DecisionTree)
    base_tree = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )
    model_spaces["bagging_trees"] = (
        BaggingClassifier(
            estimator=base_tree,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        {
            "n_estimators": [100, 200, 400],
            "max_samples": [0.6, 0.8, 1.0],
            "max_features": [0.6, 0.8, 1.0],
            "bootstrap": [True, False],
        },
    )

    # 8) MLP
    model_spaces["mlp"] = (
        MLPClassifier(
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        {
            "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 5e-3, 1e-2],
        },
    )

    all_results = []

    # ------------------------------------------------------------------
    # Run RandomizedSearchCV per model
    # ------------------------------------------------------------------
    for name, (estimator, param_dist) in model_spaces.items():
        print(f"\n=== Hyperparameter search: {name} ===")

        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_params = search.best_params_
        print(f"Best params for {name}:", best_params)

        # Save best params JSON
        (reports_dir / f"{name}_best_params.json").write_text(
            json.dumps(best_params, indent=2)
        )

        best_model = search.best_estimator_
        res = evaluate_best(
            name,
            best_model,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
        all_results.append(res)

    # ------------------------------------------------------------------
    # Save small summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(all_results)
    summary_path = reports_dir / "tuned_ensembles_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nWrote tuned ensemble models summary to {summary_path}")


if __name__ == "__main__":
    main()