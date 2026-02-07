#!/usr/bin/env python3
"""
Hyperparameter search for top models on chimeric vs clean classification.

We tune 10 model families:

  - logreg_l2
  - linear_svm_calibrated
  - random_forest
  - extra_trees
  - gradient_boosting
  - xgboost
  - lightgbm
  - catboost
  - bagging_trees
  - mlp

Usage
-----

# PYTHONPATH=src python3 -m mitochime.hyperparam_search_top \
  --train data/processed/PAIR_train_noq.tsv \
  --test  data/processed/PAIR_test_noq.tsv \
  --models-dir models_PAIR_noq_tuned \
  --reports-dir reports/hparam_tuning_PAIR_noq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

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


# ============================================================
# DATA LOADER
# ============================================================
def load_dataset(path: str):
    """
    Load TSV and return X (float, no NaNs), y (int), feature_names.

    - Drops non-feature text columns
    - Encodes strand (+/-) -> 1/0 if present
    - Forces numeric conversion
    - Median-imputes remaining NaNs
    """
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(path, sep="\t")
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")

    # Encode strand if present
    if "strand" in df.columns:
        # your file uses '+' and '-'
        df["strand"] = df["strand"].map({"+": 1, "-": 0})

    # Drop known non-numeric/non-feature columns
    drop_cols = ["read_id", "ref_name", "cigar"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Force numeric for all features (anything weird -> NaN)
    for c in df.columns:
        if c != "label":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["label"] = df["label"].astype(int)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    # Impute NaNs safely
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    return X, y, feature_cols


# ============================================================
# EVALUATION / SAVING
# ============================================================

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
    Fit best model on full train, evaluate on test, save model + report,
    and return a metrics dict for summary table.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for top-performing models."
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

    all_results = []

    # --------------------------------------------------------
    # 1) Logistic Regression (L2)
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: logreg_l2 ===")
    logreg = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    logreg_param_dist = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    logreg_search = RandomizedSearchCV(
        logreg,
        logreg_param_dist,
        n_iter=min(args.n_iter, len(logreg_param_dist["C"])),
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    logreg_search.fit(X_train, y_train)
    print("Best logreg_l2 params:", logreg_search.best_params_)
    (reports_dir / "logreg_l2_best_params.json").write_text(
        json.dumps(logreg_search.best_params_, indent=2)
    )
    best_logreg = logreg_search.best_estimator_
    all_results.append(
        evaluate_best(
            "logreg_l2",
            best_logreg,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 2) Linear SVM + calibration
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: linear_svm_calibrated ===")
    base_svm = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    svm_cal = CalibratedClassifierCV(
        estimator=base_svm,
        cv=5,
        method="sigmoid",
    )
    svm_param_dist = {
        "estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    svm_search = RandomizedSearchCV(
        svm_cal,
        svm_param_dist,
        n_iter=min(args.n_iter, len(svm_param_dist["estimator__C"])),
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    svm_search.fit(X_train, y_train)
    print("Best linear_svm_calibrated params:", svm_search.best_params_)
    (reports_dir / "linear_svm_calibrated_best_params.json").write_text(
        json.dumps(svm_search.best_params_, indent=2)
    )
    best_svm = svm_search.best_estimator_
    all_results.append(
        evaluate_best(
            "linear_svm_calibrated",
            best_svm,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 3) RandomForest
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: RandomForest ===")
    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf_param_dist = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)
    print("Best RF params:", rf_search.best_params_)
    (reports_dir / "random_forest_best_params.json").write_text(
        json.dumps(rf_search.best_params_, indent=2)
    )
    best_rf = rf_search.best_estimator_
    all_results.append(
        evaluate_best(
            "random_forest",
            best_rf,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 4) ExtraTrees
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: ExtraTrees ===")
    et = ExtraTreesClassifier(
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    et_param_dist = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    et_search = RandomizedSearchCV(
        et,
        et_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    et_search.fit(X_train, y_train)
    print("Best ExtraTrees params:", et_search.best_params_)
    (reports_dir / "extra_trees_best_params.json").write_text(
        json.dumps(et_search.best_params_, indent=2)
    )
    best_et = et_search.best_estimator_
    all_results.append(
        evaluate_best(
            "extra_trees",
            best_et,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 5) GradientBoosting
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: GradientBoosting ===")
    gb = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )
    gb_param_dist = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.03, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.9, 1.0],
        "min_samples_leaf": [1, 2, 4],
    }
    gb_search = RandomizedSearchCV(
        gb,
        gb_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    gb_search.fit(X_train, y_train)
    print("Best GradientBoosting params:", gb_search.best_params_)
    (reports_dir / "gradient_boosting_best_params.json").write_text(
        json.dumps(gb_search.best_params_, indent=2)
    )
    best_gb = gb_search.best_estimator_
    all_results.append(
        evaluate_best(
            "gradient_boosting",
            best_gb,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 6) XGBoost
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: XGBoost ===")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=6,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    xgb_param_dist = {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 5, 10],
    }
    xgb_search = RandomizedSearchCV(
        xgb,
        xgb_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    print("Best XGB params:", xgb_search.best_params_)
    (reports_dir / "xgboost_best_params.json").write_text(
        json.dumps(xgb_search.best_params_, indent=2)
    )
    best_xgb = xgb_search.best_estimator_
    all_results.append(
        evaluate_best(
            "xgboost",
            best_xgb,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 7) LightGBM
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: LightGBM ===")
    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm_param_dist = {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10, 20],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_samples": [10, 20, 40],
    }
    lgbm_search = RandomizedSearchCV(
        lgbm,
        lgbm_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    lgbm_search.fit(X_train, y_train)
    print("Best LGBM params:", lgbm_search.best_params_)
    (reports_dir / "lightgbm_best_params.json").write_text(
        json.dumps(lgbm_search.best_params_, indent=2)
    )
    best_lgbm = lgbm_search.best_estimator_
    all_results.append(
        evaluate_best(
            "lightgbm",
            best_lgbm,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 8) CatBoost
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: CatBoost ===")
    cat = CatBoostClassifier(
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE,
    )
    cat_param_dist = {
        "iterations": [200, 400, 800],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7],
    }
    cat_search = RandomizedSearchCV(
        cat,
        cat_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    cat_search.fit(X_train, y_train)
    print("Best CatBoost params:", cat_search.best_params_)
    (reports_dir / "catboost_best_params.json").write_text(
        json.dumps(cat_search.best_params_, indent=2)
    )
    best_cat = cat_search.best_estimator_
    all_results.append(
        evaluate_best(
            "catboost",
            best_cat,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 9) Bagging (decision trees)
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: Bagging (trees) ===")
    base_tree = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    bag = BaggingClassifier(
        estimator=base_tree,
        n_estimators=200,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    bag_param_dist = {
        "n_estimators": [100, 200, 400],
        "max_samples": [0.5, 0.7, 0.9],
        "max_features": [0.5, 0.8, 1.0],
    }
    bag_search = RandomizedSearchCV(
        bag,
        bag_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    bag_search.fit(X_train, y_train)
    print("Best Bagging params:", bag_search.best_params_)
    (reports_dir / "bagging_trees_best_params.json").write_text(
        json.dumps(bag_search.best_params_, indent=2)
    )
    best_bag = bag_search.best_estimator_
    all_results.append(
        evaluate_best(
            "bagging_trees",
            best_bag,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # 10) Shallow MLP
    # --------------------------------------------------------
    print("\n=== Hyperparameter search: MLP ===")
    mlp = MLPClassifier(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    mlp_param_dist = {
        "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-3, 5e-3, 1e-2],
    }
    mlp_search = RandomizedSearchCV(
        mlp,
        mlp_param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    mlp_search.fit(X_train, y_train)
    print("Best MLP params:", mlp_search.best_params_)
    (reports_dir / "mlp_best_params.json").write_text(
        json.dumps(mlp_search.best_params_, indent=2)
    )
    best_mlp = mlp_search.best_estimator_
    all_results.append(
        evaluate_best(
            "mlp",
            best_mlp,
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir,
            reports_dir,
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary_df = pd.DataFrame(all_results)
    summary_path = reports_dir / "tuned_models_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nWrote tuned models summary to {summary_path}")


if __name__ == "__main__":
    main()