#!/usr/bin/env python3
"""
Define a panel ("zoo") of ML models for chimeric vs clean read classification.

Exposed API:
    get_model_zoo(random_state: int = 42) -> dict[str, sklearn-like estimator]
"""

from __future__ import annotations

from typing import Dict

from sklearn.dummy import DummyClassifier
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_model_zoo(random_state: int = 42) -> Dict[str, object]:
    """
    Return a dict of model_name -> sklearn-compatible estimator.

    All models here are reasonably configured defaults or slightly tuned baselines.
    """

    models: Dict[str, object] = {}

    # -------------------------------------------------------------------------
    # 1) Baseline
    # -------------------------------------------------------------------------
    models["dummy_baseline"] = DummyClassifier(strategy="most_frequent")

    # -------------------------------------------------------------------------
    # 2) Logistic Regression (L2)
    # -------------------------------------------------------------------------
    models["logreg_l2"] = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",  # good for small-ish datasets, binary
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # 3) Linear SVM + probability calibration
    # -------------------------------------------------------------------------
    base_svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
    )
    models["linear_svm_calibrated"] = CalibratedClassifierCV(
        estimator=base_svm,
        cv=5,
        method="sigmoid",
    )

    # -------------------------------------------------------------------------
    # 4) Random Forest
    # -------------------------------------------------------------------------
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # 5) ExtraTrees (Extremely Randomized Trees)
    # -------------------------------------------------------------------------
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # 6) Gradient Boosting (sklearn)
    # -------------------------------------------------------------------------
    models["gradient_boosting"] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.9,
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # 7) XGBoost
    # -------------------------------------------------------------------------
    models["xgboost"] = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
    )

    # -------------------------------------------------------------------------
    # 8) LightGBM
    # -------------------------------------------------------------------------
    models["lightgbm"] = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )

    # -------------------------------------------------------------------------
    # 9) CatBoost
    #   - silent=True to avoid massive terminal spam
    # -------------------------------------------------------------------------
    models["catboost"] = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.1,
        depth=6,
        random_seed=random_state,
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # 10) k-Nearest Neighbors
    # -------------------------------------------------------------------------
    models["knn"] = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="minkowski",
        p=2,
    )

    # -------------------------------------------------------------------------
    # 11) Gaussian Naive Bayes
    # -------------------------------------------------------------------------
    models["gaussian_nb"] = GaussianNB()

    # -------------------------------------------------------------------------
    # 12) Bagging with Decision Trees
    # -------------------------------------------------------------------------
    base_tree = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
    )
    models["bagging_trees"] = BaggingClassifier(
        estimator=base_tree,
        n_estimators=200,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # 13) Shallow MLP
    # -------------------------------------------------------------------------
    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=1000,
        random_state=random_state,
    )

    return models