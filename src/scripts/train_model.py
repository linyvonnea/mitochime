#!/usr/bin/env python3
"""
train_model.py

Load dataset.tsv (features + labels), split into train/val/test,
train a baseline model (RandomForest), and report performance.

Pipeline:
  1) Read dataset.tsv
  2) Split into X (features), y (labels)
  3) Stratified train/val/test split (e.g., 70/15/15)
  4) Train RandomForest on train
  5) Evaluate on val and test

You can later:
  - add more models (SVM, XGBoost, etc.)
  - add feature scaling if needed
  - add cross-validation
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "dataset.tsv"


def main():
    print(f"[INFO] Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, sep="\t")

    # Separate features and labels
    X = df.drop(columns=["label", "read_id"])
    y = df["label"]

    # First: split off test set (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    # Then: split train vs validation from remaining (so val ≈ 15% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15
        random_state=42,
        stratify=y_trainval,
    )

    print("[INFO] Dataset shapes:")
    print("  Train:", X_train.shape)
    print("  Val:  ", X_val.shape)
    print("  Test: ", X_test.shape)

    # Baseline model: RandomForest
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    print("\n[INFO] Training RandomForest...")
    clf.fit(X_train, y_train)

    print("\n[VAL] Performance on validation set:")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=3))

    print("\n[TEST] Performance on held-out test set:")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=3))


if __name__ == "__main__":
    main()