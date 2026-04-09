"""
train.py
--------
Trains four models on fake-news datasets:
  1. Logistic Regression         (lr)
  2. Decision Tree               (dt)
  3. Random Forest               (rf)
  4. Hist Gradient Boosting      (gb)  ← uses smaller vectorizer to fit in RAM

Supported datasets (--dataset flag):
  welfake   — WELFake_Dataset.csv only            (~72K articles)
  isot      — ISOT True.csv + Fake.csv only       (~44K articles)
  combined  — WELFake + ISOT merged & shuffled    (~117K articles)

Usage:
  # WELFake only (default)
  python train.py

  # ISOT only
  python train.py --dataset isot --isot-true True.csv --isot-fake Fake.csv

  # Combined (both datasets)
  python train.py --dataset combined \
      --csv WELFake_Dataset.csv \
      --isot-true True.csv \
      --isot-fake Fake.csv

Saves to model/:
  vectorizer.joblib           ← shared TF-IDF for lr / dt / rf
  vectorizer_gb.joblib        ← smaller TF-IDF for gb only (5K features)
  classifier_lr.joblib
  classifier_dt.joblib
  classifier_rf.joblib
  classifier_gb.joblib
  confusion_matrix_<key>.png
  roc_curve_<key>.png
  metrics.csv
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from preprocess import (
    load_and_preprocess,
    load_and_preprocess_isot,
    load_and_preprocess_combined,
)


# -- Config -------------------------------------------------------------------

MODEL_DIR   = "model"
RANDOM_SEED = 42

# Main vectorizer — used by lr, dt, rf
TFIDF_PARAMS = dict(
    max_features=100_000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

# Smaller vectorizer for GB only — 5K features fits in RAM as dense matrix
TFIDF_PARAMS_GB = dict(
    max_features=5_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

CLASSIFIERS = {
    "lr": LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        random_state=RANDOM_SEED, class_weight="balanced"
    ),
    "dt": DecisionTreeClassifier(
        max_depth=30, min_samples_split=5,
        random_state=RANDOM_SEED, class_weight="balanced"
    ),
    "rf": RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=5,
        random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
    ),
    "gb": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_SEED, class_weight="balanced"
    ),
}

DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

# Dataset display info (for console output)
DATASET_LABELS = {
    "welfake":  "WELFake",
    "isot":     "ISOT",
    "combined": "WELFake + ISOT (Combined)",
}


# -- Helpers ------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, name, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake (1)", "Real (0)"],
        yticklabels=["Fake (1)", "Real (0)"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [+] Confusion matrix -> {path}")


def save_roc_curve(y_true, y_prob, name, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve -- {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [+] ROC curve        -> {path}")


# -- Main ---------------------------------------------------------------------

def train(
    dataset: str      = "welfake",
    csv_path: str     = "WELFake_Dataset.csv",
    isot_true: str    = "True.csv",
    isot_fake: str    = "Fake.csv",
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load & preprocess (dataset-aware)
    print("=" * 55)
    print(f"  Dataset: {DATASET_LABELS[dataset]}")
    print("=" * 55)

    if dataset == "welfake":
        df = load_and_preprocess(csv_path)
    elif dataset == "isot":
        df = load_and_preprocess_isot(isot_true, isot_fake)
    else:  # combined
        df = load_and_preprocess_combined(csv_path, isot_true, isot_fake)

    X, y = df["combined"].values, df["label"].values

    # 2. Shared train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # 3a. Main TF-IDF vectorizer (lr / dt / rf)
    print("[INFO] Fitting main TF-IDF vectorizer (100K features) ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_):,}")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print(f"[+] Main vectorizer saved -> {vec_path}\n")

    # 3b. Smaller TF-IDF vectorizer for GB (5K features — fits in RAM as dense)
    print("[INFO] Fitting GB TF-IDF vectorizer (5K features) ...")
    vectorizer_gb = TfidfVectorizer(**TFIDF_PARAMS_GB)
    X_train_gb = vectorizer_gb.fit_transform(X_train).toarray()
    X_test_gb  = vectorizer_gb.transform(X_test).toarray()
    print(f"[INFO] GB vocabulary size: {len(vectorizer_gb.vocabulary_):,}")
    vec_gb_path = os.path.join(MODEL_DIR, "vectorizer_gb.joblib")
    joblib.dump(vectorizer_gb, vec_gb_path)
    print(f"[+] GB vectorizer saved   -> {vec_gb_path}\n")

    # 4. Train each classifier
    all_metrics = []

    for key, clf in CLASSIFIERS.items():
        display = DISPLAY_NAMES[key]
        print("=" * 55)
        print(f"  Training: {display}")
        print("=" * 55)

        # GB uses its own smaller dense vectorizer
        if key == "gb":
            X_tr = X_train_gb
            X_te = X_test_gb
        else:
            X_tr = X_train_vec
            X_te = X_test_vec

        clf.fit(X_tr, y_train)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

        # 5-fold CV only for fast models
        if key in ("lr", "dt"):
            cv = cross_val_score(clf, X_tr, y_train, cv=5, scoring="accuracy")
            print(f"  5-Fold CV: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")

        # Save classifier
        clf_path = os.path.join(MODEL_DIR, f"classifier_{key}.joblib")
        joblib.dump(clf, clf_path)
        print(f"  [+] Classifier saved -> {clf_path}")

        # Plots
        save_confusion_matrix(
            y_test, y_pred, display,
            os.path.join(MODEL_DIR, f"confusion_matrix_{key}.png")
        )
        save_roc_curve(
            y_test, y_prob, display,
            os.path.join(MODEL_DIR, f"roc_curve_{key}.png")
        )

        # Collect metrics
        report = classification_report(
            y_test, y_pred, target_names=["Real", "Fake"], output_dict=True
        )
        all_metrics.append({
            "model":    display,
            "accuracy": round(acc * 100, 2),
            "roc_auc":  round(auc, 4),
            "fake_f1":  round(report["Fake"]["f1-score"], 4),
            "real_f1":  round(report["Real"]["f1-score"], 4),
        })

        print()

    # 5. Save combined metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[+] Combined metrics saved -> {metrics_path}")
    print("\n" + metrics_df.to_string(index=False))
    print(f"\n[+] Training complete on '{DATASET_LABELS[dataset]}' — all 4 models ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake-news detection models")

    parser.add_argument(
        "--dataset",
        choices=["welfake", "isot", "combined"],
        default="welfake",
        help=(
            "Dataset to train on: "
            "'welfake' (WELFake_Dataset.csv only), "
            "'isot' (ISOT True.csv + Fake.csv only), "
            "'combined' (both merged). "
            "Default: welfake"
        ),
    )
    parser.add_argument(
        "--csv",
        default="WELFake_Dataset.csv",
        help="Path to WELFake CSV (used when --dataset is 'welfake' or 'combined')",
    )
    parser.add_argument(
        "--isot-true",
        default="True.csv",
        dest="isot_true",
        help="Path to ISOT True.csv (used when --dataset is 'isot' or 'combined')",
    )
    parser.add_argument(
        "--isot-fake",
        default="Fake.csv",
        dest="isot_fake",
        help="Path to ISOT Fake.csv (used when --dataset is 'isot' or 'combined')",
    )

    args = parser.parse_args()
    train(
        dataset=args.dataset,
        csv_path=args.csv,
        isot_true=args.isot_true,
        isot_fake=args.isot_fake,
    )