"""Train and evaluate Baseline Logistic Regression model with 10-Fold CV."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ID_COL,
    OUTPUTS_DIR,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_JSONL,
    TRAIN_JSONL,
)
from src.cross_validation import get_folds, load_folds
from src.features import build_dataset


def run_logistic_baseline():
    print("[1/4] Parsing and extracting features from raw training data...")
    X_train, y_train = build_dataset(TRAIN_JSONL, is_train=True)
    print(f"      Train matrix shape: {X_train.shape} (Battles x Features)")

    print("\n[2/4] Loading official precomputed cross-validation folds...")
    folds_df = load_folds()

    # Mappiamo gli indici dei fold sull'indice di X_train
    oof_predictions = np.zeros(len(X_train))
    fold_accuracies = []

    print("\n[3/4] Running 10-Fold Stratified Cross-Validation...")
    print("-" * 55)
    print(f"{'Fold':<8} | {'Val Samples':<14} | {'Accuracy':<12}")
    print("-" * 55)

    for fold, train_idx, val_idx in get_folds(folds_df):
        # Selezione dei campioni per fold
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Costruzione della pipeline lineare standardizzata
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            ))
        ])

        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_va)

        acc = accuracy_score(y_va, val_preds)
        fold_accuracies.append(acc)
        oof_predictions[val_idx] = val_preds

        print(f"Fold {fold:<3} | {len(val_idx):<14} | {acc:.4%}")

    print("-" * 55)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    total_oof_acc = accuracy_score(y_train, oof_predictions)
    print(f"Mean CV Accuracy: {mean_acc:.4%} +/- {std_acc:.4%}")
    print(f"Overall OOF Accuracy: {total_oof_acc:.4%}")
    print("-" * 55)

    # 4. Retrain su tutti i 10.000 campioni e inferenza sul test set
    print("\n[4/4] Retraining on full 10,000 battles and generating submission...")
    full_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])
    full_pipeline.fit(X_train, y_train)

    X_test, _ = build_dataset(TEST_JSONL, is_train=False)
    test_preds = full_pipeline.predict(X_test)

    # Costruzione DataFrame conforme al formato sample_submission.csv
    submission_df = pd.DataFrame({
        ID_COL: X_test.index,
        TARGET_COL: test_preds.astype(int)
    })

    # Verifica formale rispetto a sample_submission.csv
    if SAMPLE_SUBMISSION_CSV.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        assert len(submission_df) == len(sample_sub), "Error: Submission rows do not match sample_submission!"
        assert (submission_df[ID_COL].values == sample_sub[ID_COL].values).all(), "Error: Battle IDs out of order!"

    out_csv = OUTPUTS_DIR / "submission_logistic_baseline.csv"
    submission_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Submission file generated at: {out_csv.resolve()}")


if __name__ == "__main__":
    run_logistic_baseline()