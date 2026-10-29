"""Train and evaluate Baseline Logistic Regression using the new Parquet Feature Store."""

import sys
from pathlib import Path

# Inclusione root del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import (
    ID_COL,
    OUTPUTS_DIR,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
)
from src.cross_validation import get_folds, load_folds
from src.features import (
    create_linear_preprocessing_pipeline,
    load_master_features,
)


def run_logistic_pipeline():
    print("[1/4] Loading features from Master Parquet cache...")
    # Carica istantaneamente X e y dai file .parquet generati
    X_train, y_train = load_master_features(split="train")
    print(f"      Train matrix loaded: {X_train.shape[0]} battles x {X_train.shape[1]} features")

    print("\n[2/4] Loading official precomputed cross-validation folds...")
    folds_df = load_folds()

    oof_predictions = np.zeros(len(X_train))
    fold_accuracies = []

    print("\n[3/4] Running 10-Fold Stratified Cross-Validation...")
    print("-" * 65)
    print(f"{'Fold':<8} | {'Val Samples':<14} | {'Accuracy':<12}")
    print("-" * 65)

    for fold, train_idx, val_idx in get_folds(folds_df):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Creazione della pipeline completa: Variance -> Collinear -> Scaler
        preprocessing_pipe = create_linear_preprocessing_pipeline(
            drop_collinear=True,
            collinear_threshold=0.95
        )

        # Modello con regolarizzazione L2 controllata da C (senza warning)
        classifier = LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        )

        # Assembliamo la pipeline end-to-end
        model = preprocessing_pipe
        model.steps.append(("classifier", classifier))

        # Fit rigoroso solo sul training fold corrente (anti-leakage)
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_va)

        acc = accuracy_score(y_va, val_preds)
        fold_accuracies.append(acc)
        oof_predictions[val_idx] = val_preds

        print(f"Fold {fold:<3} | {len(val_idx):<14} | {acc:.4%}")

    print("-" * 65)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    total_oof_acc = accuracy_score(y_train, oof_predictions)
    print(f"Mean CV Accuracy: {mean_acc:.4%} +/- {std_acc:.4%}")
    print(f"Overall OOF Accuracy: {total_oof_acc:.4%}")
    print("-" * 65)

    # 4. Retrain sull'intero training set ed inferenza sul test set
    print("\n[4/4] Retraining on full 10,000 battles and generating test predictions...")
    full_pipeline = create_linear_preprocessing_pipeline(
        drop_collinear=True,
        collinear_threshold=0.95
    )
    full_pipeline.steps.append((
        "classifier",
        LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        )
    ))
    full_pipeline.fit(X_train, y_train)

    # Carica il test set dai parquet
    X_test, _ = load_master_features(split="test")
    test_preds = full_pipeline.predict(X_test)

    submission_df = pd.DataFrame({
        ID_COL: X_test.index,
        TARGET_COL: test_preds.astype(int)
    })

    if SAMPLE_SUBMISSION_CSV.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        assert len(submission_df) == len(sample_sub), "Error: Row count mismatch!"
        assert (submission_df[ID_COL].values == sample_sub[ID_COL].values).all(), "Error: IDs mismatch!"

    out_csv = OUTPUTS_DIR / "submission_logistic_modular.csv"
    submission_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Submission saved at: {out_csv.resolve()}")


if __name__ == "__main__":
    run_logistic_pipeline()