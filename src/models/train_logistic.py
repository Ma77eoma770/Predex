"""Train and evaluate Logistic Regression using the Master Feature Store.

Demonstrates leakage-free preprocessing pipeline with:
- Collinear feature removal
- Standardization
- 10-Fold CV evaluation
- Out-of-Fold (OOF) prediction export
- Final submission generation
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.config import (
    ID_COL,
    OUTPUTS_DIR,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
)
from src.cross_validation import get_folds, load_folds
from src.features import (
    DropCollinearFeatures,
    create_linear_preprocessing_pipeline,
    load_master_features,
)


def run_training():
    print("=" * 65)
    print("Predex - Logistic Regression with Master Feature Store")
    print("=" * 65)

    # 1. Carica istantaneamente il Master Feature Store
    print("\n[1/4] Loading master training features from Parquet...")
    X_train, y_train = load_master_features("train")
    print(f"      Train matrix: {X_train.shape[0]} battles x {X_train.shape[1]} features")

    # 2. Carica le 10 fold ufficiali congelate
    print("\n[2/4] Loading official 10-fold CV splits...")
    folds_df = load_folds()

    oof_preds = np.zeros(len(X_train))
    oof_probs = np.zeros(len(X_train))
    fold_accuracies = []

    print("\n[3/4] Running 10-Fold Stratified Cross-Validation...")
    print("-" * 55)
    print(f"{'Fold':<8} | {'Val Samples':<14} | {'Accuracy':<12}")
    print("-" * 55)

    for fold, train_idx, val_idx in get_folds(folds_df):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Pipeline con filtro multicollinearità + scaler + classificatore
        pipeline = Pipeline([
            ("preprocessor", create_linear_preprocessing_pipeline(
                drop_collinear=True,
                collinear_threshold=0.95,
                k_best=None,  # Modifica se vuoi usare solo le migliori K feature
            )),
            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ])

        pipeline.fit(X_tr, y_tr)
        val_preds = pipeline.predict(X_va)
        val_probs = pipeline.predict_proba(X_va)[:, 1]

        acc = accuracy_score(y_va, val_preds)
        fold_accuracies.append(acc)
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs

        print(f"Fold {fold:<3} | {len(val_idx):<14} | {acc:.4%}")

    print("-" * 55)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    total_oof_acc = accuracy_score(y_train, oof_preds)
    print(f"Mean CV Accuracy:    {mean_acc:.4%} +/- {std_acc:.4%}")
    print(f"Overall OOF Accuracy:{total_oof_acc:.4%}")
    print("-" * 55)

    # Salva le predizioni OOF per lo stacking finale del team
    oof_df = pd.DataFrame({
        ID_COL: X_train.index,
        TARGET_COL: y_train.values,
        "pred_prob": oof_probs,
        "pred_class": oof_preds.astype(int),
    })
    oof_path = OUTPUTS_DIR / "oof_logistic_master.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\n[SAVED] Out-of-fold predictions -> {oof_path}")

    # 4. Retrain su full train e generazione submission test
    print("\n[4/4] Retraining on all 10,000 battles and inferring test set...")
    full_pipeline = Pipeline([
        ("preprocessor", create_linear_preprocessing_pipeline(
            drop_collinear=True,
            collinear_threshold=0.95,
            k_best=None,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    full_pipeline.fit(X_train, y_train)

    X_test, _ = load_master_features("test")
    test_preds = full_pipeline.predict(X_test)

    submission_df = pd.DataFrame({
        ID_COL: X_test.index,
        TARGET_COL: test_preds.astype(int),
    })

    if SAMPLE_SUBMISSION_CSV.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        assert len(submission_df) == len(sample_sub), "Submission length mismatch!"
        assert (submission_df[ID_COL].values == sample_sub[ID_COL].values).all(), "ID alignment mismatch!"

    sub_path = OUTPUTS_DIR / "submission_logistic_master.csv"
    submission_df.to_csv(sub_path, index=False)
    print(f"[SUCCESS] Submission generated -> {sub_path}")


if __name__ == "__main__":
    run_training()
