"""Train and evaluate Random Forest using the Master Feature Store.

Evaluates 10-Fold Stratified Cross-Validation strictly without leakage:
- Tree-specific preprocessing (variance filter, no scaling needed)
- 10-Fold CV evaluation with accuracy, ROC-AUC, and Brier score
- OOB (Out-of-Bag) score validation
- Out-of-Fold (OOF) probability and prediction export for ensemble stacking
- Top Feature Importance analysis (MDI)
- Final submission generation for the test set
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
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
    create_rf_preprocessing_pipeline,
    load_master_features,
)


def get_rf_classifier(
    n_estimators: int = 500,
    criterion: str = "log_loss",
    max_depth: int = 12,
    min_samples_split: int = 10,
    min_samples_leaf: int = 6,
    max_features: Union[str, float] = "sqrt",
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Instantiate a regularized Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def run_training(
    n_estimators: int = 500,
    criterion: str = "log_loss",
    max_depth: int = 12,
    min_samples_split: int = 10,
    min_samples_leaf: int = 6,
    max_features: Union[str, float] = "sqrt",
    top_k_features: Optional[int] = 300,
    force_recompute_features: bool = False,
):
    print("=" * 70)
    print("Predex - Random Forest Training & Evaluation (Tuned Model)")
    print("=" * 70)
    print(f"Hyperparameters: n_estimators={n_estimators}, criterion={criterion}, "
          f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
          f"min_samples_leaf={min_samples_leaf}, max_features={max_features}")
    print(f"Noise Pruning:   top_k_features={top_k_features} "
          f"({'active' if top_k_features else 'disabled - using all features'})")

    # 1. Carica il Master Feature Store
    print("\n[1/4] Loading master training features from Parquet...")
    X_train, y_train = load_master_features("train", force_recompute=force_recompute_features)
    print(f"      Train matrix: {X_train.shape[0]} battles x {X_train.shape[1]} features")

    # 2. Carica le 10 fold ufficiali congelate
    print("\n[2/4] Loading official 10-fold CV splits...")
    folds_df = load_folds()

    oof_preds = np.zeros(len(X_train))
    oof_probs = np.zeros(len(X_train))
    fold_accuracies = []
    fold_aucs = []
    oob_scores = []

    feature_importances_dict: Dict[str, float] = {col: 0.0 for col in X_train.columns}

    print("\n[3/4] Running 10-Fold Stratified Cross-Validation...")
    print("-" * 65)
    print(f"{'Fold':<6} | {'Val Samples':<12} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'OOB Score':<10}")
    print("-" * 65)

    for fold, train_idx, val_idx in get_folds(folds_df, ids=X_train.index):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        rf_clf = get_rf_classifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        preprocessor = create_rf_preprocessing_pipeline(
            top_k_features=top_k_features,
            random_state=RANDOM_STATE + fold,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", rf_clf),
        ])

        pipeline.fit(X_tr, y_tr)

        val_preds = pipeline.predict(X_va)
        val_probs = pipeline.predict_proba(X_va)[:, 1]

        acc = accuracy_score(y_va, val_preds)
        auc = roc_auc_score(y_va, val_probs)
        oob = rf_clf.oob_score_

        fold_accuracies.append(acc)
        fold_aucs.append(auc)
        oob_scores.append(oob)

        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs

        # Accumulate feature importances mapping back to column names
        pruner = pipeline.named_steps["preprocessor"].named_steps.get("importance_pruner")
        if pruner and hasattr(pruner, "selected_columns_"):
            cols = pruner.selected_columns_
        else:
            cols = list(X_train.columns)

        for c, imp in zip(cols, rf_clf.feature_importances_):
            feature_importances_dict[c] += float(imp) / 10.0

        print(f"Fold {fold:<2} | {len(val_idx):<12} | {acc:<10.4%} | {auc:<10.4f} | {oob:<10.4%}")

    print("-" * 65)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_auc = np.mean(fold_aucs)
    mean_oob = np.mean(oob_scores)
    total_oof_acc = accuracy_score(y_train, oof_preds)
    total_oof_auc = roc_auc_score(y_train, oof_probs)
    brier = brier_score_loss(y_train, oof_probs)

    print(f"Mean CV Accuracy:    {mean_acc:.4%} +/- {std_acc:.4%}")
    print(f"Mean OOB Score:      {mean_oob:.4%}")
    print(f"Mean ROC-AUC:        {mean_auc:.4f}")
    print(f"Overall OOF Accuracy:{total_oof_acc:.4%}")
    print(f"Overall OOF ROC-AUC: {total_oof_auc:.4f}")
    print(f"Brier Score (Calib): {brier:.4f}")
    print("-" * 65)

    # Top 20 Feature Importances
    fi_df = pd.DataFrame({
        "feature": list(feature_importances_dict.keys()),
        "importance": list(feature_importances_dict.values()),
    }).sort_values(by="importance", ascending=False)

    print("\n[TOP 20 MOST IMPORTANT FEATURES (Random Forest MDI)]:")
    for rank, row in enumerate(fi_df.head(20).itertuples(), start=1):
        print(f"  {rank:>2}. {row.feature:<35} : {row.importance:.4f}")

    # Salva le predizioni OOF per lo stacking finale del team
    oof_df = pd.DataFrame({
        ID_COL: X_train.index,
        TARGET_COL: y_train.values,
        "pred_prob": oof_probs,
        "pred_class": oof_preds.astype(int),
    })
    oof_path = OUTPUTS_DIR / "oof_random_forest.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\n[SAVED] Out-of-fold predictions -> {oof_path}")

    # 4. Retrain su full train e generazione submission test
    print("\n[4/4] Retraining on all 10,000 battles and inferring test set...")
    full_rf = get_rf_classifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    full_pipeline = Pipeline([
        ("preprocessor", create_rf_preprocessing_pipeline(
            top_k_features=top_k_features,
            random_state=RANDOM_STATE,
        )),
        ("classifier", full_rf),
    ])
    full_pipeline.fit(X_train, y_train)

    X_test, _ = load_master_features("test")
    test_preds = full_pipeline.predict(X_test)
    test_probs = full_pipeline.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame({
        ID_COL: X_test.index,
        TARGET_COL: test_preds.astype(int),
    })

    if SAMPLE_SUBMISSION_CSV.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        assert len(submission_df) == len(sample_sub), "Submission length mismatch!"
        assert (submission_df[ID_COL].values == sample_sub[ID_COL].values).all(), "ID alignment mismatch!"

    sub_path = OUTPUTS_DIR / "submission_random_forest.csv"
    submission_df.to_csv(sub_path, index=False)
    print(f"[SUCCESS] Submission generated -> {sub_path}")

    # Salva anche le probabilità di test per lo Stacking/Blending
    probs_path = OUTPUTS_DIR / "test_probs_random_forest.csv"
    pd.DataFrame({
        ID_COL: X_test.index,
        "prob_p1_win": test_probs,
    }).to_csv(probs_path, index=False)
    print(f"[SUCCESS] Test probabilities saved -> {probs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predex - Random Forest Trainer")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of trees (default: 500)")
    parser.add_argument("--criterion", type=str, default="log_loss", choices=["gini", "entropy", "log_loss"], help="Split criterion (default: log_loss)")
    parser.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth (default: 12)")
    parser.add_argument("--min-samples-split", type=int, default=10, help="Min samples to split internal node (default: 10)")
    parser.add_argument("--min-samples-leaf", type=int, default=6, help="Min samples at leaf node (default: 6)")
    parser.add_argument("--max-features", type=str, default="sqrt", help="Max features fraction or mode (default: sqrt)")
    parser.add_argument("--top-k", type=int, default=300, help="Number of top features to keep after noise pruning (0 to disable, default: 300)")
    parser.add_argument("--force-features", action="store_true", help="Force rebuild of master feature parquet")
    args = parser.parse_args()

    top_k = args.top_k if args.top_k > 0 else None
    try:
        max_feats = float(args.max_features)
    except ValueError:
        max_feats = args.max_features

    run_training(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_feats,
        top_k_features=top_k,
        force_recompute_features=args.force_features,
    )


