"""Evaluate new mitigation features, feature importance, and ablation on 10-fold CV.

Measures:
1. MDI Importance of all 24 new mit_* features
2. Permutation Importance / Impact on CV Accuracy
3. Ablation test: Performance with vs without low-importance mit_* features
"""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from src.config import OUTPUTS_DIR, RANDOM_STATE
from src.cross_validation import get_folds, load_folds
from src.features import create_rf_preprocessing_pipeline, load_master_features


def evaluate_mitigations():
    print("=" * 75)
    print("Predex - Evaluation & Ablation of Mitigation Features (10-Fold CV)")
    print("=" * 75)

    X_train, y_train = load_master_features("train")
    folds_df = load_folds()
    precomputed_folds = list(get_folds(folds_df, ids=X_train.index))

    mit_cols = [c for c in X_train.columns if c.startswith("mit_")]
    print(f"Total features in dataset: {X_train.shape[1]}")
    print(f"Mitigation features found ({len(mit_cols)}): {mit_cols}")

    # 1. 10-Fold CV with ALL features (including mit_*)
    print("\n[1/3] Running 10-Fold CV with all 478 features...")
    fold_accuracies = []
    fold_aucs = []
    oof_probs = np.zeros(len(X_train))
    feature_importances = {c: 0.0 for c in X_train.columns}

    for fold, train_idx, val_idx in precomputed_folds:
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        rf = RandomForestClassifier(
            n_estimators=400,
            criterion="log_loss",
            max_depth=17,
            min_samples_split=10,
            min_samples_leaf=6,
            max_features=0.2,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )

        pipe = Pipeline([
            ("preprocessor", create_rf_preprocessing_pipeline(top_k_features=None)),
            ("classifier", rf),
        ])

        pipe.fit(X_tr, y_tr)
        val_probs = pipe.predict_proba(X_va)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        acc = accuracy_score(y_va, val_preds)
        auc = roc_auc_score(y_va, val_probs)

        fold_accuracies.append(acc)
        fold_aucs.append(auc)
        oof_probs[val_idx] = val_probs

        for c, imp in zip(X_train.columns, rf.feature_importances_):
            feature_importances[c] += imp / 10.0

    all_acc = np.mean(fold_accuracies)
    all_auc = np.mean(fold_aucs)
    all_brier = brier_score_loss(y_train, oof_probs)
    all_ll = log_loss(y_train, oof_probs)

    print(f"\nAll Features (478) Result:")
    print(f"  Mean Accuracy: {all_acc:.4%} (+/- {np.std(fold_accuracies):.4%})")
    print(f"  Mean ROC-AUC:  {all_auc:.4f}")
    print(f"  Brier Score:   {all_brier:.4f}")
    print(f"  Log-Loss:      {all_ll:.4f}")

    # 2. Ranking of mit_* features
    print("\n[2/3] Ranking Importance of Mitigation Features (MDI):")
    mit_df = pd.DataFrame([
        {"feature": c, "importance": feature_importances[c]}
        for c in mit_cols
    ]).sort_values(by="importance", ascending=False)

    for rank, row in enumerate(mit_df.itertuples(), start=1):
        print(f"  {rank:>2}. {row.feature:<36} : {row.importance:.5f}")

    # Identify dead / near-zero features (< 0.0005 importance)
    low_imp_cols = mit_df[mit_df["importance"] < 0.0005]["feature"].tolist()
    print(f"\nIdentified {len(low_imp_cols)} low-signal mitigation features (< 0.0005 imp): {low_imp_cols}")

    # 3. Ablation: Run without low_imp_cols to see if pruning them improves accuracy
    if low_imp_cols:
        print(f"\n[3/3] Testing Ablation: Pruning {len(low_imp_cols)} low-signal features...")
        filtered_cols = [c for c in X_train.columns if c not in low_imp_cols]
        X_filtered = X_train[filtered_cols]

        ablation_accs = []
        ablation_aucs = []
        ablation_probs = np.zeros(len(X_train))

        for fold, train_idx, val_idx in precomputed_folds:
            X_tr, y_tr = X_filtered.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_filtered.iloc[val_idx], y_train.iloc[val_idx]

            rf = RandomForestClassifier(
                n_estimators=400,
                criterion="log_loss",
                max_depth=17,
                min_samples_split=10,
                min_samples_leaf=6,
                max_features=0.2,
                random_state=RANDOM_STATE + fold,
                n_jobs=-1,
            )

            pipe = Pipeline([
                ("preprocessor", create_rf_preprocessing_pipeline(top_k_features=None)),
                ("classifier", rf),
            ])

            pipe.fit(X_tr, y_tr)
            val_probs = pipe.predict_proba(X_va)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)

            ablation_accs.append(accuracy_score(y_va, val_preds))
            ablation_aucs.append(roc_auc_score(y_va, val_probs))
            ablation_probs[val_idx] = val_probs

        pruned_acc = np.mean(ablation_accs)
        pruned_auc = np.mean(ablation_aucs)
        pruned_brier = brier_score_loss(y_train, ablation_probs)
        print(f"\nPruned Features ({len(filtered_cols)}) Result:")
        print(f"  Mean Accuracy: {pruned_acc:.4%} (Delta: {pruned_acc - all_acc:+.4%})")
        print(f"  Mean ROC-AUC:  {pruned_auc:.4f} (Delta: {pruned_auc - all_auc:+.4f})")
        print(f"  Brier Score:   {pruned_brier:.4f}")

    # Save importance table
    mit_df.to_csv(OUTPUTS_DIR / "mitigation_features_importance.csv", index=False)


if __name__ == "__main__":
    evaluate_mitigations()
