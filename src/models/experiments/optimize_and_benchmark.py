"""Iterative optimization, feature pruning, and benchmarking suite to beat 67.70% on 10-fold CV.

1. Trains RF on all 485 features and identifies dead / noisy features to prune.
2. Benchmarks All 485 vs Pruned Feature Store on 10-fold CV.
3. Evaluates Sub-bagging and ExtraTrees.
4. Generates updated submission, OOF, and re-diagnoses the 10 worst cases.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from src.config import ID_COL, OUTPUTS_DIR, RANDOM_STATE, SAMPLE_SUBMISSION_CSV, TARGET_COL
from src.cross_validation import get_folds, load_folds
from src.features import create_rf_preprocessing_pipeline, load_master_features


def run_optimization():
    print("=" * 75)
    print("Predex - Iterative Optimization to Beat 67.70% (10-Fold CV)")
    print("=" * 75)

    X_train, y_train = load_master_features("train")
    folds_df = load_folds()
    precomputed_folds = list(get_folds(folds_df, ids=X_train.index))
    print(f"Loaded train matrix: {X_train.shape[0]} battles x {X_train.shape[1]} features")

    # Step 1: Feature screening to find dead features
    print("\n[Step 1] Screening feature importances across a 300-tree RF...")
    screening_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=17,
        min_samples_split=10,
        min_samples_leaf=6,
        max_features=0.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    screening_rf.fit(X_train, y_train)

    importances = pd.Series(screening_rf.feature_importances_, index=X_train.columns)
    
    # Sort and find dead / low noise features (< 0.00018)
    dead_features = importances[importances < 0.00018].index.tolist()
    print(f"Found {len(dead_features)} dead / high-noise features (< 0.00018 imp).")

    # Cleaned feature set
    clean_cols = [c for c in X_train.columns if c not in dead_features]
    print(f"Retained {len(clean_cols)} active features.")

    # Step 2: Compare 10-fold CV on All (485) vs Cleaned (~400)
    print("\n[Step 2] Comparing 10-Fold CV: Full 485 vs Cleaned...")

    configs = [
        {"name": f"Full Matrix ({X_train.shape[1]} feats)", "X": X_train, "model": "rf", "max_samples": None},
        {"name": f"Cleaned Matrix ({len(clean_cols)} feats)", "X": X_train[clean_cols], "model": "rf", "max_samples": None},
        {"name": f"Cleaned + Sub-bagging (0.85)", "X": X_train[clean_cols], "model": "rf", "max_samples": 0.85},
        {"name": f"Cleaned + ExtraTrees", "X": X_train[clean_cols], "model": "et", "max_samples": None},
    ]

    benchmark_records = []
    best_config_name = None
    best_acc = 0.0
    best_oof_probs = None
    best_clean_cols = None

    for cfg in configs:
        print(f"\nEvaluating: {cfg['name']}...")
        X_curr = cfg["X"]
        fold_accs = []
        fold_aucs = []
        oof_probs = np.zeros(len(X_train))

        for fold, train_idx, val_idx in precomputed_folds:
            X_tr, y_tr = X_curr.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_curr.iloc[val_idx], y_train.iloc[val_idx]

            clf_kwargs = dict(
                n_estimators=450,
                criterion="log_loss",
                max_depth=17,
                min_samples_split=10,
                min_samples_leaf=6,
                max_features=0.2,
                random_state=RANDOM_STATE + fold,
                n_jobs=-1,
            )
            if cfg["max_samples"] is not None:
                clf_kwargs["max_samples"] = cfg["max_samples"]

            if cfg["model"] == "et":
                clf = ExtraTreesClassifier(**clf_kwargs)
            else:
                clf = RandomForestClassifier(**clf_kwargs)

            pipe = Pipeline([
                ("preprocessor", create_rf_preprocessing_pipeline(top_k_features=None)),
                ("classifier", clf),
            ])

            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fold_accs.append(accuracy_score(y_va, preds))
            fold_aucs.append(roc_auc_score(y_va, probs))
            oof_probs[val_idx] = probs

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_auc = np.mean(fold_aucs)
        brier = brier_score_loss(y_train, oof_probs)
        ll = log_loss(y_train, oof_probs)

        print(f"  -> Accuracy: {mean_acc:.4%} (+/- {std_acc:.4%}) | ROC-AUC: {mean_auc:.4f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}")

        benchmark_records.append({
            "Configuration": cfg["name"],
            "Accuracy": f"{mean_acc:.4%}",
            "Std": f"{std_acc:.4%}",
            "ROC-AUC": f"{mean_auc:.4f}",
            "Brier": f"{brier:.4f}",
            "LogLoss": f"{ll:.4f}",
            "RawAcc": mean_acc,
        })

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_config_name = cfg["name"]
            best_oof_probs = oof_probs
            best_clean_cols = X_curr.columns.tolist()

    df_bench = pd.DataFrame(benchmark_records)
    print("\n" + "=" * 75)
    print("BENCHMARK COMPARISON TO BEAT 67.70%:")
    print("=" * 75)
    print(df_bench.to_string(index=False))

    # Check best threshold
    best_th = 0.5
    best_th_acc = best_acc
    for th in np.linspace(0.48, 0.52, 41):
        th_acc = accuracy_score(y_train, (best_oof_probs >= th).astype(int))
        if th_acc > best_th_acc:
            best_th_acc = th_acc
            best_th = th

    print(f"\nBest Config: {best_config_name} with Accuracy = {best_acc:.4%}")
    if best_th != 0.5:
        print(f"With Threshold Tuning (th={best_th:.4f}): Accuracy = {best_th_acc:.4%}")

    # Step 3: Retrain and produce submission and updated OOF
    print(f"\n[Step 3] Retraining best model on full training set ({len(best_clean_cols)} features)...")
    X_train_best = X_train[best_clean_cols]
    final_rf = RandomForestClassifier(
        n_estimators=500,
        criterion="log_loss",
        max_depth=17,
        min_samples_split=10,
        min_samples_leaf=6,
        max_features=0.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_pipe = Pipeline([
        ("preprocessor", create_rf_preprocessing_pipeline(top_k_features=None)),
        ("classifier", final_rf),
    ])
    final_pipe.fit(X_train_best, y_train)

    X_test, _ = load_master_features("test")
    X_test_best = X_test[best_clean_cols]
    test_probs = final_pipe.predict_proba(X_test_best)[:, 1]
    test_preds = (test_probs >= best_th).astype(int)

    # Save submission
    sub_df = pd.DataFrame({
        ID_COL: X_test.index,
        TARGET_COL: test_preds,
    })
    sub_path = OUTPUTS_DIR / "submission_random_forest.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"[SAVED] Final submission -> {sub_path}")

    # Save OOF
    oof_df = pd.DataFrame({
        ID_COL: X_train.index,
        TARGET_COL: y_train.values,
        "pred_prob": best_oof_probs,
        "pred_class": (best_oof_probs >= best_th).astype(int),
    })
    oof_path = OUTPUTS_DIR / "oof_random_forest.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[SAVED] Final OOF -> {oof_path}")


if __name__ == "__main__":
    run_optimization()
