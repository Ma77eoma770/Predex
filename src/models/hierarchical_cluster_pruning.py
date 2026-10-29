"""Hierarchical cluster pruning and architectural ensemble tuning to beat 68.12% CV.

1. Performs Ward linkage clustering on Spearman |rho| correlation matrix (distance threshold 0.12, |rho| > 0.88).
2. Within each cluster of collinear features, retains the single feature with highest importance.
3. Evaluates:
   - RF with log_loss and max_features in [0.12, 0.18, 0.25]
   - ExtraTrees with log_loss
   - Pure Bagging Blend (50% ExtraTrees + 50% RandomForest)
   - Threshold optimization
4. Re-evaluates 10-fold CV and diagnoses the 10 worst cases.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import ID_COL, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL
from src.cross_validation import get_folds, load_folds
from src.features import create_rf_preprocessing_pipeline, load_master_features


def cluster_prune_features(X_train: pd.DataFrame, y_train: pd.Series, correlation_threshold: float = 0.88) -> List[str]:
    """Perform hierarchical cluster pruning to remove collinear clones and retain orthogonal features."""
    print(f"\n[Cluster Pruning] Clustering {X_train.shape[1]} features using Spearman correlation...")

    # Fast initial tree to get ranking metric within clusters
    screening_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    screening_rf.fit(X_train, y_train)
    importances = pd.Series(screening_rf.feature_importances_, index=X_train.columns)

    # Compute correlation matrix
    corr_matrix = X_train.corr(method="spearman").abs().fillna(0.0)
    
    # Distance matrix D = 1 - |rho|
    dist_matrix = 1.0 - corr_matrix.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

    # Ward linkage requires condensed distance matrix
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # Cluster with distance threshold = 1 - 0.88 = 0.12
    dist_thresh = 1.0 - correlation_threshold
    clusters = fcluster(Z, t=dist_thresh, criterion="distance")

    cluster_to_cols: Dict[int, List[str]] = {}
    for col, cl_id in zip(X_train.columns, clusters):
        cluster_to_cols.setdefault(cl_id, []).append(col)

    retained_cols: List[str] = []
    # Always protect the core bipartite physics and regime-switching features!
    protected_prefixes = ("rf_diff_", "rf_greedy_", "rf_ratio_", "rf_comeback_", "rf_throw_", "rf_anti_meta_")

    for cl_id, cols in cluster_to_cols.items():
        if len(cols) == 1:
            retained_cols.append(cols[0])
        else:
            # Pick the column with highest importance in cluster
            best_col = max(cols, key=lambda c: importances.get(c, 0.0))
            retained_cols.append(best_col)
            # If any protected feature was in cluster, keep it as well
            for c in cols:
                if any(c.startswith(p) for p in protected_prefixes) and c not in retained_cols:
                    retained_cols.append(c)

    # Also drop dead features with importance < 0.00015
    retained_cols = [c for c in retained_cols if importances.get(c, 0.0) >= 0.00015 or any(c.startswith(p) for p in protected_prefixes)]

    print(f"[Cluster Pruning] Reduced {X_train.shape[1]} features down to {len(retained_cols)} orthogonal features (cut {X_train.shape[1] - len(retained_cols)} redundant clones).")
    return retained_cols


def run_evaluation():
    print("=" * 80)
    print("Predex - Advanced Physics v2 & Hierarchical Pruning Evaluation")
    print("=" * 80)

    X_train, y_train = load_master_features("train")
    folds_df = load_folds()
    precomputed_folds = list(get_folds(folds_df, ids=X_train.index))
    print(f"Loaded matrix: {X_train.shape[0]} battles x {X_train.shape[1]} features")

    # 1. Cluster Prune Features
    clean_cols = cluster_prune_features(X_train, y_train, correlation_threshold=0.88)
    X_clean = X_train[clean_cols]

    # 2. Benchmarking Architectural Variations
    print("\n[Evaluation] Benchmarking across 10 frozen CV folds...")

    experiments = [
        {
            "name": "1. ExtraTrees (Cleaned ~260 feats, max_feats=0.18, log_loss)",
            "model_type": "et",
            "max_features": 0.18,
            "max_depth": 16,
            "min_samples_leaf": 6,
            "criterion": "log_loss",
            "X": X_clean,
        },
        {
            "name": "2. RandomForest (Cleaned ~260 feats, max_feats=0.18, log_loss)",
            "model_type": "rf",
            "max_features": 0.18,
            "max_depth": 16,
            "min_samples_leaf": 6,
            "criterion": "log_loss",
            "X": X_clean,
        },
        {
            "name": "3. Pure Bagging Blend (50% ET + 50% RF on Cleaned Matrix)",
            "model_type": "blend",
            "max_features": 0.18,
            "max_depth": 16,
            "min_samples_leaf": 6,
            "criterion": "log_loss",
            "X": X_clean,
        },
        {
            "name": "4. Full Matrix (All feats) ExtraTrees (max_feats=0.18)",
            "model_type": "et",
            "max_features": 0.18,
            "max_depth": 16,
            "min_samples_leaf": 6,
            "criterion": "log_loss",
            "X": X_train,
        },
    ]

    records = []
    best_exp_name = None
    best_acc = 0.0
    best_oof_probs = None

    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        curr_X = exp["X"]
        fold_accs = []
        fold_aucs = []
        oof_probs = np.zeros(len(X_train))

        for fold, train_idx, val_idx in precomputed_folds:
            X_tr, y_tr = curr_X.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = curr_X.iloc[val_idx], y_train.iloc[val_idx]

            if exp["model_type"] == "et":
                clf = ExtraTreesClassifier(
                    n_estimators=450,
                    criterion=exp["criterion"],
                    max_depth=exp["max_depth"],
                    min_samples_split=10,
                    min_samples_leaf=exp["min_samples_leaf"],
                    max_features=exp["max_features"],
                    random_state=RANDOM_STATE + fold,
                    n_jobs=-1,
                )
                clf.fit(X_tr, y_tr)
                p_val = clf.predict_proba(X_va)[:, 1]

            elif exp["model_type"] == "rf":
                clf = RandomForestClassifier(
                    n_estimators=450,
                    criterion=exp["criterion"],
                    max_depth=exp["max_depth"],
                    min_samples_split=10,
                    min_samples_leaf=exp["min_samples_leaf"],
                    max_features=exp["max_features"],
                    random_state=RANDOM_STATE + fold,
                    n_jobs=-1,
                )
                clf.fit(X_tr, y_tr)
                p_val = clf.predict_proba(X_va)[:, 1]

            elif exp["model_type"] == "blend":
                et = ExtraTreesClassifier(
                    n_estimators=300,
                    criterion="log_loss",
                    max_depth=16,
                    min_samples_split=10,
                    min_samples_leaf=6,
                    max_features=exp["max_features"],
                    random_state=RANDOM_STATE + fold,
                    n_jobs=-1,
                )
                rf = RandomForestClassifier(
                    n_estimators=300,
                    criterion="log_loss",
                    max_depth=16,
                    min_samples_split=10,
                    min_samples_leaf=6,
                    max_features=exp["max_features"],
                    random_state=RANDOM_STATE + fold,
                    n_jobs=-1,
                )
                et.fit(X_tr, y_tr)
                rf.fit(X_tr, y_tr)
                p_val = 0.5 * et.predict_proba(X_va)[:, 1] + 0.5 * rf.predict_proba(X_va)[:, 1]

            preds = (p_val >= 0.5).astype(int)
            acc = accuracy_score(y_va, preds)
            auc = roc_auc_score(y_va, p_val)

            fold_accs.append(acc)
            fold_aucs.append(auc)
            oof_probs[val_idx] = p_val

        m_acc = np.mean(fold_accs)
        s_acc = np.std(fold_accs)
        m_auc = np.mean(fold_aucs)
        brier = brier_score_loss(y_train, oof_probs)
        ll = log_loss(y_train, oof_probs)

        print(f"  -> Accuracy: {m_acc:.4%} (+/- {s_acc:.4%}) | ROC-AUC: {m_auc:.4f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}")

        records.append({
            "Experiment": exp["name"],
            "Accuracy": f"{m_acc:.4%}",
            "Std": f"{s_acc:.4%}",
            "ROC-AUC": f"{m_auc:.4f}",
            "Brier Score": f"{brier:.4f}",
            "LogLoss": f"{ll:.4f}",
            "RawAcc": m_acc,
        })

        if m_acc > best_acc:
            best_acc = m_acc
            best_exp_name = exp["name"]
            best_oof_probs = oof_probs

    df_res = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TO BREAK 68.12%:")
    print("=" * 80)
    print(df_res.to_string(index=False))

    # Threshold Optimization
    best_th = 0.5
    best_th_acc = best_acc
    for th in np.linspace(0.48, 0.52, 41):
        th_acc = accuracy_score(y_train, (best_oof_probs >= th).astype(int))
        if th_acc > best_th_acc:
            best_th_acc = th_acc
            best_th = th

    print(f"\nBest Config: {best_exp_name} -> Raw Acc = {best_acc:.4%}")
    if best_th != 0.5:
        print(f"With Threshold Tuning (th={best_th:.4f}) -> Accuracy = {best_th_acc:.4%}")

    # Retrain on full and produce test submission and OOF
    print("\n[Export] Retraining best model on full training set...")
    best_clean_cols = clean_cols if "Cleaned" in best_exp_name else list(X_train.columns)
    X_train_final = X_train[best_clean_cols]
    X_test, _ = load_master_features("test")
    X_test_final = X_test[best_clean_cols]

    if "Blend" in best_exp_name:
        et = ExtraTreesClassifier(n_estimators=400, criterion="log_loss", max_depth=16, min_samples_split=10, min_samples_leaf=6, max_features=0.18, random_state=RANDOM_STATE, n_jobs=-1)
        rf = RandomForestClassifier(n_estimators=400, criterion="log_loss", max_depth=16, min_samples_split=10, min_samples_leaf=6, max_features=0.18, random_state=RANDOM_STATE, n_jobs=-1)
        et.fit(X_train_final, y_train)
        rf.fit(X_train_final, y_train)
        test_probs = 0.5 * et.predict_proba(X_test_final)[:, 1] + 0.5 * rf.predict_proba(X_test_final)[:, 1]
    elif "RandomForest" in best_exp_name:
        rf = RandomForestClassifier(n_estimators=500, criterion="log_loss", max_depth=16, min_samples_split=10, min_samples_leaf=6, max_features=0.18, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_final, y_train)
        test_probs = rf.predict_proba(X_test_final)[:, 1]
    else:
        et = ExtraTreesClassifier(n_estimators=500, criterion="log_loss", max_depth=16, min_samples_split=10, min_samples_leaf=6, max_features=0.18, random_state=RANDOM_STATE, n_jobs=-1)
        et.fit(X_train_final, y_train)
        test_probs = et.predict_proba(X_test_final)[:, 1]

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
    run_evaluation()
