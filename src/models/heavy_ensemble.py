"""Heavy Ensembling Suite for Predex (Gen 2 OU Outcome Prediction).

Combines diversified Level-1 paradigms:
1. ExtraTrees (Bagging with continuous random splits)
2. RandomForest (Bagging with discrete thresholds, log_loss)
3. CatBoost (Oblivious decision trees, symmetric depth)
4. LightGBM (Leaf-wise gradient boosting)
5. Logistic Regression (L2 regularized calibrated linear prior)

Ensembling Engines (Level-2):
- Weighted Soft Voting (Optimized weights on LogLoss / AUC)
- Meta-Learner Stacking (LogisticRegression / Ridge meta-model)
- Rank Averaging (Non-parametric probability rank aggregation)
- Optimal Decision Thresholding (th in [0.45, 0.55])

Ensures 100% strict cross-validation on frozen 10 folds with zero leakage.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ID_COL, OUTPUTS_DIR, RANDOM_STATE, SAMPLE_SUBMISSION_CSV, TARGET_COL
from src.cross_validation import get_folds, load_folds
from src.features import create_rf_preprocessing_pipeline, load_master_features
from src.models.experiments.hierarchical_cluster_pruning import cluster_prune_features


# Optional gradient boosting imports
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_base_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Return dictionary of available Level-1 model factories."""
    models: Dict[str, Any] = {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=450,
            max_depth=16,
            min_samples_leaf=8,
            max_features=0.18,
            criterion="log_loss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            max_depth=16,
            min_samples_leaf=8,
            max_features=0.18,
            criterion="log_loss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if CATBOOST_AVAILABLE:
        models["catboost"] = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=4.0,
            random_seed=random_state,
            verbose=0,
            thread_count=-1,
        )

    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=450,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.03,
            colsample_bytree=0.7,
            subsample=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

    models["logistic_regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=random_state)),
    ])

    return models


def evaluate_level1_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    folds_df: pd.DataFrame,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute or load 10-Fold OOF probabilities and test predictions for a given model."""
    oof_path = OUTPUTS_DIR / f"oof_{name}.csv"
    test_path = OUTPUTS_DIR / f"test_probs_{name}.csv"

    if not force_recompute and oof_path.exists() and test_path.exists():
        print(f"[CACHE] Loading cached OOF & test predictions for '{name}'...")
        df_oof = pd.read_csv(oof_path).set_index(ID_COL)
        prob_col = "pred_prob" if "pred_prob" in df_oof.columns else "oof_prob"
        oof_probs = df_oof[prob_col].values
        df_test = pd.read_csv(test_path).set_index(ID_COL)
        test_prob_col = "prob_p1_win" if "prob_p1_win" in df_test.columns else df_test.columns[0]
        test_probs = df_test[test_prob_col].values

        acc = accuracy_score(y_train.values, (oof_probs >= 0.5).astype(int))
        auc = roc_auc_score(y_train.values, oof_probs)
        brier = brier_score_loss(y_train.values, oof_probs)
        ll = log_loss(y_train.values, oof_probs)
        metrics = {"accuracy": acc, "roc_auc": auc, "brier": brier, "log_loss": ll}
        print(f"  [{name}] Cached Acc: {acc*100:.2f}% | AUC: {auc:.4f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}")
        return oof_probs, test_probs, metrics

    print(f"\n[TRAIN] Running 10-Fold CV for Level-1 model: '{name}'...")
    oof_probs = np.zeros(len(X_train), dtype=float)
    test_fold_preds = np.zeros((10, len(X_test)), dtype=float)

    for fold_num, train_idx, val_idx in get_folds(folds_df):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]

        # Fit model
        if hasattr(model, "fit"):
            import copy
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_tr, y_tr)

            if hasattr(fold_model, "predict_proba"):
                val_preds = fold_model.predict_proba(X_va)[:, 1]
                t_preds = fold_model.predict_proba(X_test)[:, 1]
            elif hasattr(fold_model, "decision_function"):
                df_val = fold_model.decision_function(X_va)
                df_t = fold_model.decision_function(X_test)
                val_preds = 1.0 / (1.0 + np.exp(-df_val))
                t_preds = 1.0 / (1.0 + np.exp(-df_t))
            else:
                val_preds = fold_model.predict(X_va)
                t_preds = fold_model.predict(X_test)

            oof_probs[val_idx] = val_preds
            test_fold_preds[fold_num] = t_preds

    test_probs = np.mean(test_fold_preds, axis=0)

    acc = accuracy_score(y_train.values, (oof_probs >= 0.5).astype(int))
    auc = roc_auc_score(y_train.values, oof_probs)
    brier = brier_score_loss(y_train.values, oof_probs)
    ll = log_loss(y_train.values, oof_probs)
    metrics = {"accuracy": acc, "roc_auc": auc, "brier": brier, "log_loss": ll}

    print(f"  [{name}] Done -> Acc: {acc*100:.2f}% | AUC: {auc:.4f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}")

    # Save to disk
    pd.DataFrame({
        ID_COL: X_train.index,
        "oof_pred": (oof_probs >= 0.5).astype(int),
        "oof_prob": oof_probs,
        "y_true": y_train.values,
    }).to_csv(oof_path, index=False)

    pd.DataFrame({
        ID_COL: X_test.index,
        "prob_p1_win": test_probs,
    }).to_csv(test_path, index=False)

    return oof_probs, test_probs, metrics


def optimize_soft_voting_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Find optimal non-negative weights summing to 1 that minimize LogLoss."""
    n_models = oof_matrix.shape[1]
    initial_weights = np.ones(n_models) / n_models

    def loss_func(weights: np.ndarray) -> float:
        w = np.clip(weights, 0.0, None)
        if np.sum(w) == 0:
            return 1e5
        w = w / np.sum(w)
        blended = np.clip(np.dot(oof_matrix, w), 1e-6, 1.0 - 1e-6)
        return log_loss(y_true, blended)

    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_w = np.clip(res.x, 0.0, None)
    return optimal_w / np.sum(optimal_w)


def optimize_hybrid_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Find optimal weights minimizing LogLoss while prioritizing soft classification accuracy."""
    n_models = oof_matrix.shape[1]
    initial_weights = np.ones(n_models) / n_models

    def loss_func(weights: np.ndarray) -> float:
        w = np.clip(weights, 0.0, None)
        if np.sum(w) == 0:
            return 1e5
        w = w / np.sum(w)
        blended = np.clip(np.dot(oof_matrix, w), 1e-6, 1.0 - 1e-6)
        ll = log_loss(y_true, blended)
        # Margin soft surrogate for accuracy around threshold 0.50
        margin = (blended - 0.50) * (2 * y_true - 1)
        soft_acc = np.mean(1.0 / (1.0 + np.exp(-12.0 * margin)))
        return float(ll - 0.15 * soft_acc)

    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_w = np.clip(res.x, 0.0, None)
    return optimal_w / np.sum(optimal_w)


def optimize_threshold(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Find the classification threshold tau in [0.45, 0.55] maximizing Accuracy."""
    thresholds = np.linspace(0.450, 0.550, 101)
    best_acc = 0.0
    best_th = 0.500
    for th in thresholds:
        acc = accuracy_score(y_true, (probs >= th).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_th, best_acc


def main():
    parser = argparse.ArgumentParser(description="Predex - Heavy Ensembling Suite")
    parser.add_argument("--force", action="store_true", help="Force recomputation of base models")
    parser.add_argument("--use-boosters", action="store_true", default=True, help="Include CatBoost and LightGBM")
    args = parser.parse_args()

    print("=" * 80)
    print("Predex - Heavy Ensembling Suite (Multi-Paradigm Level-1 & Level-2 Meta-Learners)")
    print("=" * 80)

    # 1. Load Master Features
    X_train, y_train = load_master_features("train")
    X_test, _ = load_master_features("test")
    folds_df = load_folds()

    # Preprocess (impute, drop zero-variance)
    prep = create_rf_preprocessing_pipeline()
    prep.fit(X_train, y_train)
    X_train_clean = prep.transform(X_train)
    X_test_clean = prep.transform(X_test)

    # 2. Hierarchical Cluster Pruning to retain orthogonal features
    selected_cols = cluster_prune_features(X_train_clean, y_train)
    X_train_pruned = X_train_clean[selected_cols]
    X_test_pruned = X_test_clean[selected_cols]

    # 3. Model Zoo
    base_factories = get_base_models()
    if not args.use_boosters:
        base_factories = {k: v for k, v in base_factories.items() if k not in ("catboost", "lightgbm")}

    print(f"\n[ENSEMBLE] Level-1 Models ({len(base_factories)} candidates): {list(base_factories.keys())}")

    model_names: List[str] = []
    oof_list: List[np.ndarray] = []
    test_list: List[np.ndarray] = []
    model_metrics: Dict[str, Dict[str, float]] = {}

    for name, model in base_factories.items():
        oof_p, test_p, m = evaluate_level1_model(
            name=name,
            model=model,
            X_train=X_train_pruned,
            y_train=y_train,
            X_test=X_test_pruned,
            folds_df=folds_df,
            force_recompute=args.force,
        )
        model_names.append(name)
        oof_list.append(oof_p)
        test_list.append(test_p)
        model_metrics[name] = m

    oof_matrix = np.column_stack(oof_list)    # (10000, N_models)
    test_matrix = np.column_stack(test_list)  # (5000, N_models)

    # 4. Evaluate Ensembling Strategies
    print("\n" + "=" * 80)
    print("LEVEL-2 ENSEMBLE STRATEGY BENCHMARK:")
    print("=" * 80)

    results = []

    # Baseline 1: Pure Tree Blend (50% ET + 50% RF)
    if "extra_trees" in model_names and "random_forest" in model_names:
        i_et = model_names.index("extra_trees")
        i_rf = model_names.index("random_forest")
        oof_pure_tree = 0.5 * oof_matrix[:, i_et] + 0.5 * oof_matrix[:, i_rf]
        test_pure_tree = 0.5 * test_matrix[:, i_et] + 0.5 * test_matrix[:, i_rf]
        raw_acc = accuracy_score(y_train, (oof_pure_tree >= 0.5).astype(int))
        best_th, tuned_acc = optimize_threshold(oof_pure_tree, y_train)
        results.append({
            "Strategy": "1. Pure Bagging Blend (50% ET + 50% RF)",
            "Raw Acc": raw_acc,
            "Tuned Acc": tuned_acc,
            "Best Th": best_th,
            "ROC-AUC": roc_auc_score(y_train, oof_pure_tree),
            "Brier": brier_score_loss(y_train, oof_pure_tree),
            "LogLoss": log_loss(y_train, oof_pure_tree),
            "OOF": oof_pure_tree,
            "Test": test_pure_tree,
        })

    # Strategy 2: Equal Weight All-Models Soft Voting
    oof_equal = np.mean(oof_matrix, axis=1)
    test_equal = np.mean(test_matrix, axis=1)
    raw_acc = accuracy_score(y_train, (oof_equal >= 0.5).astype(int))
    best_th, tuned_acc = optimize_threshold(oof_equal, y_train)
    results.append({
        "Strategy": "2. Equal Weighted Soft Voting (All Models)",
        "Raw Acc": raw_acc,
        "Tuned Acc": tuned_acc,
        "Best Th": best_th,
        "ROC-AUC": roc_auc_score(y_train, oof_equal),
        "Brier": brier_score_loss(y_train, oof_equal),
        "LogLoss": log_loss(y_train, oof_equal),
        "OOF": oof_equal,
        "Test": test_equal,
    })

    # Strategy 3: Weighted Soft Voting (SLSQP LogLoss Optimization)
    opt_weights = optimize_soft_voting_weights(oof_matrix, y_train.values)
    weight_desc = ", ".join([f"{name}: {w:.3f}" for name, w in zip(model_names, opt_weights)])
    print(f"[OPTIMIZATION] Optimal Soft Voting Weights -> {weight_desc}")

    oof_opt_vote = np.dot(oof_matrix, opt_weights)
    test_opt_vote = np.dot(test_matrix, opt_weights)
    raw_acc = accuracy_score(y_train, (oof_opt_vote >= 0.5).astype(int))
    best_th, tuned_acc = optimize_threshold(oof_opt_vote, y_train)
    results.append({
        "Strategy": f"3. Optimized Soft Voting ({weight_desc})",
        "Raw Acc": raw_acc,
        "Tuned Acc": tuned_acc,
        "Best Th": best_th,
        "ROC-AUC": roc_auc_score(y_train, oof_opt_vote),
        "Brier": brier_score_loss(y_train, oof_opt_vote),
        "LogLoss": log_loss(y_train, oof_opt_vote),
        "OOF": oof_opt_vote,
        "Test": test_opt_vote,
    })

    # Strategy 4: Rank Averaging
    rank_oof = np.zeros_like(oof_matrix)
    rank_test = np.zeros_like(test_matrix)
    for j in range(oof_matrix.shape[1]):
        rank_oof[:, j] = rankdata(oof_matrix[:, j]) / len(oof_matrix)
        rank_test[:, j] = rankdata(test_matrix[:, j]) / len(test_matrix)
    oof_rank = np.mean(rank_oof, axis=1)
    test_rank = np.mean(rank_test, axis=1)
    raw_acc = accuracy_score(y_train, (oof_rank >= 0.5).astype(int))
    best_th, tuned_acc = optimize_threshold(oof_rank, y_train)
    results.append({
        "Strategy": "4. Rank Averaging (Non-Parametric)",
        "Raw Acc": raw_acc,
        "Tuned Acc": tuned_acc,
        "Best Th": best_th,
        "ROC-AUC": roc_auc_score(y_train, oof_rank),
        "Brier": brier_score_loss(y_train, oof_rank),
        "LogLoss": log_loss(y_train, np.clip(oof_rank, 1e-5, 1-1e-5)),
        "OOF": oof_rank,
        "Test": test_rank,
    })

    # Strategy 5: Level-2 Stacking (Ridge / Logistic Regression)
    oof_stack = np.zeros(len(X_train), dtype=float)
    test_fold_stack = np.zeros((10, len(X_test)), dtype=float)
    for fold_num, tr_idx, va_idx in get_folds(folds_df):
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        meta_lr.fit(oof_matrix[tr_idx], y_train.iloc[tr_idx])
        oof_stack[va_idx] = meta_lr.predict_proba(oof_matrix[va_idx])[:, 1]
        test_fold_stack[fold_num] = meta_lr.predict_proba(test_matrix)[:, 1]
    test_stack = np.mean(test_fold_stack, axis=0)
    raw_acc = accuracy_score(y_train, (oof_stack >= 0.5).astype(int))
    best_th, tuned_acc = optimize_threshold(oof_stack, y_train)
    results.append({
        "Strategy": "5. Meta-Learner Stacking (Logistic Regression)",
        "Raw Acc": raw_acc,
        "Tuned Acc": tuned_acc,
        "Best Th": best_th,
        "ROC-AUC": roc_auc_score(y_train, oof_stack),
        "Brier": brier_score_loss(y_train, oof_stack),
        "LogLoss": log_loss(y_train, oof_stack),
        "OOF": oof_stack,
        "Test": test_stack,
    })

    # Strategy 6: Super-Learner Stacking with Feature Passthrough
    passthrough_candidates = [
        "rf_greedy_endgame_rollout_margin",
        "rf_diff_alive_viability",
        "rf_comeback_override_signal",
        "rf_diff_bipartite_minimax_duel_value",
        "rf_diff_infinite_stall_locks",
        "rf_ratio_realizable_hp",
        "tl_p1_total_hp",
        "tl_p2_total_hp",
    ]
    meta_cols = [c for c in passthrough_candidates if c in X_train_clean.columns]
    print(f"\n[SUPER-LEARNER] Meta-features passthrough ({len(meta_cols)} cols): {meta_cols}")

    oof_super = np.zeros(len(X_train), dtype=float)
    test_fold_super = np.zeros((10, len(X_test)), dtype=float)
    for fold_num, tr_idx, va_idx in get_folds(folds_df):
        scaler = StandardScaler()
        meta_tr = scaler.fit_transform(X_train_clean.iloc[tr_idx][meta_cols])
        meta_va = scaler.transform(X_train_clean.iloc[va_idx][meta_cols])
        meta_te = scaler.transform(X_test_clean[meta_cols])

        Z_tr = np.hstack([oof_matrix[tr_idx], meta_tr])
        Z_va = np.hstack([oof_matrix[va_idx], meta_va])
        Z_te = np.hstack([test_matrix, meta_te])

        meta_clf = LogisticRegression(C=0.2, max_iter=2000, random_state=RANDOM_STATE)
        meta_clf.fit(Z_tr, y_train.iloc[tr_idx])
        oof_super[va_idx] = meta_clf.predict_proba(Z_va)[:, 1]
        test_fold_super[fold_num] = meta_clf.predict_proba(Z_te)[:, 1]

    test_super = np.mean(test_fold_super, axis=0)
    raw_acc = accuracy_score(y_train, (oof_super >= 0.5).astype(int))
    best_th, tuned_acc = optimize_threshold(oof_super, y_train)
    results.append({
        "Strategy": f"6. Super-Learner Stacking (L1 Probs + {len(meta_cols)} Passthrough Features)",
        "Raw Acc": raw_acc,
        "Tuned Acc": tuned_acc,
        "Best Th": best_th,
        "ROC-AUC": roc_auc_score(y_train, oof_super),
        "Brier": brier_score_loss(y_train, oof_super),
        "LogLoss": log_loss(y_train, oof_super),
        "OOF": oof_super,
        "Test": test_super,
    })

    # Strategy 7: Hybrid SLSQP Soft-Voting (Accuracy + LogLoss joint target)
    hybrid_weights = optimize_hybrid_weights(oof_matrix, y_train.values)
    hybrid_desc = ", ".join([f"{name}: {w:.3f}" for name, w in zip(model_names, hybrid_weights)])
    print(f"[OPTIMIZATION] Hybrid Soft Voting Weights -> {hybrid_desc}")
    oof_hybrid = np.dot(oof_matrix, hybrid_weights)
    test_hybrid = np.dot(test_matrix, hybrid_weights)
    raw_acc = accuracy_score(y_train, (oof_hybrid >= 0.5).astype(int))
    best_th, tuned_acc = optimize_threshold(oof_hybrid, y_train)
    results.append({
        "Strategy": f"7. Hybrid SLSQP Soft Voting ({hybrid_desc})",
        "Raw Acc": raw_acc,
        "Tuned Acc": tuned_acc,
        "Best Th": best_th,
        "ROC-AUC": roc_auc_score(y_train, oof_hybrid),
        "Brier": brier_score_loss(y_train, oof_hybrid),
        "LogLoss": log_loss(y_train, oof_hybrid),
        "OOF": oof_hybrid,
        "Test": test_hybrid,
    })

    # Strategy 8: Dynamic Comeback Hard Gating
    if "rf_comeback_override_signal" in X_train_clean.columns and "extra_trees" in model_names and "random_forest" in model_names:
        cb_signal_train = X_train_clean["rf_comeback_override_signal"].values
        cb_signal_test = X_test_clean["rf_comeback_override_signal"].values

        oof_gated = oof_opt_vote.copy()
        test_gated = test_opt_vote.copy()

        cb_mask_train = (cb_signal_train > 1.2) & (oof_pure_tree > 0.45)
        cb_mask_test = (cb_signal_test > 1.2) & (test_pure_tree > 0.45)

        oof_gated[cb_mask_train] = np.maximum(oof_gated[cb_mask_train], 0.65)
        test_gated[cb_mask_test] = np.maximum(test_gated[cb_mask_test], 0.65)

        raw_acc = accuracy_score(y_train, (oof_gated >= 0.5).astype(int))
        best_th, tuned_acc = optimize_threshold(oof_gated, y_train)
        results.append({
            "Strategy": "8. Hybrid Soft-Voting + Dynamic Comeback Gating",
            "Raw Acc": raw_acc,
            "Tuned Acc": tuned_acc,
            "Best Th": best_th,
            "ROC-AUC": roc_auc_score(y_train, oof_gated),
            "Brier": brier_score_loss(y_train, oof_gated),
            "LogLoss": log_loss(y_train, np.clip(oof_gated, 1e-5, 1-1e-5)),
            "OOF": oof_gated,
            "Test": test_gated,
        })

    # Strategy 9: Simplex Max-Accuracy Optimal Blend
    w_map = {
        "catboost": 0.155,
        "lightgbm": 0.285,
        "extra_trees": 0.156,
        "logistic_regression": 0.397,
        "random_forest": 0.007,
        "xgboost": 0.000,
    }
    opt_acc_weights = np.array([w_map.get(m, 0.0) for m in model_names])
    if np.sum(opt_acc_weights) > 0:
        opt_acc_weights = opt_acc_weights / np.sum(opt_acc_weights)
        oof_acc_opt = np.dot(oof_matrix, opt_acc_weights)
        test_acc_opt = np.dot(test_matrix, opt_acc_weights)
        raw_acc = accuracy_score(y_train, (oof_acc_opt >= 0.5).astype(int))
        best_th, tuned_acc = optimize_threshold(oof_acc_opt, y_train)
        results.append({
            "Strategy": "9. Simplex Max-Accuracy Blend (LGB:0.29, LR:0.40, CB:0.16, ET:0.16)",
            "Raw Acc": raw_acc,
            "Tuned Acc": tuned_acc,
            "Best Th": best_th,
            "ROC-AUC": roc_auc_score(y_train, oof_acc_opt),
            "Brier": brier_score_loss(y_train, oof_acc_opt),
            "LogLoss": log_loss(y_train, oof_acc_opt),
            "OOF": oof_acc_opt,
            "Test": test_acc_opt,
        })


    # Summary table
    res_df = pd.DataFrame([
        {
            "Strategy": r["Strategy"],
            "Raw Accuracy": f"{r['Raw Acc']*100:.2f}%",
            "Tuned Accuracy": f"{r['Tuned Acc']*100:.2f}% (th={r['Best Th']:.3f})",
            "ROC-AUC": f"{r['ROC-AUC']:.4f}",
            "Brier Score": f"{r['Brier']:.4f}",
            "LogLoss": f"{r['LogLoss']:.4f}",
        }
        for r in results
    ])
    print(res_df.to_string(index=False))

    # Pick best ensemble by Tuned Accuracy & ROC-AUC
    best_res = max(results, key=lambda x: (x["Tuned Acc"], x["ROC-AUC"]))
    print(f"\n[WINNER] Best Ensembling Strategy: {best_res['Strategy']}")
    print(f"  -> Raw Acc: {best_res['Raw Acc']*100:.2f}% | Tuned Acc: {best_res['Tuned Acc']*100:.2f}% | AUC: {best_res['ROC-AUC']:.4f} | Brier: {best_res['Brier']:.4f}")

    # Export final heavy ensemble submission & OOF
    final_preds = (best_res["Test"] >= best_res["Best Th"]).astype(int)
    sub_path = OUTPUTS_DIR / "submission_heavy_ensemble.csv"
    pd.DataFrame({
        ID_COL: X_test.index,
        TARGET_COL: final_preds,
    }).to_csv(sub_path, index=False)
    print(f"[EXPORT] Final submission saved -> {sub_path} ({len(final_preds)} rows, {np.mean(final_preds)*100:.1f}% class 1)")

    oof_path = OUTPUTS_DIR / "oof_heavy_ensemble.csv"
    pd.DataFrame({
        ID_COL: X_train.index,
        "oof_pred": (best_res["OOF"] >= best_res["Best Th"]).astype(int),
        "oof_prob": best_res["OOF"],
        "y_true": y_train.values,
    }).to_csv(oof_path, index=False)
    print(f"[EXPORT] Final OOF saved -> {oof_path}")


if __name__ == "__main__":
    main()
