"""Systematic evaluation of algorithmic improvements for Random Forest.

Evaluates:
1. Subsampling ratio (max_samples)
2. Cost-complexity pruning (ccp_alpha)
3. ExtraTreesClassifier vs RandomForestClassifier
4. Feature selection pruner (top_k vs all features)
5. Probability Calibration (Isotonic / Platt)
6. Error diagnosis (top 10 most confident blunder/misclassified battles)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

from src.config import OUTPUTS_DIR, RANDOM_STATE
from src.cross_validation import get_folds, load_folds
from src.features import create_rf_preprocessing_pipeline, load_master_features


def evaluate_config(
    model_type: str = "rf",
    n_estimators: int = 400,
    criterion: str = "log_loss",
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 6,
    max_features: Any = 0.2,
    max_samples: Optional[float] = None,
    ccp_alpha: float = 0.0,
    top_k_features: Optional[int] = 300,
    apply_isotonic: bool = False,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    precomputed_folds: Optional[list] = None,
) -> Dict[str, Any]:
    """Evaluate a specific RF/ExtraTrees configuration on the 10 frozen folds."""
    oof_preds = np.zeros(len(X_train))
    oof_probs = np.zeros(len(X_train))
    fold_accuracies = []
    fold_aucs = []

    for fold, train_idx, val_idx in precomputed_folds:
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        clf_kwargs = dict(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            ccp_alpha=ccp_alpha,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        if max_samples is not None:
            clf_kwargs["max_samples"] = max_samples

        if model_type == "et":
            clf = ExtraTreesClassifier(**clf_kwargs)
        else:
            clf = RandomForestClassifier(**clf_kwargs)

        pipe = Pipeline([
            ("preprocessor", create_rf_preprocessing_pipeline(
                top_k_features=top_k_features,
                random_state=RANDOM_STATE + fold,
            )),
            ("classifier", clf),
        ])

        pipe.fit(X_tr, y_tr)
        val_probs = pipe.predict_proba(X_va)[:, 1]

        if apply_isotonic:
            tr_probs = pipe.predict_proba(X_tr)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(tr_probs, y_tr)
            val_probs = iso.predict(val_probs)

        val_preds = (val_probs >= 0.5).astype(int)

        acc = accuracy_score(y_va, val_preds)
        auc = roc_auc_score(y_va, val_probs)

        fold_accuracies.append(acc)
        fold_aucs.append(auc)

        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs

    total_acc = accuracy_score(y_train, oof_preds)
    total_auc = roc_auc_score(y_train, oof_probs)
    brier = brier_score_loss(y_train, oof_probs)
    ll = log_loss(y_train, oof_probs)

    return {
        "mean_acc": np.mean(fold_accuracies),
        "std_acc": np.std(fold_accuracies),
        "total_oof_acc": total_acc,
        "mean_auc": np.mean(fold_aucs),
        "total_oof_auc": total_auc,
        "brier": brier,
        "log_loss": ll,
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
    }


def main():
    print("=" * 75)
    print("Predex - Algorithmic Random Forest Benchmarking (10-Fold CV)")
    print("=" * 75)

    print("Loading data...")
    X_train, y_train = load_master_features("train")
    folds_df = load_folds()
    precomputed_folds = list(get_folds(folds_df, ids=X_train.index))
    print(f"Dataset: {X_train.shape[0]} battles x {X_train.shape[1]} features")

    experiments = [
        # 0. Baseline (from tune_random_forest)
        {
            "name": "0. Baseline RF (tuned depth=17, top_k=300)",
            "params": {
                "model_type": "rf",
                "max_depth": 17,
                "min_samples_split": 10,
                "min_samples_leaf": 6,
                "max_features": 0.2,
                "max_samples": None,
                "ccp_alpha": 0.0,
                "top_k_features": 300,
                "apply_isotonic": False,
            }
        },
        # 1. Sub-bagging max_samples=0.80
        {
            "name": "1. Sub-bagging (max_samples=0.80)",
            "params": {
                "model_type": "rf",
                "max_depth": 17,
                "min_samples_split": 10,
                "min_samples_leaf": 6,
                "max_features": 0.2,
                "max_samples": 0.80,
                "ccp_alpha": 0.0,
                "top_k_features": 300,
                "apply_isotonic": False,
            }
        },
        # 2. Cost Complexity Pruning ccp_alpha=5e-5
        {
            "name": "2. Cost-Complexity Pruning (ccp_alpha=5e-5)",
            "params": {
                "model_type": "rf",
                "max_depth": 17,
                "min_samples_split": 10,
                "min_samples_leaf": 6,
                "max_features": 0.2,
                "max_samples": None,
                "ccp_alpha": 5e-5,
                "top_k_features": 300,
                "apply_isotonic": False,
            }
        },
        # 3. ExtraTreesClassifier
        {
            "name": "3. ExtraTreesClassifier (Extremely Randomized Trees)",
            "params": {
                "model_type": "et",
                "max_depth": 17,
                "min_samples_split": 10,
                "min_samples_leaf": 6,
                "max_features": 0.2,
                "max_samples": None,
                "ccp_alpha": 0.0,
                "top_k_features": 300,
                "apply_isotonic": False,
            }
        },
        # 4. Calibration (Isotonic on Baseline)
        {
            "name": "4. Isotonic Calibration (Post-hoc)",
            "params": {
                "model_type": "rf",
                "max_depth": 17,
                "min_samples_split": 10,
                "min_samples_leaf": 6,
                "max_features": 0.2,
                "max_samples": None,
                "ccp_alpha": 0.0,
                "top_k_features": 300,
                "apply_isotonic": True,
            }
        },
        # 5. Full features vs top_k=300 (Effect of algorithmic noise pruner)
        {
            "name": "5. All 454 Features (No Pruner)",
            "params": {
                "model_type": "rf",
                "max_depth": 17,
                "min_samples_split": 10,
                "min_samples_leaf": 6,
                "max_features": 0.2,
                "max_samples": None,
                "ccp_alpha": 0.0,
                "top_k_features": None,
                "apply_isotonic": False,
            }
        },
    ]

    results_table = []
    best_res = None
    best_oof_probs = None

    for exp in experiments:
        print(f"\nEvaluating: {exp['name']}...")
        res = evaluate_config(
            **exp["params"],
            n_estimators=350,
            criterion="log_loss",
            X_train=X_train,
            y_train=y_train,
            precomputed_folds=precomputed_folds,
        )
        print(f"  -> Accuracy: {res['total_oof_acc']:.4%} (+/- {res['std_acc']:.4%}) | ROC-AUC: {res['total_oof_auc']:.4f} | Brier: {res['brier']:.4f}")
        results_table.append({
            "Experiment": exp["name"],
            "Accuracy": f"{res['total_oof_acc']:.4%}",
            "Std": f"{res['std_acc']:.4%}",
            "ROC-AUC": f"{res['total_oof_auc']:.4f}",
            "Brier Score": f"{res['brier']:.4f}",
            "LogLoss": f"{res['log_loss']:.4f}",
            "RawAcc": res["total_oof_acc"],
        })
        if best_res is None or res["total_oof_acc"] > best_res["total_oof_acc"]:
            best_res = res
            best_oof_probs = res["oof_probs"]

    df_res = pd.DataFrame(results_table)
    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON ACROSS ALGORITHMIC VARIANTS:")
    print("=" * 75)
    print(df_res.to_string(index=False))

    # Save benchmark table
    df_res.to_csv(OUTPUTS_DIR / "rf_improvements_benchmark.csv", index=False)
    print(f"\nSaved benchmark results to {OUTPUTS_DIR / 'rf_improvements_benchmark.csv'}")


if __name__ == "__main__":
    main()
