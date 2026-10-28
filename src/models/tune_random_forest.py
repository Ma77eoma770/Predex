"""Optuna Hyperparameter Tuning for Random Forest on Frozen 10-Fold CV.

Explores:
- criterion: 'gini', 'entropy', 'log_loss'
- max_depth: [12, 26]
- min_samples_split: [4, 25]
- min_samples_leaf: [2, 12]
- max_features: 'sqrt', 'log2', 0.15, 0.20, 0.25, 0.30
- n_estimators: 300 (fast exploration)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.config import OUTPUTS_DIR, RANDOM_STATE
from src.cross_validation import get_folds, load_folds
from src.features import create_rf_preprocessing_pipeline, load_master_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_tuning(n_trials: int = 25, n_estimators: int = 300):
    print("=" * 70)
    print("Predex - Systematic Random Forest Tuning (Optuna)")
    print("=" * 70)
    print(f"Running {n_trials} trials across official 10-fold CV frozen splits...")

    X_train, y_train = load_master_features("train")
    folds_df = load_folds()
    precomputed_folds = list(get_folds(folds_df, ids=X_train.index))
    print(f"Train matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")

    def objective(trial: optuna.Trial) -> float:
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        max_depth = trial.suggest_int("max_depth", 12, 26)
        min_samples_split = trial.suggest_int("min_samples_split", 4, 25)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
        max_features_type = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "0.15", "0.20", "0.25", "0.30"])

        if max_features_type in ("sqrt", "log2"):
            max_features = max_features_type
        else:
            max_features = float(max_features_type)

        fold_accuracies = []

        for fold, train_idx, val_idx in precomputed_folds:
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            pipeline = Pipeline([
                ("preprocessor", create_rf_preprocessing_pipeline()),
                ("classifier", rf),
            ])

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_va)
            acc = accuracy_score(y_va, preds)
            fold_accuracies.append(acc)

        mean_acc = float(np.mean(fold_accuracies))
        return mean_acc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 70)
    print("Optuna Tuning Completed!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best 10-Fold CV Accuracy: {study.best_value:.4%}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:<20}: {v}")
    print("=" * 70)

    # Convert best params for saving
    best_params = dict(study.best_params)
    if "max_features_type" in best_params:
        mft = best_params.pop("max_features_type")
        best_params["max_features"] = mft if mft in ("sqrt", "log2") else float(mft)
    best_params["n_estimators"] = 400

    params_file = OUTPUTS_DIR / "rf_best_params.json"
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[SAVED] Best hyperparameters -> {params_file}")
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Random Forest with Optuna")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--trees", type=int, default=300, help="Trees per trial")
    args = parser.parse_args()

    run_tuning(n_trials=args.trials, n_estimators=args.trees)
