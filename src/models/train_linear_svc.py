"""LinearSVC on feature set v1, compared with the team's logistic baseline on the same folds.

- Splits come exclusively from data/train_folds.csv (10 frozen folds).
- C is tuned with GridSearchCV on a logarithmic grid; convergence is tracked per C and fold.
- The logistic baseline is recomputed on the same folds, both with the team preprocessing
  and with the new shared preprocessing (so the second comparison changes only the model).
- Out-of-fold decision scores of the best C are saved for the final stacking.

Usage:
    python src/models/train_linear_svc.py          # print only
    python src/models/train_linear_svc.py --log    # also append rows to results/experiments.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import ID_COL, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import get_cv_splits, log_result
from src.features import create_linear_preprocessing_pipeline, filter_by_prefix, load_master_features
from src.features.preprocessing import build_preprocessor

PERSON = "B"
FEATURE_VERSION = "v1_bs_tl"
FEATURE_PREFIXES = ["bs_", "tl_"]
N_V1_FEATURES = 52

# Logarithmic grid: C scales the penalty multiplicatively (C=0.01 vs 0.1 is the same
# "step" as 1 vs 10), so equally spaced powers of 10 explore orders of magnitude evenly.
# A linear grid like [1, 2, ..., 10] would spend every point inside a single decade.
C_GRID = [0.001, 0.01, 0.1, 1, 10]
SVC_MAX_ITER = 10_000


def converged_scorer(estimator, X, y) -> float:
    """1.0 if the final classifier of a fitted Pipeline converged, else 0.0.

    A scorer receives the model already fitted on the training fold, so this reports
    convergence for each (C, fold) pair without extra fits, also inside joblib workers.
    Both liblinear and lbfgs stop at max_iter when they fail to converge.
    """
    clf = estimator[-1]
    return float(np.max(clf.n_iter_) < clf.max_iter)


SCORING = {"accuracy": "accuracy", "converged": converged_scorer}


def build_svc_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        # The scaler sits INSIDE the Pipeline: GridSearchCV refits the whole Pipeline on
        # each training fold, so means/stds never see the validation fold. Scaling the full
        # train set before the CV would leak validation statistics into training.
        ("prep", build_preprocessor()),
        # dual=False solves the primal problem. Its cost grows with n_features (52) rather
        # than n_samples (9,000 per fold), so with many more rows than columns it is the
        # faster and more reliably convergent formulation.
        ("clf", LinearSVC(C=C, dual=False, max_iter=SVC_MAX_ITER, random_state=RANDOM_STATE)),
    ])


def build_team_baseline() -> Pipeline:
    # The preprocessing is imported from the same place train_logistic.py uses; only the
    # LogisticRegression hyperparameters are copied, because they are hard-coded inside
    # run_training() (src/models/train_logistic.py, lines 67-79) and cannot be imported.
    return Pipeline([
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


def build_logistic_on_shared_prep() -> Pipeline:
    # Same LogisticRegression as the baseline, but with the new shared preprocessing:
    # versus LinearSVC, only the model changes.
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)),
    ])


def evaluate_with_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, splits) -> dict:
    scores = cross_validate(pipeline, X, y, cv=splits, scoring=SCORING, n_jobs=-1)
    return {
        "mean": scores["test_accuracy"].mean(),
        "std": scores["test_accuracy"].std(),  # ddof=0, as in the baseline scripts
        "non_converged": int(len(splits) - scores["test_converged"].sum()),
    }


def run_experiment(log: bool) -> None:
    print("=" * 70)
    print("Predex - LinearSVC on feature set v1 (person B)")
    print("=" * 70)

    X, y = load_master_features("train")
    X = filter_by_prefix(X, include_prefixes=FEATURE_PREFIXES)
    assert X.shape[1] == N_V1_FEATURES, f"Feature set v1 must have {N_V1_FEATURES} columns, got {X.shape[1]}"
    splits = get_cv_splits(X)
    n_folds = len(splits)
    print(f"Data: {X.shape[0]} battles x {X.shape[1]} features | {n_folds} folds from data/train_folds.csv")

    # ConvergenceWarnings are silenced so they cannot stop or flood the grid search;
    # convergence is reported explicitly through `converged_scorer` instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # 1. Grid search over C on the frozen folds
        grid = GridSearchCV(
            build_svc_pipeline(),
            param_grid={"clf__C": C_GRID},
            cv=splits,
            scoring=SCORING,
            refit=False,  # no final model needed in this step
            n_jobs=-1,
        )
        grid.fit(X, y)

        # 2. Logistic baselines on exactly the same folds
        team_baseline = evaluate_with_cv(build_team_baseline(), X, y, splits)
        lr_shared = evaluate_with_cv(build_logistic_on_shared_prep(), X, y, splits)

    res = grid.cv_results_
    mean_acc = res["mean_test_accuracy"]
    std_acc = res["std_test_accuracy"]
    non_converged = [
        int(n_folds - sum(res[f"split{i}_test_converged"][k] for i in range(n_folds)))
        for k in range(len(C_GRID))
    ]

    print("\n[1] LinearSVC grid search (accuracy over 10 folds)")
    print("-" * 70)
    print(f"{'C':>8} | {'Mean acc':>9} | {'Std':>7} | Non-converged folds")
    print("-" * 70)
    for k, C in enumerate(C_GRID):
        print(f"{C:>8g} | {mean_acc[k]:>9.4%} | {std_acc[k]:>7.4%} | {non_converged[k]}/{n_folds}")

    # np.argmax returns the first maximum: on a tie the smaller C (stronger regularization) wins.
    best_k = int(np.argmax(mean_acc))
    best_C = C_GRID[best_k]
    print("-" * 70)
    print(f"Best C = {best_C:g}: {mean_acc[best_k]:.4%} +/- {std_acc[best_k]:.4%}")
    # The best C is chosen on the same folds it is scored on, so its score is slightly
    # optimistic (selection bias). With only 5 candidates the effect is small.
    print("Note: the best-C score is slightly optimistic (C chosen on the same folds).")

    bad_Cs = [C for C, n in zip(C_GRID, non_converged) if n > 0]
    if bad_Cs:
        print(f"WARNING: C values with non-converged folds (max_iter={SVC_MAX_ITER}): {bad_Cs}")
    else:
        print("All C values converged on all folds.")

    if best_k in (0, len(C_GRID) - 1):
        side = "lower" if best_k == 0 else "upper"
        print(f"WARNING: best C is on the {side} edge of the grid -> extend the grid beyond {best_C:g}.")

    print("\n[2] Logistic baselines on the same folds")
    print("-" * 70)
    print(f"Team baseline (train_logistic.py config) : {team_baseline['mean']:.4%} +/- {team_baseline['std']:.4%}"
          f" | non-converged folds {team_baseline['non_converged']}/{n_folds}")
    print(f"LogReg C=1 on shared preprocessing       : {lr_shared['mean']:.4%} +/- {lr_shared['std']:.4%}"
          f" | non-converged folds {lr_shared['non_converged']}/{n_folds}")
    print(f"LinearSVC (C={best_C:g}) - LogReg shared prep  : {(mean_acc[best_k] - lr_shared['mean']) * 100:+.2f} pt")
    print(f"LinearSVC (C={best_C:g}) - team baseline       : {(mean_acc[best_k] - team_baseline['mean']) * 100:+.2f} pt")

    # 3. Out-of-fold predictions for the stacking. LinearSVC has no predict_proba: we save the
    # signed distance from the hyperplane (decision_function) instead. It is not a probability
    # (no log-loss, not directly averageable with other models' probabilities); calibration
    # (e.g. CalibratedClassifierCV) would be needed for that.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        oof_scores = cross_val_predict(
            build_svc_pipeline(C=best_C), X, y, cv=splits, method="decision_function", n_jobs=-1
        )
    oof_classes = (oof_scores > 0).astype(int)
    oof_df = pd.DataFrame({
        ID_COL: X.index,
        TARGET_COL: y.values,
        "decision_score": oof_scores,
        "pred_class": oof_classes,
    })
    oof_path = OUTPUTS_DIR / "oof_linear_svc.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\n[3] OOF accuracy (best C): {accuracy_score(y, oof_classes):.4%} -> saved {oof_path}")

    if not log:
        print("\n[--log not set] results/experiments.csv left untouched.")
        return

    def notes_for(n_bad: int) -> str:
        return f"{n_bad}/{n_folds} folds did not converge" if n_bad else ""

    log_result(
        model="LinearSVC",
        feature_version=FEATURE_VERSION,
        hyperparams={"C": best_C, "dual": False, "max_iter": SVC_MAX_ITER,
                     "prep": "build_preprocessor", "C_grid": C_GRID},
        cv_mean_acc=mean_acc[best_k],
        cv_std_acc=std_acc[best_k],
        person=PERSON,
        n_folds=n_folds,
        notes="; ".join(filter(None, ["best C from grid search on the same folds",
                                      notes_for(non_converged[best_k])])),
    )
    log_result(
        model="LogisticRegression (team baseline)",
        feature_version=FEATURE_VERSION,
        hyperparams={"C": 1.0, "solver": "lbfgs", "max_iter": 5000,
                     "prep": "create_linear_preprocessing_pipeline(drop_collinear=True, 0.95)"},
        cv_mean_acc=team_baseline["mean"],
        cv_std_acc=team_baseline["std"],
        person=PERSON,
        n_folds=n_folds,
        notes=notes_for(team_baseline["non_converged"]),
    )
    log_result(
        model="LogisticRegression",
        feature_version=FEATURE_VERSION,
        hyperparams={"C": 1.0, "solver": "lbfgs", "max_iter": 5000, "prep": "build_preprocessor"},
        cv_mean_acc=lr_shared["mean"],
        cv_std_acc=lr_shared["std"],
        person=PERSON,
        n_folds=n_folds,
        notes=notes_for(lr_shared["non_converged"]),
    )
    print("\n[--log] 3 rows appended to results/experiments.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinearSVC experiment on feature set v1")
    parser.add_argument("--log", action="store_true", help="Append the results to results/experiments.csv")
    args = parser.parse_args()
    run_experiment(log=args.log)
