"""SVM with RBF kernel on feature set v1, paired against LinearSVC on the same folds.

Question this experiment answers: is the ~66.6% ceiling of the linear models due to the
features (hypothesis A) or to the linearity of the models (hypothesis B)? Only the model
changes: same features, same folds, same preprocessing as LinearSVC.

Usage:
    python src/models/train_svm_rbf.py          # print only
    python src/models/train_svm_rbf.py --yes    # run even if the time estimate exceeds 30 min
    python src/models/train_svm_rbf.py --log    # also append one row to results/experiments.csv
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import ID_COL, OUTPUTS_DIR, TARGET_COL
from src.evaluation import get_cv_splits, log_result, paired_comparison
from src.features import filter_by_prefix, load_master_features
from src.features.preprocessing import build_preprocessor
from src.models.train_linear_svc import (
    FEATURE_PREFIXES,
    FEATURE_VERSION,
    N_V1_FEATURES,
    PERSON,
    build_svc_pipeline,
)

C_GRID = [0.01, 0.1, 1, 10, 100]
# gamma sets how far the influence of one training battle reaches: the kernel between two
# battles is exp(-gamma * squared_distance). With 52 standardized features the typical
# squared distance is ~2*52 = 104, so:
#   gamma=0.001 -> exp(-0.1) = 0.90  every battle looks alike: almost a linear boundary
#   gamma=0.01  -> exp(-1.0) = 0.35  useful range
#   gamma=0.1   -> exp(-10)  = 3e-5  each battle only "sees" itself: memorization
# The kernel reacts exponentially to gamma, so half-decade steps (x ~3) are used to have
# more resolution than C's full decades. sklearn's default "scale" (~1/52 = 0.019) sits mid-grid.
GAMMA_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]

# Best C of the LinearSVC experiment (results/experiments.csv): the linear reference.
LINEAR_REF_C = 0.1

MAX_ESTIMATED_MINUTES = 30
# Verdict thresholds agreed with the team (see the plan).
MIN_GAIN_PT = 0.5
MIN_FOLD_WINS = 8  # sign test: >= 8/10 wins happens by chance ~5% of the time if models were equivalent


def build_rbf_pipeline(C: float = 1.0, gamma: float = 0.01) -> Pipeline:
    return Pipeline([
        # Standardization matters even more than for LinearSVC: the RBF kernel is built on
        # distances, so an unscaled column in the hundreds would decide every distance alone.
        ("prep", build_preprocessor()),
        # max_iter=-1 (default) means no iteration limit: libsvm always runs to convergence.
        # cache_size (MB) holds kernel values in memory, which speeds up fits on 9,000 rows.
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, cache_size=1000)),
    ])


def estimate_grid_minutes(X: pd.DataFrame, y: pd.Series, splits) -> float:
    """Time two worst-case fits on fold 0 and estimate the full grid duration (upper bound)."""
    train_idx, _ = splits[0]
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    # Two candidates for libsvm's slowest fit: large C with a very local kernel (many support
    # vectors), and large C with an almost linear kernel (many optimizer iterations).
    timings = {}
    for gamma in (max(GAMMA_GRID), min(GAMMA_GRID)):
        t0 = time.perf_counter()
        build_rbf_pipeline(C=max(C_GRID), gamma=gamma).fit(X_tr, y_tr)
        timings[gamma] = time.perf_counter() - t0
        print(f"  fit C={max(C_GRID):g}, gamma={gamma:g}: {timings[gamma]:.1f} s")

    n_fits = len(C_GRID) * len(GAMMA_GRID) * len(splits)
    n_cores = cpu_count()  # what n_jobs=-1 uses
    minutes = max(timings.values()) * n_fits / n_cores / 60
    print(f"  estimate: {max(timings.values()):.1f} s x {n_fits} fits / {n_cores} cores = {minutes:.1f} min (upper bound)")
    return minutes


def print_grid_table(title: str, values: pd.DataFrame, fmt: str) -> None:
    print(f"\n{title}")
    print("C \\ gamma".rjust(10) + " | " + " | ".join(f"{g:>8g}" for g in values.columns))
    print("-" * (13 + 11 * len(values.columns)))
    for C, row in values.iterrows():
        print(f"{C:>10g} | " + " | ".join(format(v, fmt).rjust(8) for v in row))


def paired_verdict(mean_d_pt: float, wins: int) -> str:
    # Order matters: the first true condition wins, so a negative gain never reads as "A".
    if mean_d_pt < 0:
        return "WORSE than linear: check the grid edges and the overfitting gap."
    if mean_d_pt < MIN_GAIN_PT:
        return f"HYPOTHESIS A: gain below {MIN_GAIN_PT} pt, the limit is the features."
    if wins < MIN_FOLD_WINS:
        return f"INCONCLUSIVE: gain >= {MIN_GAIN_PT} pt but RBF wins only {wins}/10 folds (driven by few folds)."
    return "HYPOTHESIS B: consistent gain, there is non-linear structure to exploit."


def run_experiment(log: bool, yes: bool) -> None:
    print("=" * 70)
    print("Predex - SVM RBF on feature set v1 (person B)")
    print("=" * 70)

    X, y = load_master_features("train")
    X = filter_by_prefix(X, include_prefixes=FEATURE_PREFIXES)
    assert X.shape[1] == N_V1_FEATURES, f"Feature set v1 must have {N_V1_FEATURES} columns, got {X.shape[1]}"
    splits = get_cv_splits(X)
    n_folds = len(splits)
    print(f"Data: {X.shape[0]} battles x {X.shape[1]} features | {n_folds} folds from data/train_folds.csv")

    # 1. Time estimate before committing to 250 fits
    print("\n[1] Time estimate (fold 0, worst cases)")
    minutes = estimate_grid_minutes(X, y, splits)
    if minutes > MAX_ESTIMATED_MINUTES and not yes:
        print(f"\nSTOP: estimate above {MAX_ESTIMATED_MINUTES} min. Re-run with --yes to proceed.")
        sys.exit(2)

    # 2. Grid search over C x gamma on the frozen folds
    print(f"\n[2] Grid search: {len(C_GRID)} C x {len(GAMMA_GRID)} gamma x {n_folds} folds ...")
    t0 = time.perf_counter()
    grid = GridSearchCV(
        build_rbf_pipeline(),
        param_grid={"clf__C": C_GRID, "clf__gamma": GAMMA_GRID},
        cv=splits,
        scoring="accuracy",
        # Training-fold accuracy too: the train - validation gap measures overfitting.
        # A model that memorizes scores ~100% on the battles it studied but not on new ones.
        return_train_score=True,
        refit=False,
        n_jobs=-1,
    )
    grid.fit(X, y)
    print(f"    done in {(time.perf_counter() - t0) / 60:.1f} min")

    res = pd.DataFrame(grid.cv_results_)
    res["C"] = res["param_clf__C"].astype(float)
    res["gamma"] = res["param_clf__gamma"].astype(float)
    res["gap"] = res["mean_train_score"] - res["mean_test_score"]
    # Tables are pivoted on the actual parameter values, not on the order of cv_results_.
    acc_table = res.pivot(index="C", columns="gamma", values="mean_test_score")
    gap_table = res.pivot(index="C", columns="gamma", values="gap")
    print_grid_table("Validation accuracy (mean over 10 folds)", acc_table * 100, ".2f")
    print_grid_table("Overfitting gap: train - validation accuracy (pt)", gap_table * 100, ".2f")

    best = res.loc[res["mean_test_score"].idxmax()]
    best_C, best_gamma = float(best["C"]), float(best["gamma"])
    print(f"\nBest: C={best_C:g}, gamma={best_gamma:g} -> {best['mean_test_score']:.4%} "
          f"+/- {best['std_test_score']:.4%} (gap {best['gap'] * 100:.2f} pt)")
    for name, grid_values, value in (("C", C_GRID, best_C), ("gamma", GAMMA_GRID, best_gamma)):
        if value in (min(grid_values), max(grid_values)):
            print(f"WARNING: best {name}={value:g} is on the edge of its grid -> extend the {name} grid.")

    # 3. Paired comparison with LinearSVC on exactly the same folds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        linear = cross_validate(build_svc_pipeline(C=LINEAR_REF_C), X, y, cv=splits, scoring="accuracy", n_jobs=-1)
    lin_scores = linear["test_score"]
    rbf_scores = np.array([best[f"split{i}_test_score"] for i in range(n_folds)])
    # Both models are scored on the same 1,000 battles per fold (see paired_comparison).
    paired = paired_comparison(rbf_scores, lin_scores, n_train=len(splits[0][0]), n_test=len(splits[0][1]))
    d, wins, losses = paired["d"], paired["wins"], paired["losses"]
    mean_d_pt, std_d_pt = paired["mean_d"] * 100, paired["std_d"] * 100

    print(f"\n[3] Paired comparison: RBF (C={best_C:g}, gamma={best_gamma:g}) vs LinearSVC (C={LINEAR_REF_C:g})")
    print("-" * 70)
    print(f"{'Fold':>4} | {'RBF':>8} | {'Linear':>8} | {'d (pt)':>7}")
    for i in range(n_folds):
        print(f"{i:>4} | {rbf_scores[i]:>8.2%} | {lin_scores[i]:>8.2%} | {d[i] * 100:>+7.2f}")
    print("-" * 70)
    print(f"LinearSVC mean: {lin_scores.mean():.4%} +/- {lin_scores.std():.4%}")
    print(f"Mean d: {mean_d_pt:+.2f} pt +/- {std_d_pt:.2f} | RBF wins {wins}, ties {paired['ties']}, loses {losses} (of {n_folds})")

    # Neighbours of the best cell: if they also beat the linear model, the gain is not a lucky pick
    # among 25 combinations (the best score is optimistic because it was chosen on these folds).
    ci, gi = C_GRID.index(best_C), GAMMA_GRID.index(best_gamma)
    print("Neighbours of the best cell (validation accuracy, minus linear):")
    for dc, dg in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= ci + dc < len(C_GRID) and 0 <= gi + dg < len(GAMMA_GRID):
            C_n, g_n = C_GRID[ci + dc], GAMMA_GRID[gi + dg]
            acc_n = acc_table.loc[C_n, g_n]
            print(f"  C={C_n:g}, gamma={g_n:g}: {acc_n:.4%} ({(acc_n - lin_scores.mean()) * 100:+.2f} pt)")

    verdict = paired_verdict(mean_d_pt, wins)
    print(f"\nVERDICT: {verdict}")
    print("Note: the best cell is chosen on the same folds, so its gain is slightly optimistic.")

    # 4. Full grid results for the report
    grid_path = OUTPUTS_DIR / "svm_rbf_grid.csv"
    res.drop(columns=["params"]).to_csv(grid_path, index=False)
    print(f"\n[4] Grid results -> {grid_path}")

    # 5. Out-of-fold decision scores of the best cell, for the stacking.
    # No probabilities: SVC(probability=True) would run its own internal 5-fold CV on every fit
    # (~5x slower, and folds that are not ours). Calibration is left to the stacking step.
    oof_scores = cross_val_predict(
        build_rbf_pipeline(C=best_C, gamma=best_gamma), X, y, cv=splits, method="decision_function", n_jobs=-1
    )
    oof_classes = (oof_scores > 0).astype(int)
    oof_path = OUTPUTS_DIR / "oof_svm_rbf.csv"
    pd.DataFrame({
        ID_COL: X.index,
        TARGET_COL: y.values,
        "decision_score": oof_scores,
        "pred_class": oof_classes,
    }).to_csv(oof_path, index=False)
    print(f"[5] OOF accuracy (best cell): {accuracy_score(y, oof_classes):.4%} -> saved {oof_path}")

    if not log:
        print("\n[--log not set] results/experiments.csv left untouched.")
        return

    log_result(
        model="SVC RBF",
        feature_version=FEATURE_VERSION,
        hyperparams={"C": best_C, "gamma": best_gamma, "kernel": "rbf", "prep": "build_preprocessor",
                     "C_grid": C_GRID, "gamma_grid": GAMMA_GRID},
        cv_mean_acc=float(best["mean_test_score"]),
        cv_std_acc=float(best["std_test_score"]),
        person=PERSON,
        n_folds=n_folds,
        notes=(f"best cell from grid search on the same folds; paired vs LinearSVC C={LINEAR_REF_C:g}: "
               f"d={mean_d_pt:+.2f}+/-{std_d_pt:.2f} pt, wins {wins}/{n_folds}"),
    )
    print("\n[--log] 1 row appended to results/experiments.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM RBF experiment on feature set v1")
    parser.add_argument("--log", action="store_true", help="Append the result to results/experiments.csv")
    parser.add_argument("--yes", action="store_true", help=f"Run even if the estimate exceeds {MAX_ESTIMATED_MINUTES} min")
    args = parser.parse_args()
    run_experiment(log=args.log, yes=args.yes)
