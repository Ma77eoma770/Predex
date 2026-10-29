"""Shared evaluation helpers for Predex: frozen CV splits and the experiments log.

Every model script should take its splits from `get_cv_splits` (never a new KFold)
and record its score with `log_result`, so all results stay comparable.
"""

import csv
import json
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PROJECT_ROOT
from src.cross_validation import get_folds, load_folds

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "experiments.csv"
RESULTS_COLUMNS = [
    "date",
    "person",
    "model",
    "feature_version",
    "hyperparams",
    "cv_mean_acc",
    "cv_std_acc",
    "n_folds",
    "notes",
]


def get_cv_splits(X: pd.DataFrame, folds_df: Optional[pd.DataFrame] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the frozen folds of data/train_folds.csv as (train_positions, val_positions).

    Rows are matched by battle_id (X.index), so the split is correct regardless of the
    row order of X. The list can be passed directly as `cv=` to GridSearchCV,
    cross_validate and cross_val_predict.
    """
    return [(train_idx, val_idx) for _, train_idx, val_idx in get_folds(folds_df, ids=X.index)]


def paired_comparison(
    scores_new: np.ndarray,
    scores_ref: np.ndarray,
    n_train: int,
    n_test: int,
) -> Dict[str, Any]:
    """Compare two models fold by fold on the same splits.

    Both score arrays must come from the same folds, in the same order: a "hard" fold is
    hard for both models, and the per-fold difference cancels that shared noise.

    Returns the per-fold differences d (new - ref), their mean and std (ddof=0, as in the
    rest of the repo), wins/ties/losses of the new model, and the Nadeau-Bengio corrected
    paired t-test. A plain paired t-test treats the k folds as independent, but the training
    sets overlap (~89% with 10 folds), so the differences are correlated and their variance
    is underestimated; the correction inflates it by n_test/n_train:
        t = mean(d) / sqrt((1/k + n_test/n_train) * var(d, ddof=1))
    `p_value` is one-sided (H1: the new model is better), with k - 1 degrees of freedom.
    """
    scores_new = np.asarray(scores_new, dtype=float)
    scores_ref = np.asarray(scores_ref, dtype=float)
    if scores_new.shape != scores_ref.shape:
        raise ValueError(f"Score arrays differ in shape: {scores_new.shape} vs {scores_ref.shape}")

    d = scores_new - scores_ref
    k = len(d)
    ties = np.isclose(d, 0)
    var_d = d.var(ddof=1)
    if var_d > 0:
        t_stat = d.mean() / np.sqrt((1 / k + n_test / n_train) * var_d)
    else:
        # Identical differences on every fold: no spread to test against.
        t_stat = 0.0 if np.isclose(d.mean(), 0) else np.inf * np.sign(d.mean())

    return {
        "d": d,
        "mean_d": d.mean(),
        "std_d": d.std(),
        "wins": int(((d > 0) & ~ties).sum()),
        "ties": int(ties.sum()),
        "losses": int(((d < 0) & ~ties).sum()),
        "t_corrected": float(t_stat),
        "p_value": float(stats.t.sf(t_stat, df=k - 1)),
    }


def log_result(
    model: str,
    feature_version: str,
    hyperparams: Dict[str, Any],
    cv_mean_acc: float,
    cv_std_acc: float,
    person: str,
    n_folds: int,
    notes: str = "",
) -> None:
    """Append one experiment row to results/experiments.csv, creating it if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    row = {
        "date": date.today().isoformat(),
        "person": person,
        "model": model,
        "feature_version": feature_version,
        "hyperparams": json.dumps(hyperparams, sort_keys=True),
        "cv_mean_acc": f"{cv_mean_acc:.5f}",
        "cv_std_acc": f"{cv_std_acc:.5f}",
        "n_folds": n_folds,
        "notes": notes,
    }
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
