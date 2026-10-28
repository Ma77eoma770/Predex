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
