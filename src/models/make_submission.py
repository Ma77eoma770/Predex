"""Final retrain on all training battles and Kaggle submission file.

No cross-validation here: model and hyperparameters were already chosen with the 10 frozen
folds (see results/experiments.csv). In CV every model saw only 90% of the battles; the final
model is refit on all 10,000 so it learns from as much data as possible.

Usage:
    python src/models/make_submission.py                  # default: --model linear_svc
    python src/models/make_submission.py --model linear_svc
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ID_COL, OUTPUTS_DIR, SAMPLE_SUBMISSION_CSV, TARGET_COL
from src.features import filter_by_prefix, load_master_features
from src.models.train_linear_svc import FEATURE_PREFIXES, N_V1_FEATURES, build_svc_pipeline

# Model name -> builder of the UNFITTED pipeline. Adding a model is one line here.
MODELS: Dict[str, Callable[[], Pipeline]] = {
    # Best C of the LinearSVC grid search (results/experiments.csv: 66.60% on v1).
    "linear_svc": lambda: build_svc_pipeline(C=0.1),
}

SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
EXPECTED_COLUMNS = [ID_COL, TARGET_COL]


def load_v1(split: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Feature set v1 (bs_ + tl_) for one split, indexed by battle_id.

    load_master_features rebuilds a missing parquet by itself (only the missing split),
    with the same registry code for train and test.
    """
    X, y = load_master_features(split)
    X = filter_by_prefix(X, include_prefixes=FEATURE_PREFIXES)
    assert X.shape[1] == N_V1_FEATURES, f"{split}: v1 must have {N_V1_FEATURES} columns, got {X.shape[1]}"
    return X, y


def build_submission(test_ids: pd.Index, predictions: np.ndarray) -> pd.DataFrame:
    """Fill sample_submission.csv with the predictions (same logic as the starter notebook)."""
    template = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    if list(template.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected submission columns {EXPECTED_COLUMNS}, got {list(template.columns)}")
    if test_ids.duplicated().any():
        # With duplicated ids the mapping below would be ambiguous.
        raise ValueError(f"Duplicated {ID_COL} values in the test features")

    # Match predictions by battle_id, NOT by position: the template order is what Kaggle
    # expects, and nothing guarantees the feature matrix has the same row order.
    predictions_by_id = pd.Series(predictions, index=test_ids)
    submission = template.copy()
    submission[TARGET_COL] = submission[ID_COL].map(predictions_by_id)
    if submission[TARGET_COL].isna().any():
        missing = submission.loc[submission[TARGET_COL].isna(), ID_COL].tolist()
        raise ValueError(f"{len(missing)} sample submission {ID_COL} values have no prediction, e.g. {missing[:5]}")
    submission[TARGET_COL] = submission[TARGET_COL].astype(int)
    return submission


def run(model_name: str) -> None:
    print("=" * 70)
    print(f"Predex - final retrain and submission ({model_name})")
    print("=" * 70)

    X_train, y_train = load_v1("train")
    X_test, _ = load_v1("test")
    # The pipeline works on column positions after preprocessing: train and test must match exactly.
    assert list(X_train.columns) == list(X_test.columns), "Train and test feature columns differ"
    print(f"Train: {X_train.shape[0]} battles x {X_train.shape[1]} features | Test: {X_test.shape[0]} battles")

    # Final retrain on ALL training battles. ConvergenceWarnings are intentionally not
    # silenced: on the model we submit, a warning must be visible.
    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    finite = np.isfinite(X_test.to_numpy(dtype=float))
    bad_cols = X_test.columns[~finite.all(axis=0)].tolist()
    assert not bad_cols, f"X_test contains NaN or infinite values in columns: {bad_cols}"
    predictions = model.predict(X_test)

    submission = build_submission(X_test.index, predictions)
    template_rows = len(pd.read_csv(SAMPLE_SUBMISSION_CSV))
    assert len(submission) == template_rows, f"Submission has {len(submission)} rows, template has {template_rows}"

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / f"submission_{model_name}.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nRows: {len(submission)} (expected {template_rows})")
    print(f"Share of predicted 1 (P1 wins): {submission[TARGET_COL].mean():.2%} | in train: {y_train.mean():.2%}")
    print("\nFirst rows:")
    print(submission.head().to_string(index=False))
    print(f"\n[SUCCESS] Submission saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final retrain on all training battles and submission file")
    parser.add_argument("--model", choices=list(MODELS), default="linear_svc", help="Model to train and submit")
    args = parser.parse_args()
    run(args.model)
