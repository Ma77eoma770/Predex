"""Feature selection and automatic pruning utilities for Predex models.

Provides scikit-learn compatible transformers that automatically remove:
- Zero / low-variance features
- Highly collinear (redundant) features
- Low-importance features (SelectKBest, L1, etc.)
strictly inside cross-validation splits to prevent any data leakage.
"""

from typing import List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DropCollinearFeatures(BaseEstimator, TransformerMixin):
    """Drops one feature from pairs with absolute Pearson correlation above threshold.

    Fits on training folds only to prevent leakage, then applies the same drops on validation.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.to_drop_: List[str] = []
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.to_drop_ = to_drop
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.drop(columns=[c for c in self.to_drop_ if c in X.columns])


def filter_by_prefix(
    df: pd.DataFrame,
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_prefixes: Optional[Sequence[str]] = None,
    drop_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convenience helper to slice columns by naming prefix or specific blacklist."""
    cols = list(df.columns)

    if include_prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in include_prefixes)]

    if exclude_prefixes:
        cols = [c for c in cols if not any(c.startswith(p) for p in exclude_prefixes)]

    if drop_columns:
        drop_set = set(drop_columns)
        cols = [c for c in cols if c not in drop_set]

    return df[cols]


def create_linear_preprocessing_pipeline(
    drop_collinear: bool = True,
    collinear_threshold: float = 0.95,
    k_best: Optional[int] = None,
) -> Pipeline:
    """Build a standard leakage-free preprocessing pipeline for linear/distance models."""
    steps = [
        ("variance_filter", VarianceThreshold(threshold=0.001)),
    ]

    if drop_collinear:
        steps.append(("collinear_filter", DropCollinearFeatures(threshold=collinear_threshold)))

    steps.append(("scaler", StandardScaler()))

    if k_best is not None and k_best > 0:
        steps.append(("k_best", SelectKBest(score_func=f_classif, k=k_best)))

    return Pipeline(steps)
