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


class RFImportanceSelector(BaseEstimator, TransformerMixin):
    """Prunes noisy features by selecting the top-K most important features via a fast Random Forest.

    Fitted strictly on training folds to eliminate data leakage.
    Ensures that during tree splits, sampling m = sqrt(p) picks from highly informative predictors.
    """

    def __init__(
        self,
        top_k: int = 45,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
    ):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.selected_columns_: List[str] = []
        self.selected_indices_: List[int] = []
        self.feature_names_in_: List[str] = []

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = X

        total_feats = X_arr.shape[1]
        if self.top_k is None or self.top_k <= 0 or self.top_k >= total_feats:
            self.selected_indices_ = list(range(total_feats))
            self.selected_columns_ = list(self.feature_names_in_)
            return self

        selector_rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        selector_rf.fit(X_arr, y)
        importances = selector_rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        self.selected_indices_ = sorted_idx[: self.top_k].tolist()
        self.selected_columns_ = [self.feature_names_in_[i] for i in self.selected_indices_]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if all(c in X.columns for c in self.selected_columns_):
                return X[self.selected_columns_]
            return X.iloc[:, self.selected_indices_]
        return pd.DataFrame(X[:, self.selected_indices_], columns=self.selected_columns_)


def create_rf_preprocessing_pipeline(
    variance_threshold: float = 0.0,
    top_k_features: Optional[int] = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-free preprocessing pipeline tailored for Random Forest and tree ensembles.

    Notice: No feature scaling (StandardScaler) is applied since decision trees are scale-invariant.
    Removes zero-variance (constant) features and applies feature pruning (top_k_features) strictly inside folds.
    """
    vt = VarianceThreshold(threshold=variance_threshold)
    vt.set_output(transform="pandas")
    steps = [
        ("variance_filter", vt),
    ]

    if top_k_features is not None and top_k_features > 0:
        steps.append((
            "importance_pruner",
            RFImportanceSelector(
                top_k=top_k_features,
                n_estimators=100,
                max_depth=12,
                random_state=random_state,
            ),
        ))

    return Pipeline(steps)



