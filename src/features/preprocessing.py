"""Shared, leakage-free preprocessing for scale-sensitive models (LinearSVC, SVM RBF, kNN).

`build_preprocessor()` returns an unfitted transformer meant to be the FIRST step of a
Pipeline. Because it lives inside the Pipeline, cross-validation refits it on the training
part of every fold: imputation medians, scaler means/stds, one-hot categories and PCA
components are never computed on the validation fold.
"""

from typing import Optional, Sequence, Union

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def build_preprocessor(
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    pca_components: Optional[Union[int, float]] = None,
    random_state: int = RANDOM_STATE,
) -> Union[ColumnTransformer, Pipeline]:
    """Build the numeric/categorical preprocessing, optionally followed by PCA.

    Args:
        numeric_cols: Numeric column names. Default: every column with a numeric dtype.
        categorical_cols: Categorical column names. Default: every non-numeric column.
        pca_components: None (default) disables PCA. An int keeps that many components,
            a float in (0, 1) keeps enough components to explain that fraction of variance.
        random_state: Seed for PCA's randomized solver (deterministic results).

    Returns:
        A ColumnTransformer, or Pipeline(ColumnTransformer -> PCA) when PCA is enabled.
        PCA cannot sit inside the ColumnTransformer: it must see the concatenated
        output of both branches, so it is chained after it.
    """
    # Column selection only looks at dtypes, never at values, so resolving it at fit
    # time on the training fold leaks nothing. An empty selection (e.g. no categorical
    # columns, as in feature set v1) is simply skipped by the ColumnTransformer.
    num_selector = list(numeric_cols) if numeric_cols is not None else make_column_selector(dtype_include=np.number)
    cat_selector = list(categorical_cols) if categorical_cols is not None else make_column_selector(dtype_exclude=np.number)

    numeric_pipe = Pipeline([
        # Median is robust to the skewed counts/HP sums we have; v1 has no NaN today,
        # but test battles or future feature modules might.
        ("imputer", SimpleImputer(strategy="median")),
        # Linear models with L2 penalty, RBF kernels and kNN distances all depend on
        # feature scale: unscaled, a stat sum in the hundreds would dominate a 0-6 KO count.
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a species/item never seen in the training fold
        # becomes an all-zero row instead of crashing at predict time.
        # Dense output so the optional PCA step can consume it.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_selector),
            ("cat", categorical_pipe, cat_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if pca_components is None:
        return column_transformer

    return Pipeline([
        ("columns", column_transformer),
        ("pca", PCA(n_components=pca_components, random_state=random_state)),
    ])
