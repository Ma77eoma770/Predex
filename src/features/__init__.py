"""Features module public API for Predex."""

from src.features.registry import (
    REGISTRY,
    extract_all_features_for_battle,
    build_master_parquet,
    load_master_features,
    register_module,
)
from src.features.selectors import (
    DropCollinearFeatures,
    filter_by_prefix,
    create_linear_preprocessing_pipeline,
)
from src.features.modules.f_baseline import extract_features as _extract_baseline
from src.features.registry import build_feature_dataframe


def extract_features_from_battle(data: dict):
    """Legacy extractor matching the initial 26 baseline features."""
    from src.config import ID_COL
    feat = _extract_baseline(data)
    feat[ID_COL] = data.get(ID_COL, -1)
    legacy_feat = {ID_COL: feat[ID_COL]}
    for k, v in feat.items():
        if k != ID_COL:
            legacy_key = k[3:] if k.startswith("bs_") else k
            legacy_feat[legacy_key] = v
    return legacy_feat


def build_dataset(jsonl_path, is_train: bool = True):
    """Legacy loader returning (df_features, y) for baseline features."""
    return build_feature_dataframe(jsonl_path, is_train=is_train, modules=["baseline"], show_progress=False)


__all__ = [
    "REGISTRY",
    "extract_all_features_for_battle",
    "build_master_parquet",
    "load_master_features",
    "register_module",
    "DropCollinearFeatures",
    "filter_by_prefix",
    "create_linear_preprocessing_pipeline",
    "extract_features_from_battle",
    "build_dataset",
]
