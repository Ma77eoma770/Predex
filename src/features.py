"""Feature extraction pipeline for Predex (backward-compatibility bridge).

Delegates to `src.features.modules.f_baseline` and `src.features.registry`.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from src.features.modules.f_baseline import extract_features as _extract_baseline
from src.features.registry import build_feature_dataframe, load_master_features
from src.config import ID_COL


def extract_features_from_battle(data: dict) -> Dict[str, float]:
    """Legacy extractor matching the initial 26 baseline features."""
    feat = _extract_baseline(data)
    feat[ID_COL] = data.get(ID_COL, -1)
    # Remove prefix for strict legacy compatibility if needed, or keep
    # The keys in f_baseline start with bs_
    legacy_feat = {ID_COL: feat[ID_COL]}
    for k, v in feat.items():
        if k != ID_COL:
            legacy_key = k[3:] if k.startswith("bs_") else k
            legacy_feat[legacy_key] = v
    return legacy_feat


def build_dataset(
    jsonl_path: Union[str, Path], is_train: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Legacy loader function returning (df_features, y)."""
    return build_feature_dataframe(jsonl_path, is_train=is_train, modules=["baseline"])