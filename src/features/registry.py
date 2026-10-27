"""Central Feature Registry & Master Parquet Store for Predex.

Orchestrates all modular feature extractors in `src/features/modules/`,
builds and caches master `.parquet` files for train and test sets,
and provides seamless, fast loading for all downstream models.
"""

import argparse
import importlib
import json
import pkgutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import (
    FEATURES_DIR,
    ID_COL,
    TARGET_COL,
    TEST_FEATURES_PARQUET,
    TEST_JSONL,
    TRAIN_FEATURES_PARQUET,
    TRAIN_JSONL,
)
import src.features.modules as modules_pkg

# Registry mapping: module_name -> extract_function
REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, float]]] = {}


def register_module(name: str, func: Callable[[Dict[str, Any]], Dict[str, float]]) -> None:
    """Register a feature extractor callable for a module name."""
    REGISTRY[name] = func


def auto_discover_modules() -> None:
    """Automatically discover all feature modules inside `src/features/modules/`."""
    package = modules_pkg
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg and modname.startswith("f_"):
            full_module_name = f"src.features.modules.{modname}"
            mod = importlib.import_module(full_module_name)
            if hasattr(mod, "extract_features"):
                clean_name = modname[2:] if modname.startswith("f_") else modname
                register_module(clean_name, getattr(mod, "extract_features"))


# Discover modules on import
auto_discover_modules()


def extract_all_features_for_battle(
    battle_data: Dict[str, Any],
    modules: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run all (or selected) registered feature extractors on a single battle."""
    feat: Dict[str, float] = {ID_COL: battle_data.get(ID_COL, -1)}

    active_modules = modules if modules is not None else list(REGISTRY.keys())
    for mod_name in active_modules:
        if mod_name in REGISTRY:
            mod_feat = REGISTRY[mod_name](battle_data)
            feat.update(mod_feat)

    return feat


def build_feature_dataframe(
    jsonl_path: Union[str, Path],
    is_train: bool = True,
    modules: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Parse JSONL battle records and extract features from all registered modules."""
    records: List[Dict[str, float]] = []
    targets: List[int] = []

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Source file not found at: {jsonl_path}")

    # Estimate line count for progress bar
    total_lines = 10000 if is_train else 5000

    with open(jsonl_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, total=total_lines, desc=f"Extracting {jsonl_path.name}") if show_progress else f
        for line in iterator:
            data = json.loads(line)
            records.append(extract_all_features_for_battle(data, modules=modules))
            if is_train:
                targets.append(int(data[TARGET_COL]))

    df = pd.DataFrame(records)
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)

    y = pd.Series(targets, index=df.index, name=TARGET_COL) if is_train else None
    return df, y


def build_master_parquet(force: bool = False, show_progress: bool = True) -> None:
    """Build and save master Parquet files for both train and test splits."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Predex Feature Store - Building Master Parquets")
    print(f"Registered modules ({len(REGISTRY)}): {list(REGISTRY.keys())}")
    print("=" * 65)

    # 1. Train split
    if force or not TRAIN_FEATURES_PARQUET.exists():
        t0 = time.time()
        print(f"\n[1/2] Processing Training data ({TRAIN_JSONL})...")
        X_train, y_train = build_feature_dataframe(TRAIN_JSONL, is_train=True, show_progress=show_progress)
        # Store target together in train parquet for convenience
        train_store = X_train.copy()
        train_store[TARGET_COL] = y_train
        train_store.to_parquet(TRAIN_FEATURES_PARQUET, engine="pyarrow", compression="snappy")
        dt = time.time() - t0
        print(f"      Saved {X_train.shape[1]} features for {len(X_train)} battles -> {TRAIN_FEATURES_PARQUET} ({dt:.1f}s)")
    else:
        print(f"\n[1/2] Train master parquet already exists at {TRAIN_FEATURES_PARQUET} (skip with force=False)")

    # 2. Test split
    if force or not TEST_FEATURES_PARQUET.exists():
        t0 = time.time()
        print(f"\n[2/2] Processing Test data ({TEST_JSONL})...")
        X_test, _ = build_feature_dataframe(TEST_JSONL, is_train=False, show_progress=show_progress)
        X_test.to_parquet(TEST_FEATURES_PARQUET, engine="pyarrow", compression="snappy")
        dt = time.time() - t0
        print(f"      Saved {X_test.shape[1]} features for {len(X_test)} battles -> {TEST_FEATURES_PARQUET} ({dt:.1f}s)")
    else:
        print(f"\n[2/2] Test master parquet already exists at {TEST_FEATURES_PARQUET} (skip with force=False)")

    print("\n[SUCCESS] Feature store up to date!\n")


def load_master_features(
    split: str = "train",
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load feature matrix and target from cached Master Parquet files.

    If files do not exist, automatically triggers the build process.
    """
    parquet_path = TRAIN_FEATURES_PARQUET if split == "train" else TEST_FEATURES_PARQUET

    if force_recompute or not parquet_path.exists():
        build_master_parquet(force=force_recompute)

    df = pd.read_parquet(parquet_path)

    if split == "train":
        if TARGET_COL in df.columns:
            y = df[TARGET_COL]
            X = df.drop(columns=[TARGET_COL])
        else:
            y = None
            X = df
        return X, y
    else:
        return df, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predex Feature Store Master Builder")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all feature parquets")
    args = parser.parse_args()

    build_master_parquet(force=args.force)
