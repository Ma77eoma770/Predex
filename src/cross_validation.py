"""Stratified 10-Fold Cross-Validation Framework for Predex.

This module sets up reproducible, deterministic 10-fold stratified cross-validation
splits across all development branches. It ensures that any model developed by any
team member evaluates on the exact same folds.
"""

import json
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

# Ensure project root is in sys.path so both direct and modular execution work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    FOLDS_CSV,
    ID_COL,
    N_FOLDS,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_JSONL,
)


def load_train_metadata(train_path: Union[str, Path] = TRAIN_JSONL) -> pd.DataFrame:
    """Extract battle_id and player_won targets from raw train.jsonl efficiently.
    
    Reads only necessary keys without keeping timeline/heavy payloads in memory.
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found at: {train_path}")

    battle_ids = []
    targets = []

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Fail loudly on a missing id: inventing one would silently break the
            # battle_id -> fold mapping used by every model.
            battle_ids.append(data[ID_COL])
            targets.append(int(data[TARGET_COL]))

    df = pd.DataFrame({
        ID_COL: battle_ids,
        TARGET_COL: targets,
    })
    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate {ID_COL} values found in {train_path}")
    return df


def create_stratified_folds(
    train_path: Union[str, Path] = TRAIN_JSONL,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
    output_path: Optional[Union[str, Path]] = FOLDS_CSV,
) -> pd.DataFrame:
    """Create deterministic Stratified K-Fold splits and optionally save to CSV.
    
    Args:
        train_path: Path to the raw train.jsonl file.
        n_splits: Number of cross-validation folds (default: 10).
        random_state: Seed for reproducibility (default: 42).
        output_path: Path to save the resulting folds mapping CSV.
        
    Returns:
        DataFrame containing [battle_id, player_won, fold].
    """
    df = load_train_metadata(train_path)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold

    # Each row has exactly one fold value, so validation folds are disjoint by
    # construction; what can go wrong is coverage, fold count and balance.
    _validate_folds(df, n_splits)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"[Cross-Validation] Folds saved successfully to: {out_p.resolve()}")

    return df


def _validate_folds(df: pd.DataFrame, n_splits: int) -> None:
    """Raise if the fold assignment is incomplete, unbalanced or has the wrong fold count."""
    missing_cols = {ID_COL, TARGET_COL, "fold"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Folds table is missing columns: {sorted(missing_cols)}")
    if df[ID_COL].duplicated().any():
        raise ValueError(f"Folds table contains duplicate {ID_COL} values")
    if (df["fold"] < 0).any():
        raise ValueError("Some samples were not assigned to any fold")
    if sorted(df["fold"].unique()) != list(range(n_splits)):
        raise ValueError(
            f"Expected folds 0..{n_splits - 1}, got {sorted(df['fold'].unique())}. "
            f"If N_FOLDS changed, regenerate with create_stratified_folds()."
        )

    sizes = df["fold"].value_counts()
    if sizes.max() - sizes.min() > 1:
        raise ValueError(f"Fold sizes are uneven: {sizes.to_dict()}")

    # Stratification: every fold's positive rate must stay close to the global one
    # (StratifiedKFold guarantees a difference of at most ~1 sample per class).
    global_rate = df[TARGET_COL].mean()
    fold_rates = df.groupby("fold")[TARGET_COL].mean()
    tolerance = 2.0 / sizes.min()
    if (fold_rates - global_rate).abs().max() > tolerance:
        raise ValueError(f"Folds are not stratified: positive rates {fold_rates.round(4).to_dict()}")


def load_folds(
    folds_path: Union[str, Path] = FOLDS_CSV,
    n_splits: int = N_FOLDS,
) -> pd.DataFrame:
    """Load the precomputed folds CSV, creating it only if the file does not exist.

    The committed CSV is the single source of truth for the team: it is validated
    on load instead of being silently regenerated, so everyone keeps the same folds.
    """
    folds_path = Path(folds_path)
    if not folds_path.exists():
        return create_stratified_folds(n_splits=n_splits, output_path=folds_path)

    folds_df = pd.read_csv(folds_path)
    _validate_folds(folds_df, n_splits)
    return folds_df


def get_folds(
    folds_df: Optional[pd.DataFrame] = None,
    ids: Optional[Union[pd.Index, np.ndarray, List[int]]] = None,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """Generator yielding (fold, train_positions, val_positions) for CV loops.

    The yielded arrays are 0-based row positions, meant for `.iloc` / numpy indexing.

    Args:
        folds_df: Optional preloaded folds DataFrame.
        ids: battle_id of each row of your feature matrix, in its row order
            (typically `X_train.index`). Positions are then computed by matching
            battle_id, so the split stays correct even if X_train was sorted,
            filtered or built in a different order than train.jsonl.
            Without it, X_train is ASSUMED to follow the CSV row order.

    Usage:
        for fold, train_idx, val_idx in get_folds(folds_df, ids=X_train.index):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ...
    """
    if folds_df is None:
        folds_df = load_folds()

    if ids is None:
        fold_per_row = folds_df["fold"].to_numpy()
    else:
        ids = np.asarray(ids)
        fold_by_id = folds_df.set_index(ID_COL)["fold"]
        if len(ids) != len(fold_by_id) or not np.isin(ids, fold_by_id.index).all():
            raise ValueError(
                f"The given ids ({len(ids)}) do not match the {ID_COL}s of the folds table "
                f"({len(fold_by_id)}). The feature matrix must contain exactly the training battles."
            )
        fold_per_row = fold_by_id.loc[ids].to_numpy()

    for fold in np.sort(np.unique(fold_per_row)):
        train_idx = np.flatnonzero(fold_per_row != fold)
        val_idx = np.flatnonzero(fold_per_row == fold)
        yield int(fold), train_idx, val_idx


def get_fold_battle_ids(
    fold: int,
    folds_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get train and validation battle_ids for a specific fold number.
    
    Args:
        fold: Fold index (0 to N_FOLDS - 1).
        folds_df: Optional preloaded folds DataFrame.
        
    Returns:
        (train_battle_ids, val_battle_ids) as numpy arrays.
    """
    if folds_df is None:
        folds_df = load_folds()

    if fold < 0 or fold >= folds_df["fold"].nunique():
        raise ValueError(f"Invalid fold index {fold}. Must be between 0 and {folds_df['fold'].nunique() - 1}")

    train_ids = folds_df[folds_df["fold"] != fold][ID_COL].to_numpy()
    val_ids = folds_df[folds_df["fold"] == fold][ID_COL].to_numpy()
    return train_ids, val_ids


def print_fold_summary(folds_df: pd.DataFrame) -> None:
    """Print a clean verification report of the stratified folds."""
    total_samples = len(folds_df)
    n_splits = folds_df["fold"].nunique()
    global_pos = (folds_df[TARGET_COL] == 1).sum()
    global_neg = (folds_df[TARGET_COL] == 0).sum()

    print("=" * 70)
    print(" " * 18 + "10-FOLD STRATIFIED CV SUMMARY")
    print("=" * 70)
    print(f"Total Battles   : {total_samples}")
    print(f"Total Folds     : {n_splits}")
    print(f"Global Balance  : Class 0 = {global_neg} ({global_neg / total_samples:.2%}) | Class 1 = {global_pos} ({global_pos / total_samples:.2%})")
    print("-" * 70)
    print(f"{'Fold':<6} | {'Train Size':<12} | {'Val Size':<10} | {'Val Class 0':<12} | {'Val Class 1':<12} | {'Val Pos Ratio':<12}")
    print("-" * 70)

    for fold in range(n_splits):
        val_subset = folds_df[folds_df["fold"] == fold]
        train_count = total_samples - len(val_subset)
        val_count = len(val_subset)
        val_c0 = (val_subset[TARGET_COL] == 0).sum()
        val_c1 = (val_subset[TARGET_COL] == 1).sum()
        pos_ratio = val_c1 / val_count if val_count > 0 else 0.0

        print(
            f"{fold:<6} | {train_count:<12} | {val_count:<10} | "
            f"{val_c0:<12} | {val_c1:<12} | {pos_ratio:<12.2%}"
        )
    print("=" * 70)
    # Report the outcome of the actual checks rather than a fixed message.
    try:
        _validate_folds(folds_df, n_splits)
        print("Verification: OK - full coverage, even fold sizes, stratified class balance.")
    except ValueError as err:
        print(f"Verification: FAILED - {err}")
    print("=" * 70)


if __name__ == "__main__":
    print("[1/2] Creating and validating 10-Fold Stratified Cross-Validation...")
    df_folds = create_stratified_folds()
    print_fold_summary(df_folds)

    print("\n[2/2] Testing get_folds() iteration helper...")
    for fold, train_idx, val_idx in get_folds(df_folds, ids=df_folds[ID_COL]):
        print(f"  -> Fold {fold}: {len(train_idx)} train samples, {len(val_idx)} validation samples.")

    print("\n[OK] Cross-validation foundation is ready and saved for team branch usage.")
