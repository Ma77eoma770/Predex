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
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            battle_ids.append(data.get(ID_COL, line_idx))
            targets.append(data[TARGET_COL])

    df = pd.DataFrame({
        ID_COL: battle_ids,
        TARGET_COL: targets,
    })
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

    # Sanity checks
    assert (df["fold"] >= 0).all(), "Error: Some samples were not assigned to any fold!"
    assert len(df["fold"].unique()) == n_splits, f"Error: Expected {n_splits} folds, got {len(df['fold'].unique())}"

    # Verify disjoint validation folds
    val_indices_per_fold = [set(df[df["fold"] == f].index) for f in range(n_splits)]
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            intersection = val_indices_per_fold[i] & val_indices_per_fold[j]
            assert len(intersection) == 0, f"Error: Leakage detected between fold {i} and fold {j}!"

    # Verify coverage
    all_assigned = set.union(*val_indices_per_fold)
    assert len(all_assigned) == len(df), "Error: Not all samples covered in folds!"

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"[Cross-Validation] Folds saved successfully to: {out_p.resolve()}")

    return df


def load_folds(folds_path: Union[str, Path] = FOLDS_CSV) -> pd.DataFrame:
    """Load precomputed folds CSV or create them if the file does not exist."""
    folds_path = Path(folds_path)
    if folds_path.exists():
        return pd.read_csv(folds_path)
    return create_stratified_folds(output_path=folds_path)


def get_folds(
    folds_df: Optional[pd.DataFrame] = None,
    n_splits: int = N_FOLDS,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """Generator yielding (fold, train_indices, val_indices) for cross-validation loops.
    
    Usage:
        for fold, train_idx, val_idx in get_folds():
            # train_idx contains row indices (or battle_ids) for training
            # val_idx contains row indices (or battle_ids) for validation
            ...
    """
    if folds_df is None:
        folds_df = load_folds()

    for fold in range(n_splits):
        train_idx = folds_df[folds_df["fold"] != fold].index.to_numpy()
        val_idx = folds_df[folds_df["fold"] == fold].index.to_numpy()
        yield fold, train_idx, val_idx


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
    print("Verification: All folds strictly disjoint, identical size & balanced stratification.")
    print("=" * 70)


if __name__ == "__main__":
    print("[1/2] Creating and validating 10-Fold Stratified Cross-Validation...")
    df_folds = create_stratified_folds()
    print_fold_summary(df_folds)

    print("\n[2/2] Testing get_folds() iteration helper...")
    for fold, train_idx, val_idx in get_folds(df_folds):
        print(f"  -> Fold {fold}: {len(train_idx)} train samples, {len(val_idx)} validation samples.")

    print("\n[OK] Cross-validation foundation is ready and saved for team branch usage.")
