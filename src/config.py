"""Project configuration and path definitions for Predex.

Shared settings across all team branches to guarantee reproducibility.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Raw Data Files
TRAIN_JSONL = RAW_DATA_DIR / "train.jsonl"
TEST_JSONL = RAW_DATA_DIR / "test.jsonl"
TYPE_CHART_CSV = RAW_DATA_DIR / "type_chart_gen2.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

# Cross-Validation & Processed Data
FOLDS_CSV = DATA_DIR / "train_folds.csv"

# Global Cross-Validation & Experiment Parameters
RANDOM_STATE = 42
N_FOLDS = 10
TARGET_COL = "player_won"
ID_COL = "battle_id"

# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
