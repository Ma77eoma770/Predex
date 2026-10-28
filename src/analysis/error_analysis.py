"""Diagnostic & Error Analysis Tool for Predex (Gen 2 OU).

Identifies and analyzes matches where the model fails clamorously:
- High-confidence False Positives (predicted P1 win with > 75% prob, but P1 lost)
- High-confidence False Negatives (predicted P1 win with < 25% prob, but P1 won)
- Correlates errors with mechanical indicators (HP lead, Spikes, Status, Unrevealed Pokemon, Boosts)
- Inspects turn 30 endgame states of the worst predictions.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, RAW_DATA_DIR, TRAIN_FEATURES_PARQUET, TRAIN_JSONL


def analyze_model_errors(oof_path: Path = OUTPUTS_DIR / "oof_random_forest.csv", top_n: int = 10):
    print("=" * 75)
    print("Predex - Deep Error Analysis & Blindspot Diagnosis")
    print("=" * 75)

    if not oof_path.exists():
        print(f"Error: {oof_path} not found!")
        return

    oof = pd.read_csv(oof_path)
    X = pd.read_parquet(TRAIN_FEATURES_PARQUET)
    if "player_won" in X.columns:
        X = X.drop(columns=["player_won"])

    df = oof.copy()
    df["abs_error"] = (df["player_won"] - df["pred_prob"]).abs()
    df["log_loss"] = - (df["player_won"] * np.log(np.clip(df["pred_prob"], 1e-6, 1 - 1e-6)) +
                        (1 - df["player_won"]) * np.log(np.clip(1 - df["pred_prob"], 1e-6, 1 - 1e-6)))
    df["is_correct"] = (df["pred_class"] == df["player_won"]).astype(int)

    total_battles = len(df)
    total_acc = df["is_correct"].mean()
    total_errors = (df["is_correct"] == 0).sum()

    print(f"\nOverall Battles: {total_battles}")
    print(f"Overall Accuracy: {total_acc:.4%} ({df['is_correct'].sum()} correct / {total_errors} errors)")

    # Define clamorous failure segments
    fp_clamorous = df[(df["player_won"] == 0) & (df["pred_prob"] >= 0.70)].copy()
    fn_clamorous = df[(df["player_won"] == 1) & (df["pred_prob"] <= 0.30)].copy()

    fp_extreme = df[(df["player_won"] == 0) & (df["pred_prob"] >= 0.80)].copy()
    fn_extreme = df[(df["player_won"] == 1) & (df["pred_prob"] <= 0.20)].copy()

    print(f"\n[CLAMOROUS ERROR COUNTS]:")
    print(f"  False Positives (Predicted Win >= 70%, but LOST): {len(fp_clamorous)} battles ({len(fp_extreme)} extreme >= 80%)")
    print(f"  False Negatives (Predicted Win <= 30%, but WON):  {len(fn_clamorous)} battles ({len(fn_extreme)} extreme <= 20%)")
    total_clamorous = len(fp_clamorous) + len(fn_clamorous)
    print(f"  Total High-Confidence Blunders: {total_clamorous} ({total_clamorous / total_battles:.2%} of entire dataset!)")

    # Merge features to discover statistical signatures of failure
    full_df = df.join(X)

    # Key diagnostic metrics to compare across segments
    metrics_to_check = [
        "tl_diff_total_hp",
        "rf_hp_ratio",
        "tl_p1_total_hp",
        "tl_p2_total_hp",
        "rf_p2_unrevealed_hp_share",
        "rf_p2_revealed_count",
        "rf_p2_hidden_danger",
        "tl_diff_major_status",
        "rf_diff_active_asleep",
        "rf_diff_active_paralyzed",
        "rf_diff_spikes_control",
        "rf_diff_curse_count",
        "rf_diff_sweeper_index",
        "rf_active_clutch_ratio",
        "rf_diff_offensive_bp_total",
        "rf_diff_late_momentum",
    ]
    available_metrics = [m for m in metrics_to_check if m in full_df.columns]

    print("\n" + "=" * 75)
    print(f"{'Metric':<30} | {'Correct Win':<12} | {'Correct Loss':<12} | {'False Positive':<15} | {'False Negative':<15}")
    print("-" * 75)

    c_win = full_df[(full_df["player_won"] == 1) & (full_df["is_correct"] == 1)]
    c_loss = full_df[(full_df["player_won"] == 0) & (full_df["is_correct"] == 1)]

    for m in available_metrics:
        val_c_win = c_win[m].mean()
        val_c_loss = c_loss[m].mean()
        val_fp = fp_clamorous[m].mean() if m in fp_clamorous else full_df.loc[fp_clamorous.index, m].mean()
        val_fn = fn_clamorous[m].mean() if m in fn_clamorous else full_df.loc[fn_clamorous.index, m].mean()
        print(f"{m:<30} | {val_c_win:>11.3f} | {val_c_loss:>11.3f} | {val_fp:>14.3f} | {val_fn:>14.3f}")

    print("=" * 75)

    # Deep transcript inspection for the Top 3 Worst False Positives and Top 3 Worst False Negatives
    print("\n[INSPECTING TRANSCRIPTS OF THE WORST CLAMOROUS BLUNDERS]:")

    worst_fp = df[df["player_won"] == 0].sort_values(by="pred_prob", ascending=False).head(3)
    worst_fn = df[df["player_won"] == 1].sort_values(by="pred_prob", ascending=True).head(3)

    raw_battles = {}
    target_ids = set(worst_fp["battle_id"].tolist() + worst_fn["battle_id"].tolist())

    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            b_id = data.get("battle_id")
            if b_id in target_ids:
                raw_battles[b_id] = data

    print("\n>>> TOP 3 WORST FALSE POSITIVES (Model was positive P1 would win, but P1 LOST):")
    for _, row in worst_fp.iterrows():
        b_id = int(row["battle_id"])
        prob = row["pred_prob"]
        b_data = raw_battles.get(b_id, {})
        timeline = b_data.get("battle_timeline", [])
        last_t = timeline[-1] if timeline else {}
        p1_st = last_t.get("p1_pokemon_state", {})
        p2_st = last_t.get("p2_pokemon_state", {})
        print(f"\n  * Battle ID {b_id} — Predicted Prob: {prob:.4f} (Actual: 0, LOST)")
        print(f"    Turn 30 P1 Active: {p1_st.get('name')} (HP: {p1_st.get('hp_pct')}, Status: {p1_st.get('status')}, Boosts: {p1_st.get('boosts')})")
        print(f"    Turn 30 P2 Active: {p2_st.get('name')} (HP: {p2_st.get('hp_pct')}, Status: {p2_st.get('status')}, Boosts: {p2_st.get('boosts')})")
        # Check P2 team unrevealed
        seen_p2 = set(t.get("p2_pokemon_state", {}).get("name") for t in timeline if t.get("p2_pokemon_state", {}).get("name"))
        print(f"    P2 Distinct Pokemon Seen: {len(seen_p2)}/6 -> {list(seen_p2)}")

    print("\n>>> TOP 3 WORST FALSE NEGATIVES (Model thought P1 was dead, but P1 WON):")
    for _, row in worst_fn.iterrows():
        b_id = int(row["battle_id"])
        prob = row["pred_prob"]
        b_data = raw_battles.get(b_id, {})
        timeline = b_data.get("battle_timeline", [])
        last_t = timeline[-1] if timeline else {}
        p1_st = last_t.get("p1_pokemon_state", {})
        p2_st = last_t.get("p2_pokemon_state", {})
        print(f"\n  * Battle ID {b_id} — Predicted Prob: {prob:.4f} (Actual: 1, WON)")
        print(f"    Turn 30 P1 Active: {p1_st.get('name')} (HP: {p1_st.get('hp_pct')}, Status: {p1_st.get('status')}, Boosts: {p1_st.get('boosts')})")
        print(f"    Turn 30 P2 Active: {p2_st.get('name')} (HP: {p2_st.get('hp_pct')}, Status: {p2_st.get('status')}, Boosts: {p2_st.get('boosts')})")
        seen_p2 = set(t.get("p2_pokemon_state", {}).get("name") for t in timeline if t.get("p2_pokemon_state", {}).get("name"))
        print(f"    P2 Distinct Pokemon Seen: {len(seen_p2)}/6 -> {list(seen_p2)}")


if __name__ == "__main__":
    analyze_model_errors()
