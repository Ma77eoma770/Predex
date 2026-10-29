"""Diagnose top 10 worst misclassified battles by Random Forest.

Finds the 10 battles where the model was most confident but completely wrong,
extracting timeline stats, HP remaining at turn 30, fainted counts, and team composition.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import OUTPUTS_DIR, TRAIN_JSONL


def diagnose_errors(top_n: int = 10):
    oof_path = OUTPUTS_DIR / "oof_random_forest.csv"
    if not oof_path.exists():
        print(f"Error: {oof_path} not found.")
        return

    df_oof = pd.read_csv(oof_path)
    # Calculate error magnitude
    df_oof["error_magnitude"] = np.abs(df_oof["player_won"] - df_oof["pred_prob"])
    worst_battles = df_oof.sort_values(by="error_magnitude", ascending=False).head(top_n)

    target_ids = set(worst_battles["battle_id"].values)

    # Load battle details from raw train.jsonl
    raw_battles = {}
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            bid = data["battle_id"]
            if bid in target_ids:
                raw_battles[bid] = data
                if len(raw_battles) == len(target_ids):
                    break

    reports = []
    for rank, row in enumerate(worst_battles.itertuples(), start=1):
        bid = int(row.battle_id)
        raw = raw_battles.get(bid, {})

        y_true = int(row.player_won)
        p_prob = float(row.pred_prob)

        # Extract context
        p1_team = [p.get("name") or p.get("species") for p in raw.get("p1_team_details", [])]
        p2_lead = raw.get("p2_lead_details", {}).get("name") or raw.get("p2_lead_details", {}).get("species")
        timeline = raw.get("battle_timeline", [])
        
        last_turn = timeline[-1] if timeline else {}
        turn_num = last_turn.get("turn", 0)
        p1_last_state = last_turn.get("p1_pokemon_state", {})
        p2_last_state = last_turn.get("p2_pokemon_state", {})

        # P2 seen pokemon in timeline
        p2_seen = set()
        for t in timeline:
            p2_mon = t.get("p2_pokemon_state", {}).get("name")
            if p2_mon:
                p2_seen.add(p2_mon)

        error_type = "False Positive (Predicted Win, Actually Lost)" if y_true == 0 else "False Negative (Predicted Loss, Actually Won)"

        reports.append({
            "rank": rank,
            "battle_id": bid,
            "error_type": error_type,
            "y_true": y_true,
            "pred_prob": p_prob,
            "error_magnitude": row.error_magnitude,
            "p1_team": p1_team,
            "p2_lead": p2_lead,
            "p2_seen_mons": list(p2_seen),
            "final_turn_observed": turn_num,
            "p1_active_t30": f"{p1_last_state.get('name')} (HP: {p1_last_state.get('hp_pct', 0)*100:.0f}%, {p1_last_state.get('status')})",
            "p2_active_t30": f"{p2_last_state.get('name')} (HP: {p2_last_state.get('hp_pct', 0)*100:.0f}%, {p2_last_state.get('status')})",
        })

    # Save to json and markdown
    out_json = OUTPUTS_DIR / "top_10_worst_rf_errors.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"\n[DIAGNOSTIC] Analyzed top {top_n} worst misclassified battles saved to {out_json}")
    return reports


if __name__ == "__main__":
    diagnose_errors()
