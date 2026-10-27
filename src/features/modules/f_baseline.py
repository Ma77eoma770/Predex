"""Baseline feature extractor module (initial 26 features).

Extracts team stats aggregations, lead stats, and initial t30 snapshots.
Prefix: bs_
"""

from typing import Dict, Any
import numpy as np

PREFIX = "bs_"


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract baseline features for a single battle."""
    feat: Dict[str, float] = {}

    # 1. Feature statiche: Team Player 1
    p1_team = data.get("p1_team_details", [])
    p1_stats = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
    for s in p1_stats:
        vals = [p.get(s, 0) for p in p1_team]
        feat[f"{PREFIX}p1_team_sum_{s}"] = float(np.sum(vals))
        feat[f"{PREFIX}p1_team_mean_{s}"] = float(np.mean(vals)) if vals else 0.0

    # 2. Feature statiche: Lead Player 2
    p2_lead = data.get("p2_lead_details", {})
    for s in p1_stats:
        feat[f"{PREFIX}p2_lead_{s}"] = float(p2_lead.get(s, 0.0))

    # 3. Feature dinamiche e di stato al turno 30 (dalla timeline)
    timeline = data.get("battle_timeline", [])

    p1_last_hp = 100.0
    p2_last_hp = 100.0
    p1_ko_count = 0
    p2_ko_count = 0

    if timeline:
        last_turn = timeline[-1]

        # Stato Pokémon attivo all'ultimo turno osservato
        p1_state = last_turn.get("p1_pokemon_state", {})
        p2_state = last_turn.get("p2_pokemon_state", {})
        p1_last_hp = float(p1_state.get("hp_pct", 100.0))
        p2_last_hp = float(p2_state.get("hp_pct", 100.0))

        for t in timeline:
            p1_t = t.get("p1_pokemon_state", {})
            p2_t = t.get("p2_pokemon_state", {})
            if p1_t.get("hp_pct", 100.0) == 0:
                p1_ko_count += 1
            if p2_t.get("hp_pct", 100.0) == 0:
                p2_ko_count += 1

    feat[f"{PREFIX}p1_active_hp_turn30"] = p1_last_hp
    feat[f"{PREFIX}p2_active_hp_turn30"] = p2_last_hp
    feat[f"{PREFIX}p1_ko_turn30"] = float(p1_ko_count)
    feat[f"{PREFIX}p2_ko_turn30"] = float(p2_ko_count)

    # 4. Differenze esplicite P1 - P2
    feat[f"{PREFIX}diff_active_hp"] = p1_last_hp - p2_last_hp
    feat[f"{PREFIX}diff_ko"] = float(p1_ko_count - p2_ko_count)
    feat[f"{PREFIX}diff_hp_stat"] = feat[f"{PREFIX}p1_team_mean_base_hp"] - feat[f"{PREFIX}p2_lead_base_hp"]
    feat[f"{PREFIX}diff_spe_stat"] = feat[f"{PREFIX}p1_team_mean_base_spe"] - feat[f"{PREFIX}p2_lead_base_spe"]

    return feat
