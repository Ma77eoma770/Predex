"""Feature extraction pipeline for Predex."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET_COL, TEST_JSONL, TRAIN_JSONL


def extract_features_from_battle(data: dict) -> Dict[str, float]:
    """Extract a tabular numerical feature dictionary from a single raw battle JSON."""
    feat: Dict[str, float] = {}
    feat[ID_COL] = data.get(ID_COL, -1)

    # 1. Feature statiche: Team Player 1
    p1_team = data.get("p1_team_details", [])
    p1_stats = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
    for s in p1_stats:
        vals = [p.get(s, 0) for p in p1_team]
        feat[f"p1_team_sum_{s}"] = float(np.sum(vals))
        feat[f"p1_team_mean_{s}"] = float(np.mean(vals)) if vals else 0.0

    # 2. Feature statiche: Lead Player 2
    p2_lead = data.get("p2_lead_details", {})
    for s in p1_stats:
        feat[f"p2_lead_{s}"] = float(p2_lead.get(s, 0.0))

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

        # Conteggio KO lungo la timeline osservata
        for t in timeline:
            p1_t = t.get("p1_pokemon_state", {})
            p2_t = t.get("p2_pokemon_state", {})
            if p1_t.get("hp_pct", 100.0) == 0:
                p1_ko_count += 1
            if p2_t.get("hp_pct", 100.0) == 0:
                p2_ko_count += 1

    feat["p1_active_hp_turn30"] = p1_last_hp
    feat["p2_active_hp_turn30"] = p2_last_hp
    feat["p1_ko_turn30"] = float(p1_ko_count)
    feat["p2_ko_turn30"] = float(p2_ko_count)

    # 4. Differenze esplicite P1 - P2 (essenziali per modelli lineari)
    feat["diff_active_hp"] = p1_last_hp - p2_last_hp
    feat["diff_ko"] = float(p1_ko_count - p2_ko_count)
    feat["diff_hp_stat"] = feat["p1_team_mean_base_hp"] - feat["p2_lead_base_hp"]
    feat["diff_spe_stat"] = feat["p1_team_mean_base_spe"] - feat["p2_lead_base_spe"]

    return feat


def build_dataset(
        jsonl_path: Union[str, Path], is_train: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Stream JSONL file and return clean Feature DataFrame X and Target Series y."""
    records: List[Dict[str, float]] = []
    targets: List[int] = []

    jsonl_path = Path(jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            records.append(extract_features_from_battle(data))
            if is_train:
                targets.append(int(data[TARGET_COL]))

    df_features = pd.DataFrame(records)

    # Isola la colonna ID in modo che non venga usata come feature numerica
    df_features.set_index(ID_COL, inplace=True)

    y = pd.Series(targets, index=df_features.index, name=TARGET_COL) if is_train else None
    return df_features, y