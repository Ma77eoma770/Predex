"""Player 1 team composition features (static, known before the battle starts).

Complements f_baseline (which already has the sum/mean of each base stat) with extremes,
type diversity and defensive type profile of the six Pokémon.
Prefix: tm_
"""

from typing import Any, Dict

from src.features.type_data import BASE_STATS, all_types, clean_types, multiplier

PREFIX = "tm_"


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    team = data.get("p1_team_details", [])
    team_types = [clean_types(p.get("types", [])) for p in team]

    # 1. Stat extremes: the fastest and the hardest-hitting Pokémon often decide exchanges,
    #    the weakest one (e.g. an unevolved Pokémon) is a liability the mean hides.
    feat[f"{PREFIX}max_base_spe"] = float(max((p.get("base_spe", 0) for p in team), default=0))
    feat[f"{PREFIX}max_offense"] = float(max((max(p.get("base_atk", 0), p.get("base_spa", 0)) for p in team), default=0))
    feat[f"{PREFIX}min_bst"] = float(min((sum(p.get(s, 0) for s in BASE_STATS) for p in team), default=0))

    # 2. Type diversity: more distinct types -> broader coverage, fewer shared weaknesses.
    feat[f"{PREFIX}n_distinct_types"] = float(len({t for types in team_types for t in types}))

    # 3. Defensive profile against each of the 17 attacking types.
    weak_counts = [sum(multiplier(att, types) > 1 for types in team_types) for att in all_types()]
    # Largest number of team members weak to the SAME attacking type: one attack type threatens them all.
    feat[f"{PREFIX}max_shared_weakness"] = float(max(weak_counts, default=0))
    # Average number of attacking types each Pokémon resists or is immune to.
    resist_per_pokemon = [sum(multiplier(att, types) < 1 for att in all_types()) for types in team_types]
    feat[f"{PREFIX}mean_resistances"] = float(sum(resist_per_pokemon) / len(resist_per_pokemon)) if team else 0.0

    # 4. Real stats scale with level: below 100 a Pokémon is weaker than its base stats suggest.
    feat[f"{PREFIX}n_low_level"] = float(sum(p.get("level", 100) < 100 for p in team))

    return feat
