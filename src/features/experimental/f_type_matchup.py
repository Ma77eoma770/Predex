"""Type matchup features from the Gen 2 type chart.

Three groups, each with its own sub-prefix so they can be evaluated separately:
- ty_off_: effectiveness of the damaging moves each side used over the 30 turns
- ty_def_: Player 1's team against the types Player 2 has shown (lead + revealed Pokémon)
- ty_act_: matchup of the two active Pokémon at the last observed turn

Types of Player 2's non-lead Pokémon come from data/species_types.csv (competition data only).
Unknown species are skipped. Every feature has a neutral default, so no NaN is ever produced.
Prefix: ty_
"""

from typing import Any, Dict, List, Optional, Tuple

from src.features.type_data import best_stab_multiplier, clean_types, multiplier, species_stat, species_types

PREFIX = "ty_"

# Hidden Power is always recorded as type "normal" with base power 0, although its real type
# varies: its effectiveness would be wrong (e.g. 0 against Ghost), so it is left out.
UNKNOWN_TYPE_MOVES = {"hiddenpower"}


def _move_stats(effs: List[float], stabs: List[bool]) -> Tuple[float, float, float, float]:
    """mean effectiveness, super-effective share, resisted share, STAB share (neutral if no moves)."""
    if not effs:
        return 1.0, 0.0, 0.0, 0.0
    n = len(effs)
    return (
        sum(effs) / n,
        sum(e > 1 for e in effs) / n,
        sum(e < 1 for e in effs) / n,  # includes immunities (0)
        sum(stabs) / n,
    )


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    team = data.get("p1_team_details", [])
    lead = data.get("p2_lead_details", {})
    timeline = data.get("battle_timeline", [])

    p1_types = {p["name"]: clean_types(p.get("types", [])) for p in team}
    p1_spe = {p["name"]: float(p.get("base_spe", 0)) for p in team}
    lead_name = lead.get("name")
    lead_types = clean_types(lead.get("types", []))

    def p2_types(name: Optional[str]) -> Optional[Tuple[str, ...]]:
        # The lead's types are in the JSON; the others come from the species table.
        return lead_types if name == lead_name else species_types(name)

    # ---- ty_off_: offensive matchup of the moves actually used ----
    # The move of turn t is matched with the opponent active in the state of turn t
    # (switches happen before moves in Pokémon; a mid-turn KO can occasionally break this).
    effs: Dict[str, List[float]] = {"p1": [], "p2": []}
    stabs: Dict[str, List[bool]] = {"p1": [], "p2": []}
    for turn in timeline:
        p1_name = turn.get("p1_pokemon_state", {}).get("name")
        p2_name = turn.get("p2_pokemon_state", {}).get("name")
        sides = (
            ("p1", turn.get("p1_move_details"), p1_types.get(p1_name), p2_types(p2_name)),
            ("p2", turn.get("p2_move_details"), p2_types(p2_name), p1_types.get(p1_name)),
        )
        for side, move, att_types, def_types in sides:
            # Skip: no move (switch), status moves (no damage), unknown move type, unknown species.
            if not move or move.get("category") == "status" or move.get("name") in UNKNOWN_TYPE_MOVES:
                continue
            if att_types is None or def_types is None:
                continue
            effs[side].append(multiplier(move["type"], def_types))
            # STAB: a move of the user's own type deals x1.5 damage.
            stabs[side].append(move["type"] in att_types)

    off = {side: _move_stats(effs[side], stabs[side]) for side in ("p1", "p2")}
    for side in ("p1", "p2"):
        mean_eff, se_share, resisted_share, stab_share = off[side]
        feat[f"{PREFIX}off_{side}_mean_eff"] = mean_eff
        feat[f"{PREFIX}off_{side}_se_share"] = se_share
        feat[f"{PREFIX}off_{side}_resisted_share"] = resisted_share
        feat[f"{PREFIX}off_{side}_stab_share"] = stab_share
    feat[f"{PREFIX}off_diff_mean_eff"] = off["p1"][0] - off["p2"][0]
    feat[f"{PREFIX}off_diff_se_share"] = off["p1"][1] - off["p2"][1]

    # ---- ty_def_: Player 1's team against what Player 2 has shown ----
    p2_known_names = [lead_name] + [t.get("p2_pokemon_state", {}).get("name") for t in timeline]
    p2_known: List[Tuple[str, ...]] = []
    for name in dict.fromkeys(p2_known_names):  # unique, in order of appearance
        types = p2_types(name)
        if types is not None:
            p2_known.append(types)

    def n_p1_weak_to(attacker_types: Tuple[str, ...]) -> int:
        # P1 Pokémon hit super-effectively by at least one of the attacker's own (STAB) types.
        return sum(best_stab_multiplier(attacker_types, types) > 1 for types in p1_types.values())

    def n_p1_hitting(defender_types: Tuple[str, ...]) -> int:
        # P1 Pokémon whose own types hit this defender super-effectively.
        return sum(best_stab_multiplier(types, defender_types) > 1 for types in p1_types.values())

    feat[f"{PREFIX}def_p1_weak_to_lead"] = float(n_p1_weak_to(lead_types))
    weak = [n_p1_weak_to(t) for t in p2_known]
    hit = [n_p1_hitting(t) for t in p2_known]
    feat[f"{PREFIX}def_p1_weak_to_revealed"] = sum(weak) / len(weak) if weak else 0.0
    feat[f"{PREFIX}def_p2_revealed_hit_by_p1"] = sum(hit) / len(hit) if hit else 0.0
    feat[f"{PREFIX}def_diff"] = feat[f"{PREFIX}def_p2_revealed_hit_by_p1"] - feat[f"{PREFIX}def_p1_weak_to_revealed"]

    # ---- ty_act_: the two active Pokémon at the last observed turn ----
    p1_act_types: Optional[Tuple[str, ...]] = None
    p2_act_types: Optional[Tuple[str, ...]] = None
    speed_diff = 0.0
    if timeline:
        last = timeline[-1]
        p1_act = last.get("p1_pokemon_state", {}).get("name")
        p2_act = last.get("p2_pokemon_state", {}).get("name")
        p1_act_types, p2_act_types = p1_types.get(p1_act), p2_types(p2_act)
        p2_spe = float(lead.get("base_spe", 0)) if p2_act == lead_name else species_stat(p2_act, "base_spe")
        if p1_act in p1_spe and p2_spe is not None:
            # Who moves first in the next exchange (base speed, ignoring boosts and paralysis).
            speed_diff = p1_spe[p1_act] - p2_spe

    known = p1_act_types is not None and p2_act_types is not None
    p1_best = best_stab_multiplier(p1_act_types, p2_act_types) if known else 1.0
    p2_best = best_stab_multiplier(p2_act_types, p1_act_types) if known else 1.0
    feat[f"{PREFIX}act_p1_best_eff"] = p1_best
    feat[f"{PREFIX}act_p2_best_eff"] = p2_best
    feat[f"{PREFIX}act_diff_best_eff"] = p1_best - p2_best
    feat[f"{PREFIX}act_speed_diff"] = speed_diff

    return feat
