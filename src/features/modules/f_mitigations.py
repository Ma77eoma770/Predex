"""Mitigation and Advanced Sweeper / Wallbreaker Feature Extractor (Gen 2 Mechanics).

Implements deep endgame interactions designed specifically to resolve edge cases:
1. Max Attack vs Min Defense & Max SpA vs Min SpD (Wallbreaker & Damage Ratios).
2. Speed Monopoly & Uncontested Reverse Sweeper (The 'Swipe All' Feature).
3. Type Coverage Clearance (Alive sweepers with neutral/super-effective coverage on all remaining foes).
4. Unbroken Defensive Wall Lock (Snorlax/Suicune/Skarmory/Blissey resilience vs remaining offense).
5. Effective Status-Discounted Combat HP (Immobilization penalties for Freeze, Sleep, Par, Tox).
6. Information Asymmetry & Fog-of-War Expectations.
7. Timeline-Verified Setup Moves (Curse, Swords Dance, Belly Drum, Agility, Growth).

Prefix: mit_
"""

import json
from pathlib import Path
from typing import FrozenSet, Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.features.data_priors import bulky_species
from src.config import DATA_DIR, TYPE_CHART_CSV

PREFIX = "mit_"

POKEDEX_JSON_PATH = DATA_DIR / "pokedex.json"
_POKEDEX: Optional[Dict[str, Dict[str, Any]]] = None
_TYPE_CHART: Optional[Dict[Tuple[str, str], float]] = None

# Pulizia (persona D, 30/10): rimossa la classifica Smogon Gen 2 OU (dato esterno, vietato dal
# regolamento). La "viability" ora si calcola solo dalle statistiche base presenti nei dati (vedi _get_viability).
DEFAULT_OU_VIABILITY: float = 65.0
DEFAULT_UNLISTED_VIABILITY: float = 25.0

SETUP_MOVES: Set[str] = {
    "curse", "swordsdance", "bellydrum", "agility", "growth", "amnesia", "barrier"
}

STATUS_DISCOUNT: Dict[str, float] = {
    "nostatus": 1.0,
    "noeffect": 1.0,
    "fnt": 0.0,
    "slp": 0.55,
    "frz": 0.10,
    "par": 0.70,
    "tox": 0.75,
    "psn": 0.85,
    "brn": 0.85,
}

# Pulizia 04/11: prima era una lista scritta a mano; ora sono le specie con base_hp + base_def + base_spd >= 270
# nelle squadre di P1 del train (src/features/data_priors.py).
WALL_SPECIES: FrozenSet[str] = bulky_species()


def _load_pokedex() -> Dict[str, Dict[str, Any]]:
    global _POKEDEX
    if _POKEDEX is not None:
        return _POKEDEX
    if POKEDEX_JSON_PATH.exists():
        with open(POKEDEX_JSON_PATH, "r", encoding="utf-8") as f:
            _POKEDEX = json.load(f)
    else:
        _POKEDEX = {}
    return _POKEDEX


def _load_type_chart() -> Dict[Tuple[str, str], float]:
    global _TYPE_CHART
    if _TYPE_CHART is not None:
        return _TYPE_CHART
    chart: Dict[Tuple[str, str], float] = {}
    if TYPE_CHART_CSV.exists():
        df = pd.read_csv(TYPE_CHART_CSV)
        for _, row in df.iterrows():
            atk = str(row["attacking_type"]).strip().lower()
            dfn = str(row["defending_type"]).strip().lower()
            chart[(atk, dfn)] = float(row["multiplier"])
    _TYPE_CHART = chart
    return _TYPE_CHART


def _get_multiplier(atk_type: str, def_type: str) -> float:
    if not atk_type or not def_type or atk_type == "notype" or def_type == "notype":
        return 1.0
    chart = _load_type_chart()
    return chart.get((atk_type, def_type), 1.0)


def _defensive_multiplier(atk_type: str, def_types: List[str]) -> float:
    valid_def = [t for t in def_types if t and t != "notype"]
    if not valid_def:
        return 1.0
    mult = 1.0
    for d in valid_def:
        mult *= _get_multiplier(atk_type, d)
    return mult


def _get_stats(name: str) -> Dict[str, Any]:
    norm = str(name).strip().lower()
    dex = _load_pokedex()
    return dex.get(norm, {
        "types": ["normal"],
        "base_hp": 80.0,
        "base_atk": 80.0,
        "base_def": 80.0,
        "base_spa": 80.0,
        "base_spd": 80.0,
        "base_spe": 80.0,
    })


def _get_viability(name: str) -> float:
    norm = str(name).strip().lower()
    st = _get_stats(norm)
    bst = sum(st.get(f"base_{s}", 80.0) for s in ["hp", "atk", "def", "spa", "spd", "spe"])
    return float(min(80.0, max(20.0, (bst - 300.0) / 3.0)))


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    feat: Dict[str, float] = {}

    p1_team_details = data.get("p1_team_details", [])
    p1_team_names = [p.get("name", f"p1_{i}").strip().lower() for i, p in enumerate(p1_team_details)]
    p2_lead_name = data.get("p2_lead_details", {}).get("name", "p2_lead").strip().lower()

    timeline = data.get("battle_timeline", [])

    # Track HP, status, and observed moves per Pokémon up to turn 30
    p1_hp: Dict[str, float] = {name: 1.0 for name in p1_team_names}
    p1_status: Dict[str, str] = {name: "nostatus" for name in p1_team_names}
    p1_fainted: Set[str] = set()
    p1_moves: Set[str] = set()

    p2_hp: Dict[str, float] = {p2_lead_name: 1.0}
    p2_status: Dict[str, str] = {p2_lead_name: "nostatus"}
    p2_fainted: Set[str] = set()
    p2_seen_names: Set[str] = {p2_lead_name}
    p2_moves: Set[str] = set()

    p1_active_t30 = ""
    p2_active_t30 = ""
    p1_stab_bp_sum = 0.0
    p2_stab_bp_sum = 0.0

    for turn in timeline:
        p1_st = turn.get("p1_pokemon_state", {})
        p2_st = turn.get("p2_pokemon_state", {})

        p1_name = p1_st.get("name", "").strip().lower()
        p2_name = p2_st.get("name", "").strip().lower()

        if p1_name:
            p1_active_t30 = p1_name
            h = float(p1_st.get("hp_pct", 1.0))
            p1_hp[p1_name] = h
            st = p1_st.get("status", "nostatus")
            p1_status[p1_name] = st
            if h <= 0.001 or st == "fnt":
                p1_fainted.add(p1_name)
                p1_hp[p1_name] = 0.0

        if p2_name:
            p2_active_t30 = p2_name
            p2_seen_names.add(p2_name)
            h = float(p2_st.get("hp_pct", 1.0))
            p2_hp[p2_name] = h
            st = p2_st.get("status", "nostatus")
            p2_status[p2_name] = st
            if h <= 0.001 or st == "fnt":
                p2_fainted.add(p2_name)
                p2_hp[p2_name] = 0.0

        p1_mv = turn.get("p1_move_details")
        if p1_mv:
            mv_name = str(p1_mv.get("name", "")).strip().lower()
            if mv_name:
                p1_moves.add(mv_name)
            bp = float(p1_mv.get("base_power", 0) or 0)
            mv_t = str(p1_mv.get("type", "")).strip().lower()
            if bp > 0 and p1_name:
                user_types = _get_stats(p1_name).get("types", [])
                stab = 1.5 if mv_t in user_types else 1.0
                p1_stab_bp_sum += bp * stab

        p2_mv = turn.get("p2_move_details")
        if p2_mv:
            mv_name = str(p2_mv.get("name", "")).strip().lower()
            if mv_name:
                p2_moves.add(mv_name)
            bp = float(p2_mv.get("base_power", 0) or 0)
            mv_t = str(p2_mv.get("type", "")).strip().lower()
            if bp > 0 and p2_name:
                user_types = _get_stats(p2_name).get("types", [])
                stab = 1.5 if mv_t in user_types else 1.0
                p2_stab_bp_sum += bp * stab

    # Clean fainted
    for name in p1_fainted:
        p1_hp[name] = 0.0
        p1_status[name] = "fnt"
    for name in p2_fainted:
        p2_hp[name] = 0.0
        p2_status[name] = "fnt"

    p1_alive_names = [n for n in p1_team_names if n not in p1_fainted and p1_hp[n] > 0.001]
    # sorted: un set ha un ordine che cambia a ogni esecuzione (hash casuale), e il rollout dipende dall ordine
    p2_alive_names = [n for n in sorted(p2_seen_names) if n not in p2_fainted and p2_hp[n] > 0.001]

    # Pre-fetch stats of alive Pokémon
    p1_alive_stats = [_get_stats(n) for n in p1_alive_names]
    p2_alive_stats = [_get_stats(n) for n in p2_alive_names]

    # -------------------------------------------------------------------------
    # 1. Effective Combat HP (Status-Discounted) - Top 2 Feature in Forest
    # -------------------------------------------------------------------------
    p2_unrevealed = max(0, 6 - len(p2_seen_names))
    p1_eff_hp = sum(p1_hp[n] * STATUS_DISCOUNT.get(p1_status[n], 1.0) for n in p1_team_names)
    p2_eff_hp = sum(p2_hp[n] * STATUS_DISCOUNT.get(p2_status[n], 1.0) for n in p2_seen_names) + (p2_unrevealed * 1.0)

    feat[f"{PREFIX}p1_effective_combat_hp"] = float(p1_eff_hp)
    feat[f"{PREFIX}p2_effective_combat_hp"] = float(p2_eff_hp)
    feat[f"{PREFIX}diff_effective_combat_hp"] = float(p1_eff_hp - p2_eff_hp)

    # -------------------------------------------------------------------------
    # 2. Bench Threat & Viability Reserves
    # -------------------------------------------------------------------------
    p1_bench_alive = [n for n in p1_alive_names if n != p1_active_t30]
    p2_bench_alive = [n for n in p2_alive_names if n != p2_active_t30]

    p1_bench_viability = sum(_get_viability(n) for n in p1_bench_alive)
    p2_bench_viability = sum(_get_viability(n) for n in p2_bench_alive)
    feat[f"{PREFIX}p1_bench_alive_viability"] = float(p1_bench_viability)
    feat[f"{PREFIX}p2_bench_alive_viability"] = float(p2_bench_viability)
    feat[f"{PREFIX}diff_bench_alive_viability"] = float(p1_bench_viability - p2_bench_viability)

    # -------------------------------------------------------------------------
    # 3. Max Attack vs Min/Max Defense Matchups (Wallbreaker Indices)
    # -------------------------------------------------------------------------
    p1_atks = [s["base_atk"] for s in p1_alive_stats] or [80.0]
    p1_spas = [s["base_spa"] for s in p1_alive_stats] or [80.0]
    p1_defs = [s["base_def"] for s in p1_alive_stats] or [80.0]
    p1_spds = [s["base_spd"] for s in p1_alive_stats] or [80.0]

    p2_atks = [s["base_atk"] for s in p2_alive_stats] or [80.0]
    p2_spas = [s["base_spa"] for s in p2_alive_stats] or [80.0]
    p2_defs = [s["base_def"] for s in p2_alive_stats] or [80.0]
    p2_spds = [s["base_spd"] for s in p2_alive_stats] or [80.0]

    max_p1_atk, max_p1_spa = max(p1_atks), max(p1_spas)
    min_p2_def, min_p2_spd = min(p2_defs), min(p2_spds)
    max_p2_def, max_p2_spd = max(p2_defs), max(p2_spds)

    max_p2_atk, max_p2_spa = max(p2_atks), max(p2_spas)
    min_p1_def, min_p1_spd = min(p1_defs), min(p1_spds)
    max_p1_def, max_p1_spd = max(p1_defs), max(p1_spds)

    # Ratios representing ability to breakthrough opposing weak points
    p1_atk_breaker = max_p1_atk / (min_p2_def + 1.0)
    p1_spa_breaker = max_p1_spa / (min_p2_spd + 1.0)
    p2_atk_breaker = max_p2_atk / (min_p1_def + 1.0)
    p2_spa_breaker = max_p2_spa / (min_p1_spd + 1.0)

    feat[f"{PREFIX}p1_atk_vs_p2_min_def"] = float(p1_atk_breaker)
    feat[f"{PREFIX}p1_spa_vs_p2_min_spd"] = float(p1_spa_breaker)
    feat[f"{PREFIX}p2_atk_vs_p1_min_def"] = float(p2_atk_breaker)
    feat[f"{PREFIX}p2_spa_vs_p1_min_spd"] = float(p2_spa_breaker)
    feat[f"{PREFIX}diff_atk_breaker"] = float(p1_atk_breaker - p2_atk_breaker)
    feat[f"{PREFIX}diff_spa_breaker"] = float(p1_spa_breaker - p2_spa_breaker)

    # Hardest wall ratios
    p1_vs_max_wall = max_p1_atk / (max_p2_def + 1.0)
    p2_vs_max_wall = max_p2_atk / (max_p1_def + 1.0)
    feat[f"{PREFIX}diff_atk_vs_max_wall"] = float(p1_vs_max_wall - p2_vs_max_wall)

    # -------------------------------------------------------------------------
    # 4. Speed Monopoly & Uncontested Sweeper (The 'Swipe All' Feature)
    # -------------------------------------------------------------------------
    p1_speeds = [s["base_spe"] * (0.25 if p1_status.get(n) == "par" else 1.0) for n, s in zip(p1_alive_names, p1_alive_stats)] or [80.0]
    p2_speeds = [s["base_spe"] * (0.25 if p2_status.get(n) == "par" else 1.0) for n, s in zip(p2_alive_names, p2_alive_stats)] or [80.0]

    p1_top_spe = max(p1_speeds)
    p2_top_spe = max(p2_speeds)
    feat[f"{PREFIX}p1_top_alive_speed"] = float(p1_top_spe)
    feat[f"{PREFIX}p2_top_alive_speed"] = float(p2_top_spe)
    feat[f"{PREFIX}diff_top_alive_speed"] = float(p1_top_spe - p2_top_spe)

    # Does one side have an uncontested speed cleaner that outspeeds the entire enemy team by >= 5 points?
    p1_uncontested = 1.0 if (p1_top_spe >= p2_top_spe + 5.0 and (max_p1_atk >= 90.0 or max_p1_spa >= 90.0)) else 0.0
    p2_uncontested = 1.0 if (p2_top_spe >= p1_top_spe + 5.0 and (max_p2_atk >= 90.0 or max_p2_spa >= 90.0)) else 0.0
    feat[f"{PREFIX}p1_uncontested_speed_cleaner"] = p1_uncontested
    feat[f"{PREFIX}p2_uncontested_speed_cleaner"] = p2_uncontested
    feat[f"{PREFIX}diff_uncontested_speed_cleaner"] = float(p1_uncontested - p2_uncontested)

    # -------------------------------------------------------------------------
    # 5. Type Coverage Clearance & Unwallable Sweepers
    # -------------------------------------------------------------------------
    p2_def_type_lists = [s.get("types", ["normal"]) for s in p2_alive_stats]
    p1_def_type_lists = [s.get("types", ["normal"]) for s in p1_alive_stats]

    # Count how many alive P1 Pokémon hit ALL surviving P2 Pokémon with >= 1.0x (neutral) and at least one with >= 2.0x
    p1_unwallable = 0
    for s in p1_alive_stats:
        atk_types = s.get("types", ["normal"])
        hits_all_neutral = True
        hits_any_super = False
        for def_t in p2_def_type_lists:
            max_mult = max(_defensive_multiplier(a, def_t) for a in atk_types)
            if max_mult < 1.0:
                hits_all_neutral = False
            if max_mult >= 2.0:
                hits_any_super = True
        if hits_all_neutral and hits_any_super:
            p1_unwallable += 1

    p2_unwallable = 0
    for s in p2_alive_stats:
        atk_types = s.get("types", ["normal"])
        hits_all_neutral = True
        hits_any_super = False
        for def_t in p1_def_type_lists:
            max_mult = max(_defensive_multiplier(a, def_t) for a in atk_types)
            if max_mult < 1.0:
                hits_all_neutral = False
            if max_mult >= 2.0:
                hits_any_super = True
        if hits_all_neutral and hits_any_super:
            p2_unwallable += 1

    feat[f"{PREFIX}p1_unwallable_sweeper_count"] = float(p1_unwallable)
    feat[f"{PREFIX}p2_unwallable_sweeper_count"] = float(p2_unwallable)
    feat[f"{PREFIX}diff_unwallable_sweeper_count"] = float(p1_unwallable - p2_unwallable)

    # -------------------------------------------------------------------------
    # 6. Unbroken Defensive Wall Lock (Snorlax/Suicune/Skarmory etc.)
    # -------------------------------------------------------------------------
    p2_has_unbroken_wall = 0.0
    for n in p2_alive_names:
        if n in WALL_SPECIES and p2_hp[n] >= 0.70:
            # Check if any alive P1 has a super-effective STAB against this wall
            wall_types = _get_stats(n).get("types", ["normal"])
            has_se_answer = False
            for p1_s in p1_alive_stats:
                if any(_defensive_multiplier(a, wall_types) >= 2.0 for a in p1_s.get("types", ["normal"])):
                    has_se_answer = True
                    break
            if not has_se_answer:
                p2_has_unbroken_wall = 1.0
                break

    p1_has_unbroken_wall = 0.0
    for n in p1_alive_names:
        if n in WALL_SPECIES and p1_hp[n] >= 0.70:
            wall_types = _get_stats(n).get("types", ["normal"])
            has_se_answer = False
            for p2_s in p2_alive_stats:
                if any(_defensive_multiplier(a, wall_types) >= 2.0 for a in p2_s.get("types", ["normal"])):
                    has_se_answer = True
                    break
            if not has_se_answer:
                p1_has_unbroken_wall = 1.0
                break

    feat[f"{PREFIX}p1_has_unbroken_wall"] = p1_has_unbroken_wall
    feat[f"{PREFIX}p2_has_unbroken_wall"] = p2_has_unbroken_wall
    feat[f"{PREFIX}diff_unbroken_wall"] = float(p1_has_unbroken_wall - p2_has_unbroken_wall)

    # -------------------------------------------------------------------------
    # 7. Timeline-Verified Setup Moves
    # -------------------------------------------------------------------------
    p1_setup_count = sum(1 for m in p1_moves if m in SETUP_MOVES)
    p2_setup_count = sum(1 for m in p2_moves if m in SETUP_MOVES)
    feat[f"{PREFIX}p1_timeline_setup_moves"] = float(p1_setup_count)
    feat[f"{PREFIX}p2_timeline_setup_moves"] = float(p2_setup_count)
    feat[f"{PREFIX}diff_timeline_setup_moves"] = float(p1_setup_count - p2_setup_count)

    # -------------------------------------------------------------------------
    # 8. Fog of War & Reverse Sweep Candidate
    # -------------------------------------------------------------------------
    feat[f"{PREFIX}p2_unrevealed_expected_viability"] = float(p2_unrevealed * DEFAULT_OU_VIABILITY)
    feat[f"{PREFIX}high_fog_of_war_flag"] = 1.0 if p2_unrevealed >= 3 else 0.0

    p1_raw_hp = sum(p1_hp.values())
    p2_raw_hp = sum(p2_hp.values()) + (p2_unrevealed * 1.0)
    p1_hp_deficit = (p1_raw_hp < p2_raw_hp - 0.5)
    reversal_cand = 1.0 if (p1_hp_deficit and (p1_bench_viability > p2_bench_viability) and (p1_top_spe >= p2_top_spe)) else 0.0
    feat[f"{PREFIX}p1_reversal_candidate_flag"] = reversal_cand

    # -------------------------------------------------------------------------
    # 9. 4x Fatal Quad-Weakness Exploitation
    # -------------------------------------------------------------------------
    p1_exploitable_4x = 0
    for p2_s in p2_alive_stats:
        p2_def_types = p2_s.get("types", ["normal"])
        has_4x = False
        for p1_s in p1_alive_stats:
            for atk_t in p1_s.get("types", ["normal"]):
                if _defensive_multiplier(atk_t, p2_def_types) >= 3.99:
                    has_4x = True
                    break
            if has_4x:
                break
        if has_4x:
            p1_exploitable_4x += 1

    p2_exploitable_4x = 0
    for p1_s in p1_alive_stats:
        p1_def_types = p1_s.get("types", ["normal"])
        has_4x = False
        for p2_s in p2_alive_stats:
            for atk_t in p2_s.get("types", ["normal"]):
                if _defensive_multiplier(atk_t, p1_def_types) >= 3.99:
                    has_4x = True
                    break
            if has_4x:
                break
        if has_4x:
            p2_exploitable_4x += 1

    feat[f"{PREFIX}p1_exploitable_4x_count"] = float(p1_exploitable_4x)
    feat[f"{PREFIX}p2_exploitable_4x_count"] = float(p2_exploitable_4x)
    feat[f"{PREFIX}diff_exploitable_4x_count"] = float(p1_exploitable_4x - p2_exploitable_4x)

    # -------------------------------------------------------------------------
    # 10. Explicit STAB Offensive Power
    # -------------------------------------------------------------------------
    feat[f"{PREFIX}p1_total_stab_bp"] = float(p1_stab_bp_sum)
    feat[f"{PREFIX}p2_total_stab_bp"] = float(p2_stab_bp_sum)
    feat[f"{PREFIX}diff_total_stab_bp"] = float(p1_stab_bp_sum - p2_stab_bp_sum)

    # -------------------------------------------------------------------------
    # 11. Complete Defensive Resistances (<=0.5x) and Immunities (0.0x)
    # -------------------------------------------------------------------------
    p1_res = 0
    p1_imm = 0
    for p1_s in p1_alive_stats:
        p1_dt = p1_s.get("types", ["normal"])
        for p2_s in p2_alive_stats:
            for at in p2_s.get("types", ["normal"]):
                m = _defensive_multiplier(at, p1_dt)
                if m <= 0.001:
                    p1_imm += 1
                elif m <= 0.51:
                    p1_res += 1

    p2_res = 0
    p2_imm = 0
    for p2_s in p2_alive_stats:
        p2_dt = p2_s.get("types", ["normal"])
        for p1_s in p1_alive_stats:
            for at in p1_s.get("types", ["normal"]):
                m = _defensive_multiplier(at, p2_dt)
                if m <= 0.001:
                    p2_imm += 1
                elif m <= 0.51:
                    p2_res += 1

    feat[f"{PREFIX}p1_alive_resistances"] = float(p1_res)
    feat[f"{PREFIX}p2_alive_resistances"] = float(p2_res)
    feat[f"{PREFIX}diff_alive_resistances"] = float(p1_res - p2_res)
    feat[f"{PREFIX}p1_alive_immunities"] = float(p1_imm)
    feat[f"{PREFIX}p2_alive_immunities"] = float(p2_imm)
    feat[f"{PREFIX}diff_alive_immunities"] = float(p1_imm - p2_imm)

    # -------------------------------------------------------------------------
    # 12. Turn 30 Active STAB Efficiency
    # -------------------------------------------------------------------------
    p1_act_t = _get_stats(p1_active_t30).get("types", ["normal"]) if p1_active_t30 else ["normal"]
    p2_act_t = _get_stats(p2_active_t30).get("types", ["normal"]) if p2_active_t30 else ["normal"]

    p1_t30_stab = max([_defensive_multiplier(t, p2_act_t) * 1.5 for t in p1_act_t] or [1.0])
    p2_t30_stab = max([_defensive_multiplier(t, p1_act_t) * 1.5 for t in p2_act_t] or [1.0])

    feat[f"{PREFIX}p1_t30_stab_multiplier"] = float(p1_t30_stab)
    feat[f"{PREFIX}p2_t30_stab_multiplier"] = float(p2_t30_stab)
    feat[f"{PREFIX}diff_t30_stab_multiplier"] = float(p1_t30_stab - p2_t30_stab)

    return feat
