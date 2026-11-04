"""Random Forest feature extractor module (Gen 2 OU Mechanics-Driven).

Focuses exclusively on deep abstract competitive mechanics:
1. Complete 17-Type Defensive & Offensive Interaction Matrices (Type Chart Gen 2).
2. Item Valuation (Leftovers recovery turns, status berries, offensive enhancers).
3. Cumulative Team & Alive Base Stats (HP, Atk, Def, SpA, SpD, Spe, BST) for P1 & P2 via Pokedex lookup.
4. Physical vs Special Macro & Micro Split:
   - Physical Bulk (HP x Def / 100) vs Special Bulk (HP x SpD / 100) across alive team.
   - Offensive capability (Base Atk vs Base SpA) and direct macro matchups.
5. Exact Turn 30 State (Micro-Tactical Matchup):
   - Exact P1 & P2 active types, base stats, stage boosts, paralysis/burn stat modifications.
   - Exact speed order and damage exchange ratios at turn 30.
   - Active effects: Substitute, Leech Seed, Curse, Confusion, Trapped, Perish Song.
   - Active statuses: Freeze, Toxic, Burn, Sleep, Paralysis.
6. Multi-Bracket Timeline Momentum (T0-T10, T10-T20, T20-T30):
   - HP deltas, slope differentials, damage acceleration.
   - Multi-phase switching dynamics.
7. ~50 Deeply Engineered Cross-Interaction Features:
   - Sweeper threat indices, defensive wall indices.
   - Spikes residual damage potential on grounded Pokémon.
   - Leftovers hegemony index.
   - Offensive dominance indices (Physical & Special).
   - High-impact move counters (Toxic, T-Wave, Sleep, Recovery, Explosion, Phazing, Belly Drum).
   - Endgame pressure and clutch ratios.

No hardcoded Pokémon species names.
Module Name: random_forest
Prefix: rf_
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.features.data_priors import move_users, species_frequency, team_share_with_any
from src.config import DATA_DIR, TYPE_CHART_CSV

PREFIX = "rf_"

GEN2_TYPES: List[str] = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel"
]

PHYSICAL_TYPES: Set[str] = {
    "normal", "fighting", "flying", "poison", "ground",
    "rock", "bug", "ghost", "steel"
}
SPECIAL_TYPES: Set[str] = {
    "fire", "water", "grass", "electric", "psychic",
    "ice", "dragon", "dark"
}

# Pulizia (persona D, 30/10): rimossa la classifica Smogon Gen 2 OU (dato esterno, vietato dal
# regolamento). La "viability" ora si calcola solo dalle statistiche base presenti nei dati, con la
# stessa formula di f_mitigations._get_viability.
DEFAULT_OU_VIABILITY: float = 65.0  # valore a priori per i Pokémon di P2 non ancora rivelati

# ---------------------------------------------------------------------------
# Type Chart & Pokedex In-Memory Cache
# ---------------------------------------------------------------------------
_TYPE_CHART: Optional[Dict[Tuple[str, str], float]] = None
_POKEDEX: Optional[Dict[str, Dict[str, Any]]] = None

POKEDEX_JSON_PATH = DATA_DIR / "pokedex.json"


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


def _get_multiplier(atk_type: str, def_type: str, chart: Dict[Tuple[str, str], float]) -> float:
    if not atk_type or not def_type or atk_type == "notype" or def_type == "notype":
        return 1.0
    return chart.get((atk_type, def_type), 1.0)


def _defensive_multiplier(atk_type: str, def_types: List[str], chart: Dict[Tuple[str, str], float]) -> float:
    valid_def = [t for t in def_types if t and t != "notype"]
    if not valid_def:
        return 1.0
    mult = 1.0
    for d in valid_def:
        mult *= _get_multiplier(atk_type, d, chart)
    return mult


def _calc_max_offensive_eff(atk_types: List[str], def_types: List[str], chart: Dict[Tuple[str, str], float]) -> float:
    valid_atk = [t for t in atk_types if t and t != "notype"]
    valid_def = [t for t in def_types if t and t != "notype"]
    if not valid_atk or not valid_def:
        return 1.0

    max_eff = 0.0
    for a in valid_atk:
        eff = 1.0
        for d in valid_def:
            eff *= _get_multiplier(a, d, chart)
        if eff > max_eff:
            max_eff = eff
    return max_eff


STAGE_MULT = {
    -6: 0.25, -5: 0.28, -4: 0.33, -3: 0.40, -2: 0.50, -1: 0.66,
    0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0
}


def _bst_viability(name: str, pokedex: Dict[str, Dict[str, Any]]) -> float:
    st = _lookup_stats(name, pokedex)
    bst = sum(st.get(f"base_{s}", 80.0) for s in ["hp", "atk", "def", "spa", "spd", "spe"])
    return float(min(80.0, max(20.0, (bst - 300.0) / 3.0)))


def _lookup_stats(name: str, pokedex: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    norm = str(name).strip().lower()
    return pokedex.get(norm, {
        "types": ["normal"],
        "base_hp": 80.0,
        "base_atk": 80.0,
        "base_def": 80.0,
        "base_spa": 80.0,
        "base_spd": 80.0,
        "base_spe": 80.0,
    })


# ---------------------------------------------------------------------------
# Feature Extraction Function
# ---------------------------------------------------------------------------
def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    chart = _load_type_chart()
    pokedex = _load_pokedex()
    feat: Dict[str, float] = {}

    p1_team = data.get("p1_team_details", [])
    p2_lead = data.get("p2_lead_details", {})
    timeline = data.get("battle_timeline", [])

    p1_team_names = [p.get("name", f"p1_{i}").strip().lower() for i, p in enumerate(p1_team)]
    p2_lead_name = p2_lead.get("name", "p2_lead").strip().lower()

    p1_pokemon_map: Dict[str, Dict[str, Any]] = {
        p.get("name", "").strip().lower(): p for p in p1_team if p.get("name")
    }

    # Tracking states
    p1_last_hp: Dict[str, float] = {name: 1.0 for name in p1_team_names}
    p2_last_hp: Dict[str, float] = {p2_lead_name: 1.0}

    p1_last_status: Dict[str, str] = {name: "nostatus" for name in p1_team_names}
    p2_last_status: Dict[str, str] = {p2_lead_name: "nostatus"}

    p1_fainted: Set[str] = set()
    p2_fainted: Set[str] = set()
    p2_seen_names: Set[str] = {p2_lead_name}

    # Turn counter in field for Leftovers recovery estimation
    p1_turns_on_field: Dict[str, int] = {name: 0 for name in p1_team_names}
    p2_turns_on_field: Dict[str, int] = {p2_lead_name: 0}

    # Move analysis trackers
    p1_spikes_set = 0.0
    p2_spikes_set = 0.0
    p1_rapid_spin_count = 0.0
    p2_rapid_spin_count = 0.0
    p1_rest_count = 0.0
    p2_rest_count = 0.0
    p1_sleeptalk_count = 0.0
    p2_sleeptalk_count = 0.0
    p1_screens_count = 0.0
    p2_screens_count = 0.0
    p1_curse_count = 0.0
    p2_curse_count = 0.0

    # Advanced tactical move counters
    p1_toxic_count = 0.0
    p2_toxic_count = 0.0
    p1_twave_count = 0.0
    p2_twave_count = 0.0
    p1_sleep_move_count = 0.0
    p2_sleep_move_count = 0.0
    p1_heal_count = 0.0
    p2_heal_count = 0.0
    p1_substitute_count = 0.0
    p2_substitute_count = 0.0
    p1_explosion_count = 0.0
    p2_explosion_count = 0.0
    p1_phazing_count = 0.0
    p2_phazing_count = 0.0
    p1_belly_drum_count = 0.0
    p2_belly_drum_count = 0.0
    p1_baton_pass_count = 0.0
    p2_baton_pass_count = 0.0
    p1_thief_count = 0.0
    p2_thief_count = 0.0

    p1_first_faint_turn = 31.0
    p2_first_faint_turn = 31.0
    p1_second_faint_turn = 31.0
    p2_second_faint_turn = 31.0
    p1_faint_count_t10 = 0.0
    p2_faint_count_t10 = 0.0
    p1_faint_count_t20 = 0.0
    p2_faint_count_t20 = 0.0
    p1_moves_executed = 0.0
    p2_moves_executed = 0.0
    p1_last_rest_turn = -10
    p2_last_rest_turn = -10
    p1_act_streak = 0.0
    p2_act_streak = 0.0
    p1_act_toxic_streak = 0.0
    p2_act_toxic_streak = 0.0

    # Multi-bracket HP snapshots
    p1_hp_t10: Optional[float] = None
    p2_hp_t10: Optional[float] = None
    p1_hp_t20: Optional[float] = None
    p2_hp_t20: Optional[float] = None

    # Multi-bracket switch counters
    p1_switches_t1_10 = 0.0
    p2_switches_t1_10 = 0.0
    p1_switches_t11_20 = 0.0
    p2_switches_t11_20 = 0.0
    p1_switches_late = 0.0
    p2_switches_late = 0.0

    prev_p1_name = None
    prev_p2_name = None

    # Damage category counters across timeline
    p1_phys_moves = 0.0
    p1_spec_moves = 0.0
    p1_stat_moves = 0.0
    p1_phys_bp = 0.0
    p1_spec_bp = 0.0

    p2_phys_moves = 0.0
    p2_spec_moves = 0.0
    p2_stat_moves = 0.0
    p2_phys_bp = 0.0
    p2_spec_bp = 0.0

    p1_late_baton_pass = 0.0
    p2_late_baton_pass = 0.0
    p1_effective_explosion = 0.0
    p2_effective_explosion = 0.0

    for turn_idx, turn in enumerate(timeline, start=1):
        p1_st = turn.get("p1_pokemon_state", {})
        p2_st = turn.get("p2_pokemon_state", {})
        p1_mv = turn.get("p1_move_details") or {}
        p2_mv = turn.get("p2_move_details") or {}

        p1_name = p1_st.get("name", "").strip().lower()
        p2_name = p2_st.get("name", "").strip().lower()

        if p1_name:
            p1_turns_on_field[p1_name] = p1_turns_on_field.get(p1_name, 0) + 1
        if p2_name:
            p2_turns_on_field[p2_name] = p2_turns_on_field.get(p2_name, 0) + 1

        # Track switches across brackets and active streaks
        if prev_p1_name and p1_name and p1_name != prev_p1_name:
            p1_act_streak = 1.0
            p1_act_toxic_streak = 0.0
            if turn_idx <= 10:
                p1_switches_t1_10 += 1.0
            elif turn_idx <= 20:
                p1_switches_t11_20 += 1.0
            else:
                p1_switches_late += 1.0
        else:
            p1_act_streak += 1.0

        if prev_p2_name and p2_name and p2_name != prev_p2_name:
            p2_act_streak = 1.0
            p2_act_toxic_streak = 0.0
            if turn_idx <= 10:
                p2_switches_t1_10 += 1.0
            elif turn_idx <= 20:
                p2_switches_t11_20 += 1.0
            else:
                p2_switches_late += 1.0
        else:
            p2_act_streak += 1.0

        if p1_name:
            prev_p1_name = p1_name
        if p2_name:
            prev_p2_name = p2_name

        # State updates
        if p1_name:
            p1_hp = float(p1_st.get("hp_pct", 1.0))
            p1_status = p1_st.get("status", "nostatus")
            if p1_name in p1_fainted or p1_hp <= 0.001 or p1_status == "fnt":
                p1_fainted.add(p1_name)
                p1_last_hp[p1_name] = 0.0
                p1_last_status[p1_name] = "fnt"
                if p1_first_faint_turn > 30:
                    p1_first_faint_turn = float(turn_idx)
                elif p1_second_faint_turn > 30:
                    p1_second_faint_turn = float(turn_idx)
            else:
                p1_last_hp[p1_name] = p1_hp
                p1_last_status[p1_name] = p1_status
            if p1_status == "tox" and p1_name not in p1_fainted:
                p1_act_toxic_streak += 1.0

        if p2_name:
            p2_seen_names.add(p2_name)
            p2_hp = float(p2_st.get("hp_pct", 1.0))
            p2_status = p2_st.get("status", "nostatus")
            if p2_name in p2_fainted or p2_hp <= 0.001 or p2_status == "fnt":
                p2_fainted.add(p2_name)
                p2_last_hp[p2_name] = 0.0
                p2_last_status[p2_name] = "fnt"
                if p2_first_faint_turn > 30:
                    p2_first_faint_turn = float(turn_idx)
                elif p2_second_faint_turn > 30:
                    p2_second_faint_turn = float(turn_idx)
            else:
                p2_last_hp[p2_name] = p2_hp
                p2_last_status[p2_name] = p2_status
            if p2_status == "tox" and p2_name not in p2_fainted:
                p2_act_toxic_streak += 1.0

        # Move execution tracking
        if p1_mv.get("name"):
            p1_moves_executed += 1.0
        if p2_mv.get("name"):
            p2_moves_executed += 1.0

        # Move categorization for P1
        m1_cat = str(p1_mv.get("category", "")).lower()
        m1_bp = float(p1_mv.get("base_power", 0.0) or 0.0)
        m1_name = str(p1_mv.get("name", "")).lower()
        if m1_cat == "physical":
            p1_phys_moves += 1.0
            p1_phys_bp += m1_bp
        elif m1_cat == "special":
            p1_spec_moves += 1.0
            p1_spec_bp += m1_bp
        elif m1_cat == "status":
            p1_stat_moves += 1.0

        if m1_name == "spikes":
            p1_spikes_set = 1.0
        elif m1_name == "rapidspin":
            p1_rapid_spin_count += 1.0
            p1_spikes_set = 0.0
        elif m1_name == "rest":
            p1_rest_count += 1.0
            p1_last_rest_turn = turn_idx
        elif m1_name == "sleeptalk":
            p1_sleeptalk_count += 1.0
        elif m1_name in ("reflect", "lightscreen"):
            p1_screens_count += 1.0
        elif m1_name == "curse":
            p1_curse_count += 1.0
        elif m1_name == "toxic":
            p1_toxic_count += 1.0
        elif m1_name in ("thunderwave", "glare", "stunspore"):
            p1_twave_count += 1.0
        elif m1_name in ("spore", "sleeppowder", "hypnosis", "lovelykiss", "sing"):
            p1_sleep_move_count += 1.0
        elif m1_name in ("recover", "softboiled", "milkdrink", "moonlight", "synthesis", "morningsun"):
            p1_heal_count += 1.0
        elif m1_name == "substitute":
            p1_substitute_count += 1.0
        elif m1_name in ("explosion", "selfdestruct"):
            p1_explosion_count += 1.0
            if (p2_hp <= 0.001 or p2_status == "fnt" or p2_name in p2_fainted):
                p1_effective_explosion = min(1.0, p1_effective_explosion + 1.0)
            else:
                p1_effective_explosion = max(-1.0, p1_effective_explosion - 1.0)
        elif m1_name in ("whirlwind", "roar"):
            p1_phazing_count += 1.0
        elif m1_name == "bellydrum":
            p1_belly_drum_count += 1.0
        elif m1_name == "batonpass":
            p1_baton_pass_count += 1.0
            if turn_idx >= 28:
                p1_late_baton_pass = 1.0
        elif m1_name == "thief":
            p1_thief_count += 1.0

        # Move categorization for P2
        m2_cat = str(p2_mv.get("category", "")).lower()
        m2_bp = float(p2_mv.get("base_power", 0.0) or 0.0)
        m2_name = str(p2_mv.get("name", "")).lower()
        if m2_cat == "physical":
            p2_phys_moves += 1.0
            p2_phys_bp += m2_bp
        elif m2_cat == "special":
            p2_spec_moves += 1.0
            p2_spec_bp += m2_bp
        elif m2_cat == "status":
            p2_stat_moves += 1.0

        if m2_name == "spikes":
            p2_spikes_set = 1.0
        elif m2_name == "rapidspin":
            p2_rapid_spin_count += 1.0
            p2_spikes_set = 0.0
        elif m2_name == "rest":
            p2_rest_count += 1.0
            p2_last_rest_turn = turn_idx
        elif m2_name == "sleeptalk":
            p2_sleeptalk_count += 1.0
        elif m2_name in ("reflect", "lightscreen"):
            p2_screens_count += 1.0
        elif m2_name == "curse":
            p2_curse_count += 1.0
        elif m2_name == "toxic":
            p2_toxic_count += 1.0
        elif m2_name in ("thunderwave", "glare", "stunspore"):
            p2_twave_count += 1.0
        elif m2_name in ("spore", "sleeppowder", "hypnosis", "lovelykiss", "sing"):
            p2_sleep_move_count += 1.0
        elif m2_name in ("recover", "softboiled", "milkdrink", "moonlight", "synthesis", "morningsun"):
            p2_heal_count += 1.0
        elif m2_name == "substitute":
            p2_substitute_count += 1.0
        elif m2_name in ("explosion", "selfdestruct"):
            p2_explosion_count += 1.0
            if (p1_hp <= 0.001 or p1_status == "fnt" or p1_name in p1_fainted):
                p2_effective_explosion = min(1.0, p2_effective_explosion + 1.0)
            else:
                p2_effective_explosion = max(-1.0, p2_effective_explosion - 1.0)
        elif m2_name in ("whirlwind", "roar"):
            p2_phazing_count += 1.0
        elif m2_name == "bellydrum":
            p2_belly_drum_count += 1.0
        elif m2_name == "batonpass":
            p2_baton_pass_count += 1.0
            if turn_idx >= 28:
                p2_late_baton_pass = 1.0
        elif m2_name == "thief":
            p2_thief_count += 1.0

        # Snapshot at T10
        if turn_idx == 10:
            p1_hp_t10 = sum(hp for n, hp in p1_last_hp.items() if n not in p1_fainted)
            p2_unrev10 = max(0, 6 - len(p2_seen_names))
            p2_hp_t10 = sum(hp for n, hp in p2_last_hp.items() if n not in p2_fainted) + (p2_unrev10 * 1.0)
            p1_faint_count_t10 = float(len(p1_fainted))
            p2_faint_count_t10 = float(len(p2_fainted))

        # Snapshot at T20
        if turn_idx == 20:
            p1_hp_t20 = sum(hp for n, hp in p1_last_hp.items() if n not in p1_fainted)
            p2_unrev20 = max(0, 6 - len(p2_seen_names))
            p2_hp_t20 = sum(hp for n, hp in p2_last_hp.items() if n not in p2_fainted) + (p2_unrev20 * 1.0)
            p1_faint_count_t20 = float(len(p1_fainted))
            p2_faint_count_t20 = float(len(p2_fainted))

    # Guarantee fainted Pokémon are strictly zeroed out across all state
    for fn in p1_fainted:
        p1_last_hp[fn] = 0.0
        p1_last_status[fn] = "fnt"
    for fn in p2_fainted:
        p2_last_hp[fn] = 0.0
        p2_last_status[fn] = "fnt"

    # =========================================================================
    # 1. ASSE A: INTERAZIONI DEI TIPI (Matrice 17 Tipi Gen 2)
    # =========================================================================
    p1_total_weaknesses = 0
    p1_total_resistances = 0
    p1_type_blindspots = 0

    for t in GEN2_TYPES:
        weak_count = 0
        res_count = 0
        for p in p1_team:
            types = p.get("types", ["notype", "notype"])
            mult = _defensive_multiplier(t, types, chart)
            if mult > 1.0:
                weak_count += 1
            elif mult < 1.0:
                res_count += 1

        if t != "normal":
            feat[f"{PREFIX}type_weak_{t}"] = float(weak_count)
        feat[f"{PREFIX}type_res_{t}"] = float(res_count)
        feat[f"{PREFIX}type_net_{t}"] = float(res_count - weak_count)

        p1_total_weaknesses += weak_count
        p1_total_resistances += res_count
        if weak_count >= 3:
            p1_type_blindspots += 1

    feat[f"{PREFIX}p1_total_type_weaknesses"] = float(p1_total_weaknesses)
    feat[f"{PREFIX}p1_total_type_resistances"] = float(p1_total_resistances)
    feat[f"{PREFIX}p1_type_blindspots"] = float(p1_type_blindspots)

    # Lead Matchup Type Interaction
    p1_lead = p1_team[0] if p1_team else {}
    p1_lead_types = p1_lead.get("types", ["notype", "notype"])
    p2_lead_types = p2_lead.get("types", ["notype", "notype"])

    p1_lead_eff = _calc_max_offensive_eff(p1_lead_types, p2_lead_types, chart)
    p2_lead_eff = _calc_max_offensive_eff(p2_lead_types, p1_lead_types, chart)

    feat[f"{PREFIX}lead_p1_type_eff"] = float(p1_lead_eff)
    feat[f"{PREFIX}lead_p2_type_eff"] = float(p2_lead_eff)
    feat[f"{PREFIX}diff_lead_type_eff"] = float(p1_lead_eff - p2_lead_eff)

    # =========================================================================
    # 2. ASSE B: VALUTAZIONE OGGETTI (Leftovers & Item Economy)
    # =========================================================================
    p1_leftovers_count = 0
    p1_berries_count = 0
    p1_offensive_items = 0
    p1_lefties_turns = 0

    for p in p1_team:
        item = str(p.get("item", "")).strip().lower()
        pname = p.get("name", "").strip().lower()
        if item == "leftovers":
            p1_leftovers_count += 1
            p1_lefties_turns += p1_turns_on_field.get(pname, 0)
        elif "berry" in item or item in ("miracleberry", "mintberry", "bitterberry"):
            p1_berries_count += 1
        elif item in ("thickclub", "lightball", "charcoal", "mysticwater", "miracleseed",
                      "magnet", "hardstone", "spelltag", "blackglasses", "metalcoat",
                      "dragonscale", "sharpbeak", "softsand", "poisonbarb", "nevermeltice",
                      "twistedspoon", "blackbelt", "silverpowder", "focusband", "kingsrock"):
            p1_offensive_items += 1

    feat[f"{PREFIX}p1_leftovers_count"] = float(p1_leftovers_count)
    feat[f"{PREFIX}p1_berries_count"] = float(p1_berries_count)
    feat[f"{PREFIX}p1_offensive_items"] = float(p1_offensive_items)
    feat[f"{PREFIX}p1_lefties_active_turns"] = float(p1_lefties_turns)

    # Lead item comparison
    p1_lead_item = str(p1_lead.get("item", "")).strip().lower()
    feat[f"{PREFIX}p1_lead_has_leftovers"] = 1.0 if p1_lead_item == "leftovers" else 0.0

    # =========================================================================
    # 3. ASSE C: CUMULATIVE STATS (HP, ATK, DEF, SPA, SPD, SPE, BST) & BULK
    # =========================================================================
    p1_team_stats = {"hp": 0.0, "atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0}
    p1_alive_stats = {"hp": 0.0, "atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0}
    p1_alive_phys_bulks: List[float] = []
    p1_alive_spec_bulks: List[float] = []
    p1_alive_count = 0
    p1_alive_grounded = 0

    for p in p1_team:
        pname = p.get("name", "").strip().lower()
        hp = float(p.get("base_hp", 80))
        atk = float(p.get("base_atk", 80))
        dfn = float(p.get("base_def", 80))
        spa = float(p.get("base_spa", 80))
        spd = float(p.get("base_spd", 80))
        spe = float(p.get("base_spe", 80))
        types = [t.lower() for t in p.get("types", []) if t]

        p1_team_stats["hp"] += hp
        p1_team_stats["atk"] += atk
        p1_team_stats["def"] += dfn
        p1_team_stats["spa"] += spa
        p1_team_stats["spd"] += spd
        p1_team_stats["spe"] += spe

        if pname not in p1_fainted:
            p1_alive_count += 1
            p1_alive_stats["hp"] += hp
            p1_alive_stats["atk"] += atk
            p1_alive_stats["def"] += dfn
            p1_alive_stats["spa"] += spa
            p1_alive_stats["spd"] += spd
            p1_alive_stats["spe"] += spe

            p1_alive_phys_bulks.append((hp * dfn) / 100.0)
            p1_alive_spec_bulks.append((hp * spd) / 100.0)
            if "flying" not in types:
                p1_alive_grounded += 1

    # P2 stats from revealed Pokémon and pokedex
    p2_revealed_stats = {"hp": 0.0, "atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0}
    p2_alive_stats = {"hp": 0.0, "atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0}
    p2_alive_phys_bulks: List[float] = []
    p2_alive_spec_bulks: List[float] = []
    p2_alive_count = 0
    p2_alive_grounded = 0

    for p2_name in p2_seen_names:
        stats = _lookup_stats(p2_name, pokedex)
        hp = float(stats["base_hp"])
        atk = float(stats["base_atk"])
        dfn = float(stats["base_def"])
        spa = float(stats["base_spa"])
        spd = float(stats["base_spd"])
        spe = float(stats["base_spe"])
        types = [t.lower() for t in stats.get("types", []) if t]

        p2_revealed_stats["hp"] += hp
        p2_revealed_stats["atk"] += atk
        p2_revealed_stats["def"] += dfn
        p2_revealed_stats["spa"] += spa
        p2_revealed_stats["spd"] += spd
        p2_revealed_stats["spe"] += spe

        if p2_name not in p2_fainted:
            p2_alive_count += 1
            p2_alive_stats["hp"] += hp
            p2_alive_stats["atk"] += atk
            p2_alive_stats["def"] += dfn
            p2_alive_stats["spa"] += spa
            p2_alive_stats["spd"] += spd
            p2_alive_stats["spe"] += spe

            p2_alive_phys_bulks.append((hp * dfn) / 100.0)
            p2_alive_spec_bulks.append((hp * spd) / 100.0)
            if "flying" not in types:
                p2_alive_grounded += 1

    p2_unrevealed = max(0, 6 - len(p2_seen_names))
    p2_alive_total_count = p2_alive_count + p2_unrevealed

    # Fill unrevealed with standard Gen 2 OU average (85 across stats)
    p2_alive_stats["hp"] += p2_unrevealed * 85.0
    p2_alive_stats["atk"] += p2_unrevealed * 85.0
    p2_alive_stats["def"] += p2_unrevealed * 85.0
    p2_alive_stats["spa"] += p2_unrevealed * 85.0
    p2_alive_stats["spd"] += p2_unrevealed * 85.0
    p2_alive_stats["spe"] += p2_unrevealed * 85.0
    p2_alive_grounded += int(round(p2_unrevealed * 0.80))

    # Base Stat Totals (BST)
    p1_alive_bst = sum(p1_alive_stats.values())
    p2_alive_bst = sum(p2_alive_stats.values())

    feat[f"{PREFIX}p1_alive_total_bst"] = float(p1_alive_bst)
    feat[f"{PREFIX}p2_alive_total_bst"] = float(p2_alive_bst)
    feat[f"{PREFIX}diff_alive_total_bst"] = float(p1_alive_bst - p2_alive_bst)

    # Cumulative Stat Differentials
    for stat in ("atk", "def", "spa", "spd", "spe", "hp"):
        feat[f"{PREFIX}p1_alive_sum_{stat}"] = float(p1_alive_stats[stat])
        feat[f"{PREFIX}p2_alive_sum_{stat}"] = float(p2_alive_stats[stat])
        feat[f"{PREFIX}diff_alive_sum_{stat}"] = float(p1_alive_stats[stat] - p2_alive_stats[stat])

    # Bulk summaries
    feat[f"{PREFIX}p1_alive_max_atk"] = float(max(p1_alive_stats["atk"] / max(1, p1_alive_count), 0.0))
    feat[f"{PREFIX}p1_alive_mean_atk"] = float(p1_alive_stats["atk"] / max(1, p1_alive_count))
    feat[f"{PREFIX}p1_alive_max_spa"] = float(max(p1_alive_stats["spa"] / max(1, p1_alive_count), 0.0))
    feat[f"{PREFIX}p1_alive_mean_spa"] = float(p1_alive_stats["spa"] / max(1, p1_alive_count))

    p1_alive_phys_sum = sum(p1_alive_phys_bulks)
    p1_alive_spec_sum = sum(p1_alive_spec_bulks)
    p2_alive_phys_sum = sum(p2_alive_phys_bulks) + (p2_unrevealed * 72.0)
    p2_alive_spec_sum = sum(p2_alive_spec_bulks) + (p2_unrevealed * 72.0)

    feat[f"{PREFIX}p1_alive_phys_bulk_sum"] = float(p1_alive_phys_sum)
    feat[f"{PREFIX}p1_alive_spec_bulk_sum"] = float(p1_alive_spec_sum)
    feat[f"{PREFIX}p2_alive_phys_bulk_sum"] = float(p2_alive_phys_sum)
    feat[f"{PREFIX}p2_alive_spec_bulk_sum"] = float(p2_alive_spec_sum)
    feat[f"{PREFIX}diff_alive_phys_bulk_sum"] = float(p1_alive_phys_sum - p2_alive_phys_sum)
    feat[f"{PREFIX}diff_alive_spec_bulk_sum"] = float(p1_alive_spec_sum - p2_alive_spec_sum)

    feat[f"{PREFIX}p1_alive_max_phys_bulk"] = float(max(p1_alive_phys_bulks) if p1_alive_phys_bulks else 0.0)
    feat[f"{PREFIX}p1_alive_max_spec_bulk"] = float(max(p1_alive_spec_bulks) if p1_alive_spec_bulks else 0.0)

    p1_mean_phys_bulk = p1_alive_phys_sum / max(1, p1_alive_count)
    p1_mean_spec_bulk = p1_alive_spec_sum / max(1, p1_alive_count)
    feat[f"{PREFIX}p1_alive_bulk_bias_ratio"] = float(p1_mean_phys_bulk / max(0.01, p1_mean_spec_bulk))

    # Lead comparison bulk
    p2_l_hp = float(p2_lead.get("base_hp", 80))
    p2_l_def = float(p2_lead.get("base_def", 80))
    p2_l_spd = float(p2_lead.get("base_spd", 80))
    p2_l_phys_bulk = (p2_l_hp * p2_l_def) / 100.0
    p2_l_spec_bulk = (p2_l_hp * p2_l_spd) / 100.0

    feat[f"{PREFIX}diff_phys_bulk_vs_lead"] = float(p1_mean_phys_bulk - p2_l_phys_bulk)
    feat[f"{PREFIX}diff_spec_bulk_vs_lead"] = float(p1_mean_spec_bulk - p2_l_spec_bulk)

    # =========================================================================
    # 4. ASSE D: EXACT TURN 30 STATE (Micro-Tactical State & Active Matchup)
    # =========================================================================
    p1_act_phys_boost_diff = 0.0
    p1_act_spec_boost_diff = 0.0
    p1_act_phys_bulk_t30 = 0.0
    p1_act_spec_bulk_t30 = 0.0
    p2_act_phys_bulk_t30 = 0.0
    p2_act_spec_bulk_t30 = 0.0

    p1_t30_type_eff = 1.0
    p2_t30_type_eff = 1.0
    p1_act_speed_est = 100.0
    p2_act_speed_est = 100.0

    p1_act_is_asleep = 0.0
    p2_act_is_asleep = 0.0
    p1_act_is_paralyzed = 0.0
    p2_act_is_paralyzed = 0.0
    p1_act_is_frozen = 0.0
    p2_act_is_frozen = 0.0
    p1_act_is_toxic = 0.0
    p2_act_is_toxic = 0.0
    p1_act_is_burned = 0.0
    p2_act_is_burned = 0.0

    p1_has_substitute = 0.0
    p2_has_substitute = 0.0
    p1_has_leechseed = 0.0
    p2_has_leechseed = 0.0
    p1_is_confused = 0.0
    p2_is_confused = 0.0
    p1_is_trapped = 0.0
    p2_is_trapped = 0.0
    p1_is_perish = 0.0
    p2_is_perish = 0.0

    p1_act_eff_atk = 80.0
    p2_act_eff_atk = 80.0
    p1_act_eff_spa = 80.0
    p2_act_eff_spa = 80.0
    p1_act_eff_def = 80.0
    p2_act_eff_def = 80.0
    p1_act_eff_spd = 80.0
    p2_act_eff_spd = 80.0

    p1_act_is_fnt = 0.0
    p2_act_is_fnt = 0.0
    p1_act_name = ""
    p2_act_name = ""
    p1_hp_t30 = 1.0
    p2_hp_t30 = 1.0

    if timeline:
        last_turn = timeline[-1]
        p1_st30 = last_turn.get("p1_pokemon_state", {})
        p2_st30 = last_turn.get("p2_pokemon_state", {})

        p1_act_name = p1_st30.get("name", "").strip().lower()
        p2_act_name = p2_st30.get("name", "").strip().lower()

        p1_hp_t30 = float(p1_st30.get("hp_pct", 1.0))
        p2_hp_t30 = float(p2_st30.get("hp_pct", 1.0))

        p1_status_t30 = p1_st30.get("status", "nostatus")
        p2_status_t30 = p2_st30.get("status", "nostatus")

        p1_act_is_fnt = 1.0 if (p1_hp_t30 <= 0.001 or p1_status_t30 == "fnt" or p1_act_name in p1_fainted) else 0.0
        p2_act_is_fnt = 1.0 if (p2_hp_t30 <= 0.001 or p2_status_t30 == "fnt" or p2_act_name in p2_fainted) else 0.0

        if p1_act_is_fnt:
            p1_hp_t30 = 0.0
        if p2_act_is_fnt:
            p2_hp_t30 = 0.0

        p1_act_is_asleep = 1.0 if (p1_status_t30 == "slp" and not p1_act_is_fnt) else 0.0
        p2_act_is_asleep = 1.0 if (p2_status_t30 == "slp" and not p2_act_is_fnt) else 0.0
        p1_act_is_paralyzed = 1.0 if (p1_status_t30 == "par" and not p1_act_is_fnt) else 0.0
        p2_act_is_paralyzed = 1.0 if (p2_status_t30 == "par" and not p2_act_is_fnt) else 0.0
        p1_act_is_frozen = 1.0 if (p1_status_t30 == "frz" and not p1_act_is_fnt) else 0.0
        p2_act_is_frozen = 1.0 if (p2_status_t30 == "frz" and not p2_act_is_fnt) else 0.0
        p1_act_is_toxic = 1.0 if (p1_status_t30 == "tox" and not p1_act_is_fnt) else 0.0
        p2_act_is_toxic = 1.0 if (p2_status_t30 == "tox" and not p2_act_is_fnt) else 0.0
        p1_act_is_burned = 1.0 if (p1_status_t30 == "brn" and not p1_act_is_fnt) else 0.0
        p2_act_is_burned = 1.0 if (p2_status_t30 == "brn" and not p2_act_is_fnt) else 0.0

        p1_effects = p1_st30.get("effects", []) if not p1_act_is_fnt else []
        p2_effects = p2_st30.get("effects", []) if not p2_act_is_fnt else []

        p1_has_substitute = 1.0 if "substitute" in p1_effects else 0.0
        p2_has_substitute = 1.0 if "substitute" in p2_effects else 0.0
        p1_has_leechseed = 1.0 if "leechseed" in p1_effects else 0.0
        p2_has_leechseed = 1.0 if "leechseed" in p2_effects else 0.0
        p1_is_confused = 1.0 if "confusion" in p1_effects else 0.0
        p2_is_confused = 1.0 if "confusion" in p2_effects else 0.0
        p1_is_trapped = 1.0 if any(e in p1_effects for e in ("trapped", "wrap", "firespin", "whirlpool", "clamp")) else 0.0
        p2_is_trapped = 1.0 if any(e in p2_effects for e in ("trapped", "wrap", "firespin", "whirlpool", "clamp")) else 0.0
        p1_is_perish = 1.0 if "perish" in p1_effects else 0.0
        p2_is_perish = 1.0 if "perish" in p2_effects else 0.0

        # Look up exact details from p1_team or pokedex
        p1_det = p1_pokemon_map.get(p1_act_name) or _lookup_stats(p1_act_name, pokedex)
        p2_det = _lookup_stats(p2_act_name, pokedex)

        p1_types = [t.lower() for t in p1_det.get("types", ["notype", "notype"]) if t]
        p2_types = [t.lower() for t in p2_det.get("types", ["notype", "notype"]) if t]

        p1_t30_type_eff = _calc_max_offensive_eff(p1_types, p2_types, chart)
        p2_t30_type_eff = _calc_max_offensive_eff(p2_types, p1_types, chart)

        # Stat boosts (zeroed out if fainted)
        p1_b = p1_st30.get("boosts", {}) if not p1_act_is_fnt else {}
        p2_b = p2_st30.get("boosts", {}) if not p2_act_is_fnt else {}

        p1_atk_b = float(p1_b.get("atk", 0))
        p1_def_b = float(p1_b.get("def", 0))
        p1_spa_b = float(p1_b.get("spa", 0))
        p1_spd_b = float(p1_b.get("spd", 0))
        p1_spe_b = float(p1_b.get("spe", 0))

        p2_atk_b = float(p2_b.get("atk", 0))
        p2_def_b = float(p2_b.get("def", 0))
        p2_spa_b = float(p2_b.get("spa", 0))
        p2_spd_b = float(p2_b.get("spd", 0))
        p2_spe_b = float(p2_b.get("spe", 0))

        p1_act_phys_boost_diff = (p1_atk_b - p2_def_b) - (p2_atk_b - p1_def_b)
        p1_act_spec_boost_diff = (p1_spa_b - p2_spd_b) - (p2_spa_b - p1_spd_b)

        b1_spe_mult = STAGE_MULT.get(int(max(-6, min(6, p1_spe_b))), 1.0)
        b2_spe_mult = STAGE_MULT.get(int(max(-6, min(6, p2_spe_b))), 1.0)

        par1_mult = 0.25 if p1_act_is_paralyzed else 1.0
        par2_mult = 0.25 if p2_act_is_paralyzed else 1.0

        p1_base_spe = float(p1_det.get("base_spe", 80))
        p2_base_spe = float(p2_det.get("base_spe", 80))

        p1_act_speed_est = (p1_base_spe * b1_spe_mult * par1_mult) if not p1_act_is_fnt else 0.0
        p2_act_speed_est = (p2_base_spe * b2_spe_mult * par2_mult) if not p2_act_is_fnt else 0.0

        p1_base_hp = float(p1_det.get("base_hp", 80))
        p1_base_def = float(p1_det.get("base_def", 80))
        p1_base_spd = float(p1_det.get("base_spd", 80))
        p1_base_atk = float(p1_det.get("base_atk", 80))
        p1_base_spa = float(p1_det.get("base_spa", 80))

        p2_base_hp = float(p2_det.get("base_hp", 80))
        p2_base_def = float(p2_det.get("base_def", 80))
        p2_base_spd = float(p2_det.get("base_spd", 80))
        p2_base_atk = float(p2_det.get("base_atk", 80))
        p2_base_spa = float(p2_det.get("base_spa", 80))

        p1_act_phys_bulk_t30 = ((p1_base_hp * p1_base_def * STAGE_MULT.get(int(max(-6, min(6, p1_def_b))), 1.0)) / 100.0) if not p1_act_is_fnt else 0.0
        p1_act_spec_bulk_t30 = ((p1_base_hp * p1_base_spd * STAGE_MULT.get(int(max(-6, min(6, p1_spd_b))), 1.0)) / 100.0) if not p1_act_is_fnt else 0.0
        p2_act_phys_bulk_t30 = ((p2_base_hp * p2_base_def * STAGE_MULT.get(int(max(-6, min(6, p2_def_b))), 1.0)) / 100.0) if not p2_act_is_fnt else 0.0
        p2_act_spec_bulk_t30 = ((p2_base_hp * p2_base_spd * STAGE_MULT.get(int(max(-6, min(6, p2_spd_b))), 1.0)) / 100.0) if not p2_act_is_fnt else 0.0

        p1_act_eff_atk = (p1_base_atk * STAGE_MULT.get(int(max(-6, min(6, p1_atk_b))), 1.0) * (0.5 if p1_act_is_burned else 1.0)) if not p1_act_is_fnt else 0.0
        p2_act_eff_atk = (p2_base_atk * STAGE_MULT.get(int(max(-6, min(6, p2_atk_b))), 1.0) * (0.5 if p2_act_is_burned else 1.0)) if not p2_act_is_fnt else 0.0
        p1_act_eff_spa = (p1_base_spa * STAGE_MULT.get(int(max(-6, min(6, p1_spa_b))), 1.0)) if not p1_act_is_fnt else 0.0
        p2_act_eff_spa = (p2_base_spa * STAGE_MULT.get(int(max(-6, min(6, p2_spa_b))), 1.0)) if not p2_act_is_fnt else 0.0
        p1_act_eff_def = (p1_base_def * STAGE_MULT.get(int(max(-6, min(6, p1_def_b))), 1.0)) if not p1_act_is_fnt else 1.0
        p2_act_eff_def = (p2_base_def * STAGE_MULT.get(int(max(-6, min(6, p2_def_b))), 1.0)) if not p2_act_is_fnt else 1.0
        p1_act_eff_spd = (p1_base_spd * STAGE_MULT.get(int(max(-6, min(6, p1_spd_b))), 1.0)) if not p1_act_is_fnt else 1.0
        p2_act_eff_spd = (p2_base_spd * STAGE_MULT.get(int(max(-6, min(6, p2_spd_b))), 1.0)) if not p2_act_is_fnt else 1.0

    feat[f"{PREFIX}p1_act_is_fnt"] = float(p1_act_is_fnt)
    feat[f"{PREFIX}p2_act_is_fnt"] = float(p2_act_is_fnt)
    feat[f"{PREFIX}diff_act_is_fnt"] = float(p2_act_is_fnt - p1_act_is_fnt)

    feat[f"{PREFIX}t30_net_phys_boost_diff"] = float(p1_act_phys_boost_diff)
    feat[f"{PREFIX}t30_net_spec_boost_diff"] = float(p1_act_spec_boost_diff)
    feat[f"{PREFIX}t30_act_phys_bulk_diff"] = float(p1_act_phys_bulk_t30 - p2_act_phys_bulk_t30)
    feat[f"{PREFIX}t30_act_spec_bulk_diff"] = float(p1_act_spec_bulk_t30 - p2_act_spec_bulk_t30)

    feat[f"{PREFIX}p1_t30_type_eff"] = float(p1_t30_type_eff)
    feat[f"{PREFIX}p2_t30_type_eff"] = float(p2_t30_type_eff)
    feat[f"{PREFIX}diff_t30_type_eff"] = float(p1_t30_type_eff - p2_t30_type_eff)

    # Anti-Blunder: STAB Weakness Trap (e.g. Battle 4445)
    p1_weakness_trap = 1.0 if (p2_t30_type_eff >= 2.0 and p1_t30_type_eff <= 1.0 and not p1_act_is_fnt) else 0.0
    p2_weakness_trap = 1.0 if (p1_t30_type_eff >= 2.0 and p2_t30_type_eff <= 1.0 and not p2_act_is_fnt) else 0.0
    feat[f"{PREFIX}p1_act_weakness_trap"] = float(p1_weakness_trap)
    feat[f"{PREFIX}p2_act_weakness_trap"] = float(p2_weakness_trap)
    feat[f"{PREFIX}diff_act_weakness_trap"] = float(p2_weakness_trap - p1_weakness_trap)

    # Anti-Blunder: Sleep / Crippled Trap (e.g. Battle 7648)
    p1_sleep_trap = 1.0 if (p1_act_is_asleep and p2_hp_t30 > 0.25 and not p2_act_is_fnt) else 0.0
    p2_sleep_trap = 1.0 if (p2_act_is_asleep and p1_hp_t30 > 0.25 and not p1_act_is_fnt) else 0.0
    feat[f"{PREFIX}p1_act_sleep_trap"] = float(p1_sleep_trap)
    feat[f"{PREFIX}p2_act_sleep_trap"] = float(p2_sleep_trap)
    feat[f"{PREFIX}diff_act_sleep_trap"] = float(p2_sleep_trap - p1_sleep_trap)

    p1_crippled = 1.0 if ((p1_act_is_asleep or p1_act_is_frozen or (p1_act_is_paralyzed and p1_act_speed_est < p2_act_speed_est * 0.5)) and not p1_act_is_fnt) else 0.0
    p2_crippled = 1.0 if ((p2_act_is_asleep or p2_act_is_frozen or (p2_act_is_paralyzed and p2_act_speed_est < p1_act_speed_est * 0.5)) and not p2_act_is_fnt) else 0.0
    feat[f"{PREFIX}p1_act_crippled"] = float(p1_crippled)
    feat[f"{PREFIX}p2_act_crippled"] = float(p2_crippled)
    feat[f"{PREFIX}diff_act_crippled"] = float(p2_crippled - p1_crippled)

    # Anti-Blunder: Forced Switch into Spikes Penalty (Turn 31)
    p1_forced_spikes = 1.0 if (p1_act_is_fnt and p2_spikes_set) else 0.0
    p2_forced_spikes = 1.0 if (p2_act_is_fnt and p1_spikes_set) else 0.0
    feat[f"{PREFIX}p1_forced_switch_spikes"] = float(p1_forced_spikes)
    feat[f"{PREFIX}p2_forced_switch_spikes"] = float(p2_forced_spikes)
    feat[f"{PREFIX}diff_forced_switch_spikes"] = float(p2_forced_spikes - p1_forced_spikes)

    feat[f"{PREFIX}t30_speed_advantage"] = float(1.0 if p1_act_speed_est > p2_act_speed_est else 0.0)
    feat[f"{PREFIX}t30_diff_eff_spe"] = float(p1_act_speed_est - p2_act_speed_est)
    feat[f"{PREFIX}t30_speed_ratio"] = float(p1_act_speed_est / max(0.1, p1_act_speed_est + p2_act_speed_est))

    # Turn 30 Active Tactical Damage Exchange
    t30_p1_phys_ratio = (p1_act_eff_atk / max(1.0, p2_act_eff_def)) if not p1_act_is_fnt else 0.0
    t30_p1_spec_ratio = (p1_act_eff_spa / max(1.0, p2_act_eff_spd)) if not p1_act_is_fnt else 0.0
    t30_p2_phys_ratio = (p2_act_eff_atk / max(1.0, p1_act_eff_def)) if not p2_act_is_fnt else 0.0
    t30_p2_spec_ratio = (p2_act_eff_spa / max(1.0, p1_act_eff_spd)) if not p2_act_is_fnt else 0.0

    t30_p1_max_eff_dmg = max(t30_p1_phys_ratio, t30_p1_spec_ratio) * p1_t30_type_eff
    t30_p2_max_eff_dmg = max(t30_p2_phys_ratio, t30_p2_spec_ratio) * p2_t30_type_eff

    feat[f"{PREFIX}t30_p1_max_damage_pot"] = float(t30_p1_max_eff_dmg)
    feat[f"{PREFIX}t30_p2_max_damage_pot"] = float(t30_p2_max_eff_dmg)
    feat[f"{PREFIX}t30_exchange_advantage"] = float(t30_p1_max_eff_dmg - t30_p2_max_eff_dmg)

    # Active Turn 30 Statuses & Effects
    feat[f"{PREFIX}p1_active_is_asleep"] = float(p1_act_is_asleep)
    feat[f"{PREFIX}p2_active_is_asleep"] = float(p2_act_is_asleep)
    feat[f"{PREFIX}diff_active_asleep"] = float(p1_act_is_asleep - p2_act_is_asleep)

    feat[f"{PREFIX}p1_active_is_paralyzed"] = float(p1_act_is_paralyzed)
    feat[f"{PREFIX}p2_active_is_paralyzed"] = float(p2_act_is_paralyzed)
    feat[f"{PREFIX}diff_active_paralyzed"] = float(p1_act_is_paralyzed - p2_act_is_paralyzed)

    feat[f"{PREFIX}p1_active_is_toxic"] = float(p1_act_is_toxic)
    feat[f"{PREFIX}p2_active_is_toxic"] = float(p2_act_is_toxic)
    feat[f"{PREFIX}diff_active_toxic"] = float(p1_act_is_toxic - p2_act_is_toxic)

    feat[f"{PREFIX}p1_t30_has_substitute"] = float(p1_has_substitute)
    feat[f"{PREFIX}p2_t30_has_substitute"] = float(p2_has_substitute)
    feat[f"{PREFIX}diff_t30_substitute"] = float(p1_has_substitute - p2_has_substitute)

    # =========================================================================
    # 5. ASSE E: TIMELINE MOVES (Damage Volume & Stall Ratio)
    # =========================================================================
    p1_tot_moves = max(1.0, p1_phys_moves + p1_spec_moves + p1_stat_moves)
    p2_tot_moves = max(1.0, p2_phys_moves + p2_spec_moves + p2_stat_moves)

    feat[f"{PREFIX}p1_phys_move_share"] = float(p1_phys_moves / p1_tot_moves)
    feat[f"{PREFIX}p1_spec_move_share"] = float(p1_spec_moves / p1_tot_moves)
    feat[f"{PREFIX}p1_stall_move_share"] = float(p1_stat_moves / p1_tot_moves)
    feat[f"{PREFIX}p2_phys_move_share"] = float(p2_phys_moves / p2_tot_moves)
    feat[f"{PREFIX}p2_spec_move_share"] = float(p2_spec_moves / p2_tot_moves)
    feat[f"{PREFIX}p2_stall_move_share"] = float(p2_stat_moves / p2_tot_moves)

    feat[f"{PREFIX}p1_phys_bp_total"] = float(p1_phys_bp)
    feat[f"{PREFIX}p1_spec_bp_total"] = float(p1_spec_bp)
    feat[f"{PREFIX}p2_phys_bp_total"] = float(p2_phys_bp)
    feat[f"{PREFIX}p2_spec_bp_total"] = float(p2_spec_bp)
    feat[f"{PREFIX}diff_offensive_bp_total"] = float((p1_phys_bp + p1_spec_bp) - (p2_phys_bp + p2_spec_bp))

    # =========================================================================
    # 6. ASSE F: MULTI-BRACKET MOMENTUM & DYNAMICS (T0-T10, T10-T20, T20-T30)
    # =========================================================================
    p1_total_hp = sum(hp for name, hp in p1_last_hp.items() if name not in p1_fainted)
    p2_total_hp = sum(hp for name, hp in p2_last_hp.items() if name not in p2_fainted) + (p2_unrevealed * 1.0)

    p1_alive = max(0, 6 - len(p1_fainted))
    p2_alive = max(0, 6 - len(p2_fainted))

    feat[f"{PREFIX}hp_ratio"] = float(p1_total_hp / (p2_total_hp + 1e-4))
    feat[f"{PREFIX}alive_ratio"] = float(p1_alive / (p2_alive + 1e-4))

    p1_avg_hp = p1_total_hp / max(1, p1_alive)
    p2_avg_hp = p2_total_hp / max(1, p2_alive)
    feat[f"{PREFIX}p1_avg_hp_per_alive"] = float(p1_avg_hp)
    feat[f"{PREFIX}p2_avg_hp_per_alive"] = float(p2_avg_hp)
    feat[f"{PREFIX}diff_avg_hp_per_alive"] = float(p1_avg_hp - p2_avg_hp)

    p1_healthy = sum(1 for name, hp in p1_last_hp.items() if hp >= 0.70 and name not in p1_fainted)
    p2_healthy = sum(1 for name, hp in p2_last_hp.items() if hp >= 0.70 and name not in p2_fainted) + p2_unrevealed
    p1_critical = sum(1 for name, hp in p1_last_hp.items() if 0.0 < hp <= 0.25 and name not in p1_fainted)
    p2_critical = sum(1 for name, hp in p2_last_hp.items() if 0.0 < hp <= 0.25 and name not in p2_fainted)

    feat[f"{PREFIX}p1_healthy_count"] = float(p1_healthy)
    feat[f"{PREFIX}p2_healthy_count"] = float(p2_healthy)
    feat[f"{PREFIX}diff_healthy_count"] = float(p1_healthy - p2_healthy)
    feat[f"{PREFIX}p1_critical_count"] = float(p1_critical)
    feat[f"{PREFIX}p2_critical_count"] = float(p2_critical)
    feat[f"{PREFIX}diff_critical_count"] = float(p1_critical - p2_critical)

    # HP Standard Deviation across alive Pokémon
    p1_alive_hps = [hp for name, hp in p1_last_hp.items() if name not in p1_fainted]
    p2_alive_hps = [hp for name, hp in p2_last_hp.items() if name not in p2_fainted] + ([1.0] * p2_unrevealed)

    feat[f"{PREFIX}p1_alive_hp_std"] = float(np.std(p1_alive_hps)) if p1_alive_hps else 0.0
    feat[f"{PREFIX}p2_alive_hp_std"] = float(np.std(p2_alive_hps)) if p2_alive_hps else 0.0

    # Hazards
    feat[f"{PREFIX}p1_spikes_set"] = float(p1_spikes_set)
    feat[f"{PREFIX}p2_spikes_set"] = float(p2_spikes_set)
    feat[f"{PREFIX}diff_spikes_control"] = float(p1_spikes_set - p2_spikes_set)
    feat[f"{PREFIX}p1_rapid_spin_count"] = float(p1_rapid_spin_count)
    feat[f"{PREFIX}p2_rapid_spin_count"] = float(p2_rapid_spin_count)
    feat[f"{PREFIX}p1_screens_count"] = float(p1_screens_count)
    feat[f"{PREFIX}p2_screens_count"] = float(p2_screens_count)

    # RestTalk & Curse
    feat[f"{PREFIX}p1_rest_count"] = float(p1_rest_count)
    feat[f"{PREFIX}p2_rest_count"] = float(p2_rest_count)
    feat[f"{PREFIX}diff_rest_count"] = float(p1_rest_count - p2_rest_count)
    feat[f"{PREFIX}p1_sleeptalk_count"] = float(p1_sleeptalk_count)
    feat[f"{PREFIX}p2_sleeptalk_count"] = float(p2_sleeptalk_count)
    feat[f"{PREFIX}p1_curse_count"] = float(p1_curse_count)
    feat[f"{PREFIX}p2_curse_count"] = float(p2_curse_count)
    feat[f"{PREFIX}diff_curse_count"] = float(p1_curse_count - p2_curse_count)

    # Status of ALIVE pokemon only
    p1_alive_par = sum(1 for n, s in p1_last_status.items() if s == "par" and n not in p1_fainted)
    p2_alive_par = sum(1 for n, s in p2_last_status.items() if s == "par" and n not in p2_fainted)
    feat[f"{PREFIX}p1_alive_par_count"] = float(p1_alive_par)
    feat[f"{PREFIX}p2_alive_par_count"] = float(p2_alive_par)
    feat[f"{PREFIX}diff_alive_par_count"] = float(p1_alive_par - p2_alive_par)

    p1_alive_slp = sum(1 for n, s in p1_last_status.items() if s in ("slp", "frz") and n not in p1_fainted)
    p2_alive_slp = sum(1 for n, s in p2_last_status.items() if s in ("slp", "frz") and n not in p2_fainted)
    feat[f"{PREFIX}p1_alive_slp_count"] = float(p1_alive_slp)
    feat[f"{PREFIX}p2_alive_slp_count"] = float(p2_alive_slp)
    feat[f"{PREFIX}diff_alive_slp_count"] = float(p1_alive_slp - p2_alive_slp)

    p1_alive_tox = sum(1 for n, s in p1_last_status.items() if s == "tox" and n not in p1_fainted)
    p2_alive_tox = sum(1 for n, s in p2_last_status.items() if s == "tox" and n not in p2_fainted)
    feat[f"{PREFIX}p1_alive_tox_count"] = float(p1_alive_tox)
    feat[f"{PREFIX}p2_alive_tox_count"] = float(p2_alive_tox)
    feat[f"{PREFIX}diff_alive_tox_count"] = float(p1_alive_tox - p2_alive_tox)

    # Multi-bracket Deltas (Slopes & Trajectory)
    t10_p1 = p1_hp_t10 if p1_hp_t10 is not None else 6.0
    t10_p2 = p2_hp_t10 if p2_hp_t10 is not None else 6.0
    t20_p1 = p1_hp_t20 if p1_hp_t20 is not None else t10_p1
    t20_p2 = p2_hp_t20 if p2_hp_t20 is not None else t10_p2

    delta_0_10_p1 = 6.0 - t10_p1
    delta_0_10_p2 = 6.0 - t10_p2
    delta_10_20_p1 = t10_p1 - t20_p1
    delta_10_20_p2 = t10_p2 - t20_p2
    delta_20_30_p1 = t20_p1 - p1_total_hp
    delta_20_30_p2 = t20_p2 - p2_total_hp

    feat[f"{PREFIX}diff_delta_0_10"] = float(delta_0_10_p2 - delta_0_10_p1)
    feat[f"{PREFIX}diff_delta_10_20"] = float(delta_10_20_p2 - delta_10_20_p1)
    feat[f"{PREFIX}diff_delta_20_30"] = float(delta_20_30_p2 - delta_20_30_p1)

    p1_accel = delta_20_30_p1 - delta_10_20_p1
    p2_accel = delta_20_30_p2 - delta_10_20_p2
    feat[f"{PREFIX}diff_hp_acceleration"] = float(p2_accel - p1_accel)

    feat[f"{PREFIX}p1_late_loss"] = float(delta_20_30_p1)
    feat[f"{PREFIX}p2_late_loss"] = float(delta_20_30_p2)
    feat[f"{PREFIX}diff_late_momentum"] = float(delta_20_30_p2 - delta_20_30_p1)

    feat[f"{PREFIX}p1_switches_t1_10"] = float(p1_switches_t1_10)
    feat[f"{PREFIX}p2_switches_t1_10"] = float(p2_switches_t1_10)
    feat[f"{PREFIX}diff_switches_t1_10"] = float(p1_switches_t1_10 - p2_switches_t1_10)
    feat[f"{PREFIX}p1_switches_t11_20"] = float(p1_switches_t11_20)
    feat[f"{PREFIX}p2_switches_t11_20"] = float(p2_switches_t11_20)
    feat[f"{PREFIX}diff_switches_t11_20"] = float(p1_switches_t11_20 - p2_switches_t11_20)
    feat[f"{PREFIX}p1_switches_late"] = float(p1_switches_late)
    feat[f"{PREFIX}p2_switches_late"] = float(p2_switches_late)
    feat[f"{PREFIX}diff_switches_late"] = float(p1_switches_late - p2_switches_late)

    feat[f"{PREFIX}p2_unrevealed_hp_share"] = float((p2_unrevealed * 1.0) / max(0.01, p2_total_hp))
    feat[f"{PREFIX}p1_first_faint_turn"] = float(p1_first_faint_turn)
    feat[f"{PREFIX}p2_first_faint_turn"] = float(p2_first_faint_turn)
    feat[f"{PREFIX}diff_first_faint_turn"] = float(p2_first_faint_turn - p1_first_faint_turn)

    # =========================================================================
    # 7. ASSE G: ~50 DEEPLY ENGINEERED CROSS-INTERACTION FEATURES
    # =========================================================================
    # A. Sweeper Threat Index & Fragile Sweeper Anti-Blunder (e.g. Battle 7989)
    p1_sweeper_index = (max(0.0, p1_atk_b, p1_spa_b) * (1.0 + max(0.0, p1_spe_b))) if not p1_act_is_fnt else 0.0
    p2_sweeper_index = (max(0.0, p2_atk_b, p2_spa_b) * (1.0 + max(0.0, p2_spe_b))) if not p2_act_is_fnt else 0.0
    feat[f"{PREFIX}p1_sweeper_index"] = float(p1_sweeper_index)
    feat[f"{PREFIX}p2_sweeper_index"] = float(p2_sweeper_index)
    feat[f"{PREFIX}diff_sweeper_index"] = float(p1_sweeper_index - p2_sweeper_index)

    p1_act_eff_sweeper = p1_sweeper_index * p1_hp_t30
    p2_act_eff_sweeper = p2_sweeper_index * p2_hp_t30
    feat[f"{PREFIX}p1_act_effective_sweeper"] = float(p1_act_eff_sweeper)
    feat[f"{PREFIX}p2_act_effective_sweeper"] = float(p2_act_eff_sweeper)
    feat[f"{PREFIX}diff_act_effective_sweeper"] = float(p1_act_eff_sweeper - p2_act_eff_sweeper)

    # B. Defensive Wall Index (Def + SpD boosts)
    p1_wall_index = (max(0.0, p1_def_b) + max(0.0, p1_spd_b)) if not p1_act_is_fnt else 0.0
    p2_wall_index = (max(0.0, p2_def_b) + max(0.0, p2_spd_b)) if not p2_act_is_fnt else 0.0
    feat[f"{PREFIX}p1_wall_index"] = float(p1_wall_index)
    feat[f"{PREFIX}p2_wall_index"] = float(p2_wall_index)
    feat[f"{PREFIX}diff_wall_index"] = float(p1_wall_index - p2_wall_index)

    # C. Spikes Guaranteed Residual Damage (Spikes on field x Grounded Survivors)
    p1_spikes_threat = p1_spikes_set * p2_alive_grounded * 0.125
    p2_spikes_threat = p2_spikes_set * p1_alive_grounded * 0.125
    feat[f"{PREFIX}p1_spikes_residual_threat"] = float(p1_spikes_threat)
    feat[f"{PREFIX}p2_spikes_residual_threat"] = float(p2_spikes_threat)
    feat[f"{PREFIX}diff_spikes_residual_threat"] = float(p1_spikes_threat - p2_spikes_threat)

    # D. Leftovers Recovery Hegemony (Noise eliminated)
    p1_lefties_hp_est = p1_lefties_turns * 0.0625
    feat[f"{PREFIX}p1_leftovers_healed_est"] = float(p1_lefties_hp_est)

    # Anti-Blunder: Quality-Weighted Effective Bulk HP (e.g. Snorlax/Blissey vs glass cannons)
    p1_effective_bulk_hp = 0.0
    for p in p1_team:
        pname = p.get("name", "").strip().lower()
        if pname not in p1_fainted:
            hp_val = p1_last_hp.get(pname, 1.0)
            bulk = float(p.get("base_hp", 80)) * (float(p.get("base_def", 80)) + float(p.get("base_spd", 80))) / 200.0
            p1_effective_bulk_hp += hp_val * bulk

    p2_effective_bulk_hp = 0.0
    for p2_name in p2_seen_names:
        if p2_name not in p2_fainted:
            hp_val = p2_last_hp.get(p2_name, 1.0)
            st = _lookup_stats(p2_name, pokedex)
            bulk = float(st.get("base_hp", 80)) * (float(st.get("base_def", 80)) + float(st.get("base_spd", 80))) / 200.0
            p2_effective_bulk_hp += hp_val * bulk
    p2_effective_bulk_hp += p2_unrevealed * 75.0

    feat[f"{PREFIX}p1_alive_effective_bulk_hp"] = float(p1_effective_bulk_hp)
    feat[f"{PREFIX}p2_alive_effective_bulk_hp"] = float(p2_effective_bulk_hp)
    feat[f"{PREFIX}diff_alive_effective_bulk_hp"] = float(p1_effective_bulk_hp - p2_effective_bulk_hp)
    feat[f"{PREFIX}ratio_alive_effective_bulk_hp"] = float(p1_effective_bulk_hp / max(1.0, p2_effective_bulk_hp))

    # Anti-Blunder: Endgame Snorlax & Curse Threat
    p1_snorlax_alive = 1.0 if ("snorlax" in p1_team_names and "snorlax" not in p1_fainted) else 0.0
    p2_snorlax_alive = 1.0 if ("snorlax" in p2_seen_names and "snorlax" not in p2_fainted) else 0.0
    feat[f"{PREFIX}p1_snorlax_alive"] = float(p1_snorlax_alive)
    feat[f"{PREFIX}p2_snorlax_alive"] = float(p2_snorlax_alive)
    feat[f"{PREFIX}diff_snorlax_alive"] = float(p1_snorlax_alive - p2_snorlax_alive)

    p1_curse_snorlax = 1.0 if (p1_snorlax_alive and p1_curse_count > 0) else 0.0
    p2_curse_snorlax = 1.0 if (p2_snorlax_alive and p2_curse_count > 0) else 0.0
    feat[f"{PREFIX}p1_curse_snorlax"] = float(p1_curse_snorlax)
    feat[f"{PREFIX}p2_curse_snorlax"] = float(p2_curse_snorlax)
    feat[f"{PREFIX}diff_curse_snorlax"] = float(p1_curse_snorlax - p2_curse_snorlax)

    # Anti-Blunder 1: Electric Unresisted Sweep (Jolteon/Raikou/Zapdos alive vs zero ground immunities)
    p1_has_elec = any(e in p1_team_names and e not in p1_fainted and p1_last_hp.get(e, 0.0) >= 0.30 for e in ("jolteon", "raikou", "zapdos"))
    p2_has_ground = any(name not in p2_fainted and "ground" in [t.lower() for t in _lookup_stats(name, pokedex).get("types", [])] for name in p2_seen_names)
    p1_elec_sweep = 0.0
    if p1_has_elec and p2_alive > 0 and not p2_has_ground:
        p1_elec_sweep = 1.0 if p2_unrevealed == 0 else 0.5

    p2_has_elec = any(e in p2_seen_names and e not in p2_fainted and p2_last_hp.get(e, 0.0) >= 0.30 for e in ("jolteon", "raikou", "zapdos"))
    p1_has_ground = any(p.get("name", "").strip().lower() not in p1_fainted and "ground" in [t.lower() for t in p.get("types", [])] for p in p1_team)
    p2_elec_sweep = 1.0 if (p2_has_elec and p1_alive > 0 and not p1_has_ground) else 0.0

    feat[f"{PREFIX}p1_electric_unresisted_sweep"] = float(p1_elec_sweep)
    feat[f"{PREFIX}p2_electric_unresisted_sweep"] = float(p2_elec_sweep)
    feat[f"{PREFIX}diff_electric_unresisted_sweep"] = float(p1_elec_sweep - p2_elec_sweep)

    # Anti-Blunder 2: Cleric Stall & Recovery Potential (Miltank / Blissey alive with Heal Bell / Milk Drink)
    p1_cleric = 1.0 if any(c in p1_team_names and c not in p1_fainted and p1_last_hp.get(c, 0.0) >= 0.35 for c in move_users(frozenset({"healbell"}))) else 0.0
    p2_cleric = 1.0 if any(c in p2_seen_names and c not in p2_fainted and p2_last_hp.get(c, 0.0) >= 0.35 for c in move_users(frozenset({"healbell"}))) else 0.0
    feat[f"{PREFIX}p1_cleric_recovery_potential"] = float(p1_cleric)
    feat[f"{PREFIX}p2_cleric_recovery_potential"] = float(p2_cleric)
    feat[f"{PREFIX}diff_cleric_recovery_potential"] = float(p1_cleric - p2_cleric)

    # Anti-Blunder 3: Thick Club Marowak Raw Power Spike (Gen 2 518 Attack Monster)
    p1_marowak = 1.0 if ("marowak" in p1_team_names and "marowak" not in p1_fainted and p1_last_hp.get("marowak", 0.0) >= 0.25) else 0.0
    p2_marowak = 1.0 if ("marowak" in p2_seen_names and "marowak" not in p2_fainted and p2_last_hp.get("marowak", 0.0) >= 0.25) else 0.0
    feat[f"{PREFIX}p1_thick_club_marowak_alive"] = float(p1_marowak)
    feat[f"{PREFIX}p2_thick_club_marowak_alive"] = float(p2_marowak)
    feat[f"{PREFIX}diff_thick_club_marowak_alive"] = float(p1_marowak - p2_marowak)

    # Anti-Blunder 4: Turn 30 Active Baton Pass Momentum (Late-game stat pass active)
    p1_late_bp_act = p1_late_baton_pass if not p1_act_is_fnt else 0.0
    p2_late_bp_act = p2_late_baton_pass if not p2_act_is_fnt else 0.0
    feat[f"{PREFIX}p1_late_baton_pass_active"] = float(p1_late_bp_act)
    feat[f"{PREFIX}p2_late_baton_pass_active"] = float(p2_late_bp_act)
    feat[f"{PREFIX}diff_late_baton_pass_active"] = float(p1_late_bp_act - p2_late_bp_act)

    # Anti-Blunder 5: Explosion Trade Efficiency (Normalized 1-for-1 KO trade vs wasted explosion)
    feat[f"{PREFIX}p1_explosion_net_efficiency"] = float(p1_effective_explosion)
    feat[f"{PREFIX}p2_explosion_net_efficiency"] = float(p2_effective_explosion)
    feat[f"{PREFIX}diff_explosion_net_efficiency"] = float(p1_effective_explosion - p2_effective_explosion)

    # Viability of surviving Pokémon (from base stats in the data; Smogon ranking removed)
    p1_alive_viability = sum(_bst_viability(p.get("name", ""), pokedex) for p in p1_team if p.get("name", "").strip().lower() not in p1_fainted) / 100.0
    p2_alive_viability = (sum(_bst_viability(n, pokedex) for n in p2_seen_names if n not in p2_fainted) + (p2_unrevealed * DEFAULT_OU_VIABILITY)) / 100.0
    feat[f"{PREFIX}p1_alive_viability"] = float(p1_alive_viability)
    feat[f"{PREFIX}p2_alive_viability"] = float(p2_alive_viability)
    feat[f"{PREFIX}diff_alive_viability"] = float(p1_alive_viability - p2_alive_viability)

    # E. Macro Offensive Dominance Indices
    p1_phys_dom = p1_alive_stats["atk"] / max(1.0, p2_alive_stats["def"])
    p1_spec_dom = p1_alive_stats["spa"] / max(1.0, p2_alive_stats["spd"])
    p2_phys_dom = p2_alive_stats["atk"] / max(1.0, p1_alive_stats["def"])
    p2_spec_dom = p2_alive_stats["spa"] / max(1.0, p1_alive_stats["spd"])

    feat[f"{PREFIX}p1_phys_dominance"] = float(p1_phys_dom)
    feat[f"{PREFIX}p1_spec_dominance"] = float(p1_spec_dom)
    feat[f"{PREFIX}p2_phys_dominance"] = float(p2_phys_dom)
    feat[f"{PREFIX}p2_spec_dominance"] = float(p2_spec_dom)
    feat[f"{PREFIX}net_phys_dominance"] = float(p1_phys_dom - p2_phys_dom)
    feat[f"{PREFIX}net_spec_dominance"] = float(p1_spec_dom - p2_spec_dom)
    feat[f"{PREFIX}net_total_dominance"] = float((p1_phys_dom + p1_spec_dom) - (p2_phys_dom + p2_spec_dom))

    # F. Key Tactical Move Differentials
    feat[f"{PREFIX}p1_toxic_count"] = float(p1_toxic_count)
    feat[f"{PREFIX}p2_toxic_count"] = float(p2_toxic_count)
    feat[f"{PREFIX}diff_toxic_count"] = float(p1_toxic_count - p2_toxic_count)

    feat[f"{PREFIX}p1_twave_count"] = float(p1_twave_count)
    feat[f"{PREFIX}p2_twave_count"] = float(p2_twave_count)
    feat[f"{PREFIX}diff_twave_count"] = float(p1_twave_count - p2_twave_count)

    feat[f"{PREFIX}p1_sleep_moves"] = float(p1_sleep_move_count)
    feat[f"{PREFIX}p2_sleep_moves"] = float(p2_sleep_move_count)
    feat[f"{PREFIX}diff_sleep_moves"] = float(p1_sleep_move_count - p2_sleep_move_count)

    feat[f"{PREFIX}p1_heal_count"] = float(p1_heal_count)
    feat[f"{PREFIX}p2_heal_count"] = float(p2_heal_count)
    feat[f"{PREFIX}diff_heal_count"] = float(p1_heal_count - p2_heal_count)

    feat[f"{PREFIX}p1_substitute_count"] = float(p1_substitute_count)
    feat[f"{PREFIX}p2_substitute_count"] = float(p2_substitute_count)
    feat[f"{PREFIX}diff_substitute_count"] = float(p1_substitute_count - p2_substitute_count)

    feat[f"{PREFIX}p1_explosion_count"] = float(p1_explosion_count)
    feat[f"{PREFIX}p2_explosion_count"] = float(p2_explosion_count)
    feat[f"{PREFIX}diff_explosion_count"] = float(p1_explosion_count - p2_explosion_count)

    feat[f"{PREFIX}p1_phazing_count"] = float(p1_phazing_count)
    feat[f"{PREFIX}p2_phazing_count"] = float(p2_phazing_count)
    feat[f"{PREFIX}diff_phazing_count"] = float(p1_phazing_count - p2_phazing_count)

    feat[f"{PREFIX}p1_belly_drum_count"] = float(p1_belly_drum_count)
    feat[f"{PREFIX}p2_belly_drum_count"] = float(p2_belly_drum_count)
    feat[f"{PREFIX}diff_belly_drum_count"] = float(p1_belly_drum_count - p2_belly_drum_count)

    feat[f"{PREFIX}p1_baton_pass_count"] = float(p1_baton_pass_count)
    feat[f"{PREFIX}p2_baton_pass_count"] = float(p2_baton_pass_count)

    feat[f"{PREFIX}p1_thief_count"] = float(p1_thief_count)
    feat[f"{PREFIX}p2_thief_count"] = float(p2_thief_count)
    feat[f"{PREFIX}diff_thief_count"] = float(p1_thief_count - p2_thief_count)

    # G. P2 Information Revelation & Uncertainty
    p2_revealed_count = len(p2_seen_names)
    feat[f"{PREFIX}p2_revealed_count"] = float(p2_revealed_count)
    feat[f"{PREFIX}p2_revealed_ratio"] = float(p2_revealed_count / 6.0)
    feat[f"{PREFIX}p2_hidden_danger"] = float(p2_unrevealed * (1.0 if p2_alive_total_count > 3 else 0.5))

    # H. Endgame Pressure & Clutch Ratio
    endgame_advantage = (p1_alive - p2_alive_total_count) * (p1_total_hp / max(0.1, p2_total_hp))
    feat[f"{PREFIX}endgame_advantage_index"] = float(endgame_advantage)

    p1_act_hp = 0.0 if (p1_act_is_fnt or p1_act_name in p1_fainted) else float(p1_last_hp.get(p1_act_name, 1.0))
    p2_act_hp = 0.0 if (p2_act_is_fnt or p2_act_name in p2_fainted) else float(p2_last_hp.get(p2_act_name, 1.0))
    feat[f"{PREFIX}active_clutch_ratio"] = float((p1_act_hp + 0.05) / (p2_act_hp + 0.05))
    feat[f"{PREFIX}diff_active_hp_t30"] = float(p1_act_hp - p2_act_hp)

    # =========================================================================
    # 8. ASSE H: 50 SIMPLE & DIRECT FEATURES SUI DATI BASICI
    # =========================================================================
    # 1. Timeline & Action Counts
    feat[f"{PREFIX}p1_moves_executed"] = float(p1_moves_executed)
    feat[f"{PREFIX}p2_moves_executed"] = float(p2_moves_executed)
    feat[f"{PREFIX}diff_moves_executed"] = float(p1_moves_executed - p2_moves_executed)
    feat[f"{PREFIX}p1_move_success_rate"] = float(p1_moves_executed / 30.0)
    feat[f"{PREFIX}p2_move_success_rate"] = float(p2_moves_executed / 30.0)

    feat[f"{PREFIX}p1_second_faint_turn"] = float(p1_second_faint_turn)
    feat[f"{PREFIX}p2_second_faint_turn"] = float(p2_second_faint_turn)
    feat[f"{PREFIX}diff_second_faint_turn"] = float(p2_second_faint_turn - p1_second_faint_turn)

    feat[f"{PREFIX}p1_faint_count_t10"] = float(p1_faint_count_t10)
    feat[f"{PREFIX}p2_faint_count_t10"] = float(p2_faint_count_t10)
    feat[f"{PREFIX}diff_faint_count_t10"] = float(p1_faint_count_t10 - p2_faint_count_t10)
    feat[f"{PREFIX}p1_faint_count_t20"] = float(p1_faint_count_t20)
    feat[f"{PREFIX}p2_faint_count_t20"] = float(p2_faint_count_t20)
    feat[f"{PREFIX}diff_faint_count_t20"] = float(p1_faint_count_t20 - p2_faint_count_t20)

    # 2. Bench HP distributions
    p1_alive_hps_list = [hp for name, hp in p1_last_hp.items() if name not in p1_fainted]
    p2_alive_hps_list = [hp for name, hp in p2_last_hp.items() if name not in p2_fainted] + ([1.0] * p2_unrevealed)

    p1_min_hp = float(min(p1_alive_hps_list)) if p1_alive_hps_list else 0.0
    p1_max_hp = float(max(p1_alive_hps_list)) if p1_alive_hps_list else 0.0
    p1_med_hp = float(np.median(p1_alive_hps_list)) if p1_alive_hps_list else 0.0

    p2_min_hp = float(min(p2_alive_hps_list)) if p2_alive_hps_list else 0.0
    p2_max_hp = float(max(p2_alive_hps_list)) if p2_alive_hps_list else 0.0
    p2_med_hp = float(np.median(p2_alive_hps_list)) if p2_alive_hps_list else 0.0

    feat[f"{PREFIX}p1_alive_min_hp"] = p1_min_hp
    feat[f"{PREFIX}p1_alive_max_hp"] = p1_max_hp
    feat[f"{PREFIX}p1_alive_median_hp"] = p1_med_hp
    feat[f"{PREFIX}p2_alive_min_hp"] = p2_min_hp
    feat[f"{PREFIX}p2_alive_max_hp"] = p2_max_hp
    feat[f"{PREFIX}p2_alive_median_hp"] = p2_med_hp
    feat[f"{PREFIX}diff_alive_min_hp"] = p1_min_hp - p2_min_hp
    feat[f"{PREFIX}diff_alive_max_hp"] = p1_max_hp - p2_max_hp
    feat[f"{PREFIX}diff_alive_median_hp"] = p1_med_hp - p2_med_hp

    # 3. Fresh Reserves & HP Brackets
    p1_fresh = sum(1 for n, hp in p1_last_hp.items() if hp >= 0.99 and n not in p1_fainted)
    p2_fresh = sum(1 for n, hp in p2_last_hp.items() if hp >= 0.99 and n not in p2_fainted) + p2_unrevealed
    feat[f"{PREFIX}p1_fresh_reserves"] = float(p1_fresh)
    feat[f"{PREFIX}p2_fresh_reserves"] = float(p2_fresh)
    feat[f"{PREFIX}diff_fresh_reserves"] = float(p1_fresh - p2_fresh)

    p1_above_half = sum(1 for n, hp in p1_last_hp.items() if hp > 0.50 and n not in p1_fainted)
    p2_above_half = sum(1 for n, hp in p2_last_hp.items() if hp > 0.50 and n not in p2_fainted) + p2_unrevealed
    feat[f"{PREFIX}p1_above_half_count"] = float(p1_above_half)
    feat[f"{PREFIX}p2_above_half_count"] = float(p2_above_half)
    feat[f"{PREFIX}diff_above_half_count"] = float(p1_above_half - p2_above_half)

    p1_near_death = sum(1 for n, hp in p1_last_hp.items() if 0.0 < hp <= 0.20 and n not in p1_fainted)
    p2_near_death = sum(1 for n, hp in p2_last_hp.items() if 0.0 < hp <= 0.20 and n not in p2_fainted)
    feat[f"{PREFIX}p1_near_death_count"] = float(p1_near_death)
    feat[f"{PREFIX}p2_near_death_count"] = float(p2_near_death)
    feat[f"{PREFIX}diff_near_death_count"] = float(p1_near_death - p2_near_death)

    p1_healthy_lead = p1_above_half - p1_near_death
    p2_healthy_lead = p2_above_half - p2_near_death
    feat[f"{PREFIX}p1_healthy_lead"] = float(p1_healthy_lead)
    feat[f"{PREFIX}p2_healthy_lead"] = float(p2_healthy_lead)
    feat[f"{PREFIX}diff_healthy_lead"] = float(p1_healthy_lead - p2_healthy_lead)

    # 4. Big 6 Competitive Elemental Types
    p1_types_all = [t.lower() for p in p1_team for t in p.get("types", []) if t]
    p2_types_all = []
    for name in p2_seen_names:
        st = _lookup_stats(name, pokedex)
        p2_types_all.extend([t.lower() for t in st.get("types", []) if t])

    for elem in ("water", "electric", "normal", "ground", "psychic", "flying"):
        c1 = float(p1_types_all.count(elem))
        c2 = float(p2_types_all.count(elem))
        feat[f"{PREFIX}p1_{elem}_count"] = c1
        feat[f"{PREFIX}p2_seen_{elem}_count"] = c2
        feat[f"{PREFIX}diff_{elem}_count"] = c1 - c2

    # 5. Active Streak on field
    feat[f"{PREFIX}p1_act_streak"] = float(p1_act_streak)
    feat[f"{PREFIX}p2_act_streak"] = float(p2_act_streak)
    feat[f"{PREFIX}diff_act_streak"] = float(p1_act_streak - p2_act_streak)

    # 6. Key Active Stage Boosts
    feat[f"{PREFIX}p1_act_boost_atk_val"] = float(p1_atk_b)
    feat[f"{PREFIX}p1_act_boost_spe_val"] = float(p1_spe_b)

    feat[f"{PREFIX}p2_act_boost_atk_val"] = float(p2_atk_b)
    feat[f"{PREFIX}p2_act_boost_spe_val"] = float(p2_spe_b)

    feat[f"{PREFIX}diff_act_boost_atk"] = float(p1_atk_b - p2_atk_b)
    feat[f"{PREFIX}diff_act_boost_spe"] = float(p1_spe_b - p2_spe_b)

    p1_pos_boosts = sum(max(0.0, b) for b in (p1_atk_b, p1_def_b, p1_spa_b, p1_spd_b, p1_spe_b))
    p2_pos_boosts = sum(max(0.0, b) for b in (p2_atk_b, p2_def_b, p2_spa_b, p2_spd_b, p2_spe_b))
    feat[f"{PREFIX}p1_act_total_pos_boosts"] = float(p1_pos_boosts)
    feat[f"{PREFIX}p2_act_total_pos_boosts"] = float(p2_pos_boosts)
    feat[f"{PREFIX}diff_act_total_pos_boosts"] = float(p1_pos_boosts - p2_pos_boosts)

    p1_neg_boosts = sum(min(0.0, b) for b in (p1_atk_b, p1_def_b, p1_spa_b, p1_spd_b, p1_spe_b))
    p2_neg_boosts = sum(min(0.0, b) for b in (p2_atk_b, p2_def_b, p2_spa_b, p2_spd_b, p2_spe_b))
    feat[f"{PREFIX}p1_act_total_neg_boosts"] = float(p1_neg_boosts)
    feat[f"{PREFIX}p2_act_total_neg_boosts"] = float(p2_neg_boosts)
    feat[f"{PREFIX}diff_act_total_neg_boosts"] = float(p1_neg_boosts - p2_neg_boosts)

    # =========================================================================
    # 9. ASSE I: 30+ COMPLEX STRATEGIC FEATURES ISPIRATE ALLE PIÙ PERFORMANTI
    # =========================================================================
    # 1. Active Speed Dominance over Opponent's Remaining Bench
    p2_alive_speeds = []
    for name in p2_seen_names:
        if name not in p2_fainted and name != p2_act_name:
            st = _lookup_stats(name, pokedex)
            p2_alive_speeds.append(float(st.get("base_spe", 80)))
    if p2_unrevealed > 0:
        p2_alive_speeds.extend([80.0] * p2_unrevealed)

    p1_alive_speeds = []
    for p in p1_team:
        pname = p.get("name", "").strip().lower()
        if pname not in p1_fainted and pname != p1_act_name:
            p1_alive_speeds.append(float(p.get("base_spe", 80)))

    p1_act_faster_ratio = (sum(1 for s in p2_alive_speeds if p1_act_speed_est > s) / max(1, len(p2_alive_speeds))) if not p1_act_is_fnt else 0.0
    p2_act_faster_ratio = (sum(1 for s in p1_alive_speeds if p2_act_speed_est > s) / max(1, len(p1_alive_speeds))) if not p2_act_is_fnt else 0.0
    feat[f"{PREFIX}p1_act_faster_than_p2_bench_ratio"] = float(p1_act_faster_ratio)
    feat[f"{PREFIX}p2_act_faster_than_p1_bench_ratio"] = float(p2_act_faster_ratio)
    feat[f"{PREFIX}diff_act_speed_dominance_vs_bench"] = float(p1_act_faster_ratio - p2_act_faster_ratio)

    # 2. Revenge Killer Contingency (Active Fainted Recovery)
    p1_revenge_ready = 0.0
    if p1_act_is_fnt:
        for p in p1_team:
            pname = p.get("name", "").strip().lower()
            if pname not in p1_fainted:
                b_spe = float(p.get("base_spe", 80))
                b_types = [t.lower() for t in p.get("types", []) if t]
                eff = _calc_max_offensive_eff(b_types, p2_types, chart)
                if b_spe > p2_act_speed_est and eff >= 1.5:
                    p1_revenge_ready = 1.0
                    break

    p2_revenge_ready = 0.0
    if p2_act_is_fnt:
        for name in p2_seen_names:
            if name not in p2_fainted:
                st = _lookup_stats(name, pokedex)
                b_spe = float(st.get("base_spe", 80))
                b_types = [t.lower() for t in st.get("types", []) if t]
                eff = _calc_max_offensive_eff(b_types, p1_types, chart)
                if b_spe > p1_act_speed_est and eff >= 1.5:
                    p2_revenge_ready = 1.0
                    break

    feat[f"{PREFIX}p1_revenge_killer_ready"] = float(p1_revenge_ready)
    feat[f"{PREFIX}p2_revenge_killer_ready"] = float(p2_revenge_ready)
    feat[f"{PREFIX}diff_revenge_killer_ready"] = float(p1_revenge_ready - p2_revenge_ready)

    # 3. Bayesian P2 Unrevealed Danger Distribution
    # Probabilità a priori calcolate dal train (prima: 0.95 / 0.65 / 0.70 scritti a mano)
    spikers = move_users(frozenset({"spikes"}))
    p2_unrev_snorlax_prob = (species_frequency("snorlax") * p2_unrevealed / 6.0) if ("snorlax" not in p2_seen_names) else 0.0
    p2_unrev_elec_prob = (team_share_with_any(("zapdos", "raikou")) * p2_unrevealed / 6.0) if not any(n in p2_seen_names for n in ("zapdos", "raikou")) else 0.0
    p2_unrev_spikes_prob = (team_share_with_any(spikers) * p2_unrevealed / 6.0) if not any(n in p2_seen_names for n in spikers) else 0.0
    p2_total_unrev_danger = p2_unrev_snorlax_prob * 1.5 + p2_unrev_elec_prob * 1.2 + p2_unrev_spikes_prob * 1.0

    feat[f"{PREFIX}p2_unrevealed_snorlax_prob"] = float(p2_unrev_snorlax_prob)
    feat[f"{PREFIX}p2_unrevealed_electric_prob"] = float(p2_unrev_elec_prob)
    feat[f"{PREFIX}p2_total_unrevealed_threat_index"] = float(p2_total_unrev_danger)

    # 4. Phazing Disruption Against Boosted Sweepers
    p1_vuln_phazing = 1.0 if (p1_sweeper_index >= 2.0 and p2_phazing_count > 0 and not p1_act_is_fnt) else 0.0
    p2_vuln_phazing = 1.0 if (p2_sweeper_index >= 2.0 and p1_phazing_count > 0 and not p2_act_is_fnt) else 0.0
    feat[f"{PREFIX}p1_setup_vulnerable_to_phazing"] = float(p1_vuln_phazing)
    feat[f"{PREFIX}p2_setup_vulnerable_to_phazing"] = float(p2_vuln_phazing)
    feat[f"{PREFIX}diff_setup_vulnerable_to_phazing"] = float(p2_vuln_phazing - p1_vuln_phazing)

    # 5. Toxic Acceleration Burnout
    p1_tox_dmg = (min(1.0, (p1_act_toxic_streak + 1) * 0.0625)) if p1_act_is_toxic else 0.0
    p2_tox_dmg = (min(1.0, (p2_act_toxic_streak + 1) * 0.0625)) if p2_act_is_toxic else 0.0
    feat[f"{PREFIX}p1_act_toxic_next_turn_dmg"] = float(p1_tox_dmg)
    feat[f"{PREFIX}p2_act_toxic_next_turn_dmg"] = float(p2_tox_dmg)
    feat[f"{PREFIX}diff_act_toxic_next_turn_dmg"] = float(p2_tox_dmg - p1_tox_dmg)

    # 6. Spikes Grounded Exposure Differential
    p1_switches_total = p1_switches_t1_10 + p1_switches_t11_20 + p1_switches_late
    p2_switches_total = p2_switches_t1_10 + p2_switches_t11_20 + p2_switches_late
    p1_spikes_fric = p2_spikes_set * p1_alive_grounded * (p1_switches_total / 30.0)
    p2_spikes_fric = p1_spikes_set * p2_alive_grounded * (p2_switches_total / 30.0)
    feat[f"{PREFIX}p1_grounded_spikes_exposure"] = float(p1_spikes_fric)
    feat[f"{PREFIX}p2_grounded_spikes_exposure"] = float(p2_spikes_fric)
    feat[f"{PREFIX}diff_grounded_spikes_exposure"] = float(p2_spikes_fric - p1_spikes_fric)

    # 7. Paralysis Speed Inversion
    p1_par_inv = 1.0 if (p1_base_spe > p2_base_spe and p1_act_speed_est < p2_act_speed_est and not p1_act_is_fnt) else 0.0
    p2_par_inv = 1.0 if (p2_base_spe > p1_base_spe and p2_act_speed_est < p1_act_speed_est and not p2_act_is_fnt) else 0.0
    feat[f"{PREFIX}p1_par_inversion"] = float(p1_par_inv)
    feat[f"{PREFIX}p2_par_inversion"] = float(p2_par_inv)
    feat[f"{PREFIX}diff_par_inversion"] = float(p2_par_inv - p1_par_inv)

    # 8. NEW: Speed Monopoly (Taking into account Gen 2 75% Paralysis speed penalty)
    p1_alive_eff_speeds = []
    for p in p1_team:
        pname = p.get("name", "").strip().lower()
        if pname not in p1_fainted:
            b_spe = float(p.get("base_spe", 80))
            par_p = 0.25 if (p1_last_status.get(pname) == "par") else 1.0
            boost_p = STAGE_MULT.get(int(max(-6, min(6, p1_spe_b))), 1.0) if (pname == p1_act_name and not p1_act_is_fnt) else 1.0
            p1_alive_eff_speeds.append(b_spe * par_p * boost_p)

    p2_alive_eff_speeds = []
    for name in p2_seen_names:
        if name not in p2_fainted:
            st = _lookup_stats(name, pokedex)
            b_spe = float(st.get("base_spe", 80))
            par_p = 0.25 if (p2_last_status.get(name) == "par") else 1.0
            boost_p = STAGE_MULT.get(int(max(-6, min(6, p2_spe_b))), 1.0) if (name == p2_act_name and not p2_act_is_fnt) else 1.0
            p2_alive_eff_speeds.append(b_spe * par_p * boost_p)
    if p2_unrevealed > 0:
        p2_alive_eff_speeds.append(85.0)

    p1_max_spe = max(p1_alive_eff_speeds) if p1_alive_eff_speeds else 0.0
    p2_max_spe = max(p2_alive_eff_speeds) if p2_alive_eff_speeds else 0.0
    feat[f"{PREFIX}p1_max_alive_speed"] = float(p1_max_spe / 100.0)
    feat[f"{PREFIX}p2_max_alive_speed"] = float(p2_max_spe / 100.0)
    feat[f"{PREFIX}diff_max_alive_speed"] = float((p1_max_spe - p2_max_spe) / 100.0)
    p1_spe_monopoly = 1.0 if (p1_max_spe >= p2_max_spe + 15.0) else 0.0
    p2_spe_monopoly = 1.0 if (p2_max_spe >= p1_max_spe + 15.0) else 0.0
    feat[f"{PREFIX}diff_speed_monopoly"] = float(p1_spe_monopoly - p2_spe_monopoly)

    # 9. NEW: Defensive Stalling Wall Lock (Skarmory / Steelix without super-effective answers)
    p1_has_skarmory = ("skarmory" in p1_team_names and "skarmory" not in p1_fainted and p1_last_hp.get("skarmory", 0.0) >= 0.35)
    p2_has_skarmory = ("skarmory" in p2_seen_names and "skarmory" not in p2_fainted and p2_last_hp.get("skarmory", 0.0) >= 0.35)
    p2_has_fire_elec = any("fire" in [t.lower() for t in _lookup_stats(n, pokedex).get("types", [])] or "electric" in [t.lower() for t in _lookup_stats(n, pokedex).get("types", [])] for n in p2_seen_names if n not in p2_fainted)
    p1_has_fire_elec = any("fire" in [t.lower() for t in p.get("types", [])] or "electric" in [t.lower() for t in p.get("types", [])] for p in p1_team if p.get("name", "").strip().lower() not in p1_fainted)
    p1_wall_lock = 1.0 if (p1_has_skarmory and not p2_has_fire_elec and p2_unrevealed == 0) else (0.5 if (p1_has_skarmory and not p2_has_fire_elec) else 0.0)
    p2_wall_lock = 1.0 if (p2_has_skarmory and not p1_has_fire_elec) else 0.0
    feat[f"{PREFIX}p1_defensive_wall_lock"] = float(p1_wall_lock)
    feat[f"{PREFIX}p2_defensive_wall_lock"] = float(p2_wall_lock)
    feat[f"{PREFIX}diff_defensive_wall_lock"] = float(p1_wall_lock - p2_wall_lock)

    # 10. NEW: Permanent Spikes vs Dead Rapid Spinner (Cloyster, Forretress, Starmie)
    spinners = move_users(frozenset({"rapidspin"}))  # dal train (prima: lista scritta a mano)
    p1_spinners_alive = sum(1 for p in p1_team if p.get("name", "").strip().lower() in spinners and p.get("name", "").strip().lower() not in p1_fainted)
    p2_spinners_alive = sum(1 for n in p2_seen_names if n in spinners and n not in p2_fainted)
    p1_perm_spikes = 1.0 if (p1_spikes_set and p2_spinners_alive == 0 and p2_unrevealed == 0) else (0.5 if (p1_spikes_set and p2_spinners_alive == 0) else 0.0)
    p2_perm_spikes = 1.0 if (p2_spikes_set and p1_spinners_alive == 0) else 0.0
    feat[f"{PREFIX}p1_permanent_spikes_active"] = float(p1_perm_spikes)
    feat[f"{PREFIX}p2_permanent_spikes_active"] = float(p2_perm_spikes)
    feat[f"{PREFIX}diff_permanent_spikes_active"] = float(p1_perm_spikes - p2_perm_spikes)

    # 11. NEW: Phazing Defense vs Opponent Setup Boosts (Roar / Whirlwind wipes stat stages)
    phazers = move_users(frozenset({"roar", "whirlwind"}))  # dal train (prima: lista scritta a mano)
    p1_has_phazer = any((p.get("name", "").strip().lower() in phazers or p1_phazing_count > 0) and p.get("name", "").strip().lower() not in p1_fainted for p in p1_team)
    p2_has_phazer = any((n in phazers or p2_phazing_count > 0) and n not in p2_fainted for n in p2_seen_names)
    p1_phazing_denial = 1.0 if (p2_pos_boosts >= 2.0 and p1_has_phazer) else 0.0
    p2_phazing_denial = 1.0 if (p1_pos_boosts >= 2.0 and p2_has_phazer) else 0.0
    feat[f"{PREFIX}p1_phazing_denial_ready"] = float(p1_phazing_denial)
    feat[f"{PREFIX}p2_phazing_denial_ready"] = float(p2_phazing_denial)
    feat[f"{PREFIX}diff_phazing_denial_ready"] = float(p1_phazing_denial - p2_phazing_denial)

    # 12. Macro Offensive Pressure vs Opponent Effective Bulk
    p1_alive_firepower = sum(max(float(p.get("base_atk", 80)), float(p.get("base_spa", 80))) for p in p1_team if p.get("name", "").strip().lower() not in p1_fainted)
    p2_alive_firepower = sum(max(float(stats.get("base_atk", 80)), float(stats.get("base_spa", 80))) for name in p2_seen_names if name not in p2_fainted for stats in [_lookup_stats(name, pokedex)]) + (p2_unrevealed * 85.0)

    p1_macro_press = p1_alive_firepower / max(1.0, p2_effective_bulk_hp)
    p2_macro_press = p2_alive_firepower / max(1.0, p1_effective_bulk_hp)
    feat[f"{PREFIX}p1_macro_pressure_ratio"] = float(p1_macro_press)
    feat[f"{PREFIX}p2_macro_pressure_ratio"] = float(p2_macro_press)
    feat[f"{PREFIX}diff_macro_pressure_ratio"] = float(p1_macro_press - p2_macro_press)

    # 12. Defensive Blindspot Exploitation by Opponent
    p1_blindspots_types = [t for t in GEN2_TYPES if sum(1 for p in p1_team if _defensive_multiplier(t, p.get("types", []), chart) > 1.0) >= 3]
    p2_exploit_count = 0.0
    for name in p2_seen_names:
        st = _lookup_stats(name, pokedex)
        types_p2 = [t.lower() for t in st.get("types", []) if t]
        if any(t in p1_blindspots_types for t in types_p2):
            p2_exploit_count += 1.0
    feat[f"{PREFIX}p2_exploits_p1_blindspots_count"] = float(p2_exploit_count)
    feat[f"{PREFIX}p2_exploits_p1_blindspots_ratio"] = float(p2_exploit_count / max(1.0, float(len(p2_seen_names))))

    # 13. Alive Team Mean Speed Differentials
    p1_mean_spe = p1_alive_stats["spe"] / max(1, p1_alive)
    p2_mean_spe = p2_alive_stats["spe"] / max(1, p2_alive_total_count)
    feat[f"{PREFIX}p1_alive_mean_speed"] = float(p1_mean_spe)
    feat[f"{PREFIX}p2_alive_mean_speed"] = float(p2_mean_spe)
    feat[f"{PREFIX}diff_alive_mean_speed"] = float(p1_mean_spe - p2_mean_spe)
    feat[f"{PREFIX}ratio_alive_mean_speed"] = float(p1_mean_spe / max(1.0, p2_mean_spe))

    # 14. Active HP vs Opponent Average Bench HP
    p1_act_hp_vs_p2_bench = p1_act_hp - (p2_total_hp / max(1, p2_alive_total_count))
    p2_act_hp_vs_p1_bench = p2_act_hp - (p1_total_hp / max(1, p1_alive))
    feat[f"{PREFIX}p1_act_hp_vs_p2_bench"] = float(p1_act_hp_vs_p2_bench)
    feat[f"{PREFIX}p2_act_hp_vs_p1_bench"] = float(p2_act_hp_vs_p1_bench)
    feat[f"{PREFIX}diff_act_hp_vs_bench"] = float(p1_act_hp_vs_p2_bench - p2_act_hp_vs_p1_bench)

    # 15. Alive Team Bulk Balance (Physical vs Special)
    p1_phys_vs_spec_bulk = p1_alive_phys_sum / max(1.0, p1_alive_spec_sum)
    p2_phys_vs_spec_bulk = p2_alive_phys_sum / max(1.0, p2_alive_spec_sum)
    feat[f"{PREFIX}p1_team_phys_vs_spec_bulk"] = float(p1_phys_vs_spec_bulk)
    feat[f"{PREFIX}p2_team_phys_vs_spec_bulk"] = float(p2_phys_vs_spec_bulk)
    feat[f"{PREFIX}diff_team_phys_vs_spec_bulk"] = float(p1_phys_vs_spec_bulk - p2_phys_vs_spec_bulk)

    return feat
