"""Bipartite Duel Matrix & Non-Greedy Physical Mechanics v2 (Gen 2 GSC).

Upgrades:
1. Item Multipliers in OffStat_m(i): Light Ball Pikachu (2x SpA -> 396), Thick Club Marowak (2x Atk -> 516).
2. Infinite Stall Lock & Active Recovery: 50% heals, Rest + Curse snowball loop in rollout,
   and rf_diff_infinite_stall_locks (threshold < 0.31).
3. Latent Detonation Value: Explosion/Self-Destruct Def-halving (500 BP eff) via rf_diff_latent_boom_leverage.
4. Empirical Move Coverage Priors: P(m|s) computed from train.jsonl only (src/features/data_priors.py) integrated into d_hat.
5. Proactive Switch at Turn 31 in Rollout.
6. Non-Linear Regime-Switching Features:
   - rf_comeback_override_signal
   - rf_throw_collapse_signal
   - rf_anti_meta_viability_inversion
   - rf_post_t30_bench_vs_active_delta

Prefix: rf_
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.features.data_priors import move_priors, move_users, species_frequency, top_species
from src.config import DATA_DIR, TYPE_CHART_CSV

PREFIX = "rf_"

PHYSICAL_TYPES: Set[str] = {
    "normal", "fighting", "flying", "poison", "ground",
    "rock", "bug", "ghost", "steel"
}
SPECIAL_TYPES: Set[str] = {
    "fire", "water", "grass", "electric", "psychic",
    "ice", "dragon", "dark"
}

STAGE_MULT: Dict[int, float] = {
    -6: 0.25, -5: 0.28, -4: 0.33, -3: 0.40, -2: 0.50, -1: 0.66,
    0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0
}

# Pulizia 04/11 (persona D): le liste scritte a mano (chi usa Explosion, i 20 Pokémon più usati in OU
# con le loro frequenze d'uso) sono state sostituite da statistiche calcolate dal solo train.jsonl
# (src/features/data_priors.py): il regolamento vieta usage statistics e tabelle raccolte a mano.
BOOM_SPECIES: FrozenSet[str] = move_users(frozenset({"explosion", "selfdestruct"}))
TOP_OU_K = 20  # specie più frequenti nelle squadre di P1 del train


POKEDEX_JSON_PATH = DATA_DIR / "pokedex.json"
_POKEDEX: Optional[Dict[str, Dict[str, Any]]] = None
_TYPE_CHART: Optional[Dict[Tuple[str, str], float]] = None
_MOVE_PRIORS: Optional[Dict[str, List[Dict[str, Any]]]] = None


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


_TOP_OU: Optional[List[Dict[str, Any]]] = None


def _top_ou_priors() -> List[Dict[str, Any]]:
    """Le TOP_OU_K specie più frequenti nel train, con statistiche dal Pokédex (costruito dai dati)."""
    global _TOP_OU
    if _TOP_OU is None:
        _TOP_OU = [{"name": n, "spe": float(_get_stats(n).get("base_spe", 80.0)), "freq": species_frequency(n)}
                   for n in top_species(TOP_OU_K)]
    return _TOP_OU


def _load_move_priors() -> Dict[str, List[Dict[str, Any]]]:
    """P(mossa | specie) calcolata dal solo train (prima veniva da data/empirical_move_priors.json)."""
    return move_priors()


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


def _calc_damage_fraction(
    attacker: Dict[str, Any],
    defender: Dict[str, Any],
    is_explosion: bool = False,
) -> float:
    """Calculates max single-hit damage fraction d_hat(i -> j) as % of j's max Lv.100 HP."""
    hp_lv100_j = defender["hp_lv100"]
    max_frac = 0.001

    i_types = attacker["types"]
    j_types = defender["types"]

    moves_to_eval = attacker["moves"]
    if is_explosion:
        moves_to_eval = [{"name": "explosion", "power": 250.0, "type": "normal"}]

    for move in moves_to_eval:
        pwr = move.get("power", 75.0)
        m_type = move.get("type", "normal")
        is_phys = m_type in PHYSICAL_TYPES

        if is_phys:
            off_stat = attacker["atk_eff"]
            def_stat = defender["def_eff"] * (0.5 if move.get("name") in ("explosion", "selfdestruct") else 1.0)
        else:
            off_stat = attacker["spa_eff"]
            def_stat = defender["spd_eff"]

        stab = 1.5 if m_type in i_types else 1.0
        type_mult = _defensive_multiplier(m_type, j_types)

        if type_mult <= 0.0001:
            continue

        raw_dmg = ((0.84 * pwr * (off_stat / max(def_stat, 1.0)) + 2.0) * stab * type_mult)
        frac = raw_dmg / max(hp_lv100_j, 1.0)
        prob_weight = float(move.get("prior_weight", 1.0))
        weighted_frac = frac * prob_weight
        if weighted_frac > max_frac:
            max_frac = weighted_frac

    return max_frac


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract 24 bipartite physical duel features for Gen 2 (v2 with non-greedy mechanics)."""
    feat: Dict[str, float] = {}

    p1_team_details = data.get("p1_team_details", [])
    p1_team_names = [p.get("name", f"p1_{i}").strip().lower() for i, p in enumerate(p1_team_details)]
    p2_lead_name = data.get("p2_lead_details", {}).get("name", "p2_lead").strip().lower()

    timeline = data.get("battle_timeline", [])

    p1_hp: Dict[str, float] = {name: 1.0 for name in p1_team_names}
    p1_status: Dict[str, str] = {name: "nostatus" for name in p1_team_names}
    p1_boosts: Dict[str, Dict[str, int]] = {name: {} for name in p1_team_names}
    p1_fainted: Set[str] = set()
    p1_moves_seen: Dict[str, List[Dict[str, Any]]] = {name: [] for name in p1_team_names}
    p1_items: Dict[str, str] = {p.get("name", "").strip().lower(): str(p.get("item", "")).strip().lower() for p in p1_team_details}

    p2_hp: Dict[str, float] = {p2_lead_name: 1.0}
    p2_status: Dict[str, str] = {p2_lead_name: "nostatus"}
    p2_boosts: Dict[str, Dict[str, int]] = {p2_lead_name: {}}
    p2_fainted: Set[str] = set()
    p2_seen_names: Set[str] = {p2_lead_name}
    p2_moves_seen: Dict[str, List[Dict[str, Any]]] = {p2_lead_name: []}

    p1_active_t30 = ""
    p2_active_t30 = ""
    p1_act_hp_t30 = 1.0
    p2_act_hp_t30 = 1.0
    spikes_p1 = False
    spikes_p2 = False

    for turn in timeline:
        p1_st = turn.get("p1_pokemon_state", {})
        p2_st = turn.get("p2_pokemon_state", {})

        p1_name = p1_st.get("name", "").strip().lower()
        p2_name = p2_st.get("name", "").strip().lower()

        if p1_name:
            p1_active_t30 = p1_name
            h = float(p1_st.get("hp_pct", 1.0))
            p1_hp[p1_name] = h
            p1_act_hp_t30 = h
            st = p1_st.get("status", "nostatus")
            p1_status[p1_name] = st
            p1_boosts[p1_name] = p1_st.get("boosts", {})
            if h <= 0.001 or st == "fnt":
                p1_fainted.add(p1_name)
                p1_hp[p1_name] = 0.0

        if p2_name:
            p2_active_t30 = p2_name
            p2_seen_names.add(p2_name)
            if p2_name not in p2_moves_seen:
                p2_moves_seen[p2_name] = []
                p2_boosts[p2_name] = {}
            h = float(p2_st.get("hp_pct", 1.0))
            p2_hp[p2_name] = h
            p2_act_hp_t30 = h
            st = p2_st.get("status", "nostatus")
            p2_status[p2_name] = st
            p2_boosts[p2_name] = p2_st.get("boosts", {})
            if h <= 0.001 or st == "fnt":
                p2_fainted.add(p2_name)
                p2_hp[p2_name] = 0.0

        p1_mv = turn.get("p1_move_details")
        if p1_mv and p1_name:
            mv_name = str(p1_mv.get("name", "")).strip().lower()
            if mv_name == "spikes":
                spikes_p2 = True
            bp = float(p1_mv.get("base_power", 0) or 0)
            mv_t = str(p1_mv.get("type", "normal")).strip().lower()
            p1_moves_seen[p1_name].append({"name": mv_name, "power": bp if bp > 0 else 75.0, "type": mv_t, "prior_weight": 1.0})

        p2_mv = turn.get("p2_move_details")
        if p2_mv and p2_name:
            mv_name = str(p2_mv.get("name", "")).strip().lower()
            if mv_name == "spikes":
                spikes_p1 = True
            bp = float(p2_mv.get("base_power", 0) or 0)
            mv_t = str(p2_mv.get("type", "normal")).strip().lower()
            p2_moves_seen[p2_name].append({"name": mv_name, "power": bp if bp > 0 else 75.0, "type": mv_t, "prior_weight": 1.0})

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

    move_priors_db = _load_move_priors()

    # Build structured combat records with Item Multipliers & Move Priors
    def build_combat_record(name: str, hp_pct: float, status: str, boosts: Dict[str, int], moves_seen: List[Dict[str, Any]], item: str = "") -> Dict[str, Any]:
        st = _get_stats(name)
        types = st.get("types", ["normal"])
        base_hp = float(st.get("base_hp", 80.0))
        base_atk = float(st.get("base_atk", 80.0))
        base_def = float(st.get("base_def", 80.0))
        base_spa = float(st.get("base_spa", 80.0))
        base_spd = float(st.get("base_spd", 80.0))
        base_spe = float(st.get("base_spe", 80.0))

        hp_lv100 = 2.0 * base_hp + 203.0
        atk_lv100 = 2.0 * base_atk + 98.0
        def_lv100 = 2.0 * base_def + 98.0
        spa_lv100 = 2.0 * base_spa + 98.0
        spd_lv100 = 2.0 * base_spd + 98.0
        spe_lv100 = 2.0 * base_spe + 98.0

        # Pillar 1.1: Item Stat Multipliers (Light Ball Pikachu, Thick Club Marowak)
        norm_name = name.strip().lower()
        # Only when the item is in the data (P1); P2 items are unknown, so no assumption is made
        if norm_name == "pikachu" and item == "lightball":
            spa_lv100 *= 2.0  # Light Ball doubles SpA
        elif norm_name == "marowak" and item == "thickclub":
            atk_lv100 *= 2.0  # Thick Club doubles Atk

        spe_eff = spe_lv100 * STAGE_MULT.get(boosts.get("spe", 0), 1.0) * (0.25 if status == "par" else 1.0)
        atk_eff = atk_lv100 * STAGE_MULT.get(boosts.get("atk", 0), 1.0)
        def_eff = def_lv100 * STAGE_MULT.get(boosts.get("def", 0), 1.0)
        spa_eff = spa_lv100 * STAGE_MULT.get(boosts.get("spa", 0), 1.0)
        spd_eff = spd_lv100 * STAGE_MULT.get(boosts.get("spd", 0), 1.0)

        # Build offensive vector M(i)
        m_list = list(moves_seen)
        seen_names = {m.get("name") for m in moves_seen}

        # Add native STAB moves with standard power
        for t in types:
            if t and t != "notype":
                m_list.append({"name": f"stab_{t}", "power": 90.0, "type": t, "prior_weight": 1.0})

        # Pillar 1.4: Empirical Move Coverage Priors
        if norm_name in move_priors_db:
            for cand in move_priors_db[norm_name]:
                if cand["name"] not in seen_names and cand.get("power", 0) > 0:
                    prob = float(cand.get("prob", 0.1))
                    if prob >= 0.10:  # Only significant meta moves
                        m_list.append({
                            "name": cand["name"],
                            "power": cand["power"],
                            "type": cand["type"],
                            "prior_weight": min(1.0, prob * 1.2),  # Expected value weight
                        })

        has_leftovers = (item == "leftovers" or item == "")
        is_grounded = "flying" not in types

        # Check recovery & boom traits
        has_instant_heal = any(m.get("name") in ("milkdrink", "softboiled", "recover") for m in m_list)
        has_rest = any(m.get("name") == "rest" for m in m_list)
        has_curse = any(m.get("name") == "curse" for m in m_list)
        is_boom_user = (norm_name in BOOM_SPECIES)

        return {
            "name": name,
            "hp_pct": hp_pct,
            "status": status,
            "types": types,
            "hp_lv100": hp_lv100,
            "spe_eff": spe_eff,
            "atk_eff": atk_eff,
            "def_eff": def_eff,
            "spa_eff": spa_eff,
            "spd_eff": spd_eff,
            "moves": m_list,
            "has_leftovers": has_leftovers,
            "is_grounded": is_grounded,
            "has_instant_heal": has_instant_heal,
            "has_rest": has_rest,
            "has_curse": has_curse,
            "is_boom_user": is_boom_user,
        }

    p1_alive = [build_combat_record(n, p1_hp[n], p1_status[n], p1_boosts.get(n, {}), p1_moves_seen.get(n, []), p1_items.get(n, "")) for n in p1_alive_names]
    p2_alive = [build_combat_record(n, p2_hp[n], p2_status[n], p2_boosts.get(n, {}), p2_moves_seen.get(n, [])) for n in p2_alive_names]

    p2_unrevealed = max(0, 6 - len(p2_seen_names))
    if p2_unrevealed > 0:
        # Stand-in for unrevealed P2 Pokémon: the most frequent species in the training teams, with its
        # most used attacking move in the training battles (before: Snorlax + Body Slam written by hand)
        sur_name = top_species(1)[0]
        sur_move = next((m for m in move_priors().get(sur_name, []) if m["power"] > 0), {"name": "stab", "power": 80.0, "type": "normal"})
        surrogate = build_combat_record(sur_name, 1.0, "nostatus", {}, [{"name": sur_move["name"], "power": sur_move["power"], "type": sur_move["type"]}])
        p2_alive.append(surrogate)

    if not p1_alive:
        p1_alive = [build_combat_record("fainted_placeholder", 0.0, "fnt", {}, [])]
    if not p2_alive:
        p2_alive = [build_combat_record("fainted_placeholder", 0.0, "fnt", {}, [])]

    # Precalculate Bipartite Damage Matrix
    D1 = np.zeros((len(p1_alive), len(p2_alive)))
    D2 = np.zeros((len(p2_alive), len(p1_alive)))

    for i, p1_mon in enumerate(p1_alive):
        for j, p2_mon in enumerate(p2_alive):
            D1[i, j] = _calc_damage_fraction(p1_mon, p2_mon)
            D2[j, i] = _calc_damage_fraction(p2_mon, p1_mon)

    # -------------------------------------------------------------------------
    # 1. Sweep Chain
    # -------------------------------------------------------------------------
    def calc_sweep_chain(D_matrix, D_opp, my_team, opp_team):
        max_chain = 0
        for i, att in enumerate(my_team):
            chain = 0
            for j, dfn in enumerate(opp_team):
                outspeed = att["spe_eff"] > dfn["spe_eff"]
                dmg = D_matrix[i, j]
                opp_dmg = D_opp[j, i]
                hp_target = dfn["hp_pct"]
                hp_me = att["hp_pct"]
                if outspeed and (dmg >= hp_target or (dmg >= 0.5 * hp_target and opp_dmg < hp_me)):
                    chain += 1
            if chain > max_chain:
                max_chain = chain
        return float(max_chain)

    p1_chain = calc_sweep_chain(D1, D2, p1_alive, p2_alive)
    p2_chain = calc_sweep_chain(D2, D1, p2_alive, p1_alive)
    feat[f"{PREFIX}diff_uncontested_sweep_chain"] = p1_chain - p2_chain

    # -------------------------------------------------------------------------
    # 2. Paralysis Lethal Inversion
    # -------------------------------------------------------------------------
    def calc_par_penalty(team, opp_team):
        penalty = 0.0
        for i, mon in enumerate(team):
            if mon["status"] == "par":
                spe_unpar = (2.0 * _get_stats(mon["name"]).get("base_spe", 80.0) + 98.0)
                spe_par = mon["spe_eff"]
                for j, opp in enumerate(opp_team):
                    if spe_unpar > opp["spe_eff"] and spe_par < opp["spe_eff"]:
                        max_mult = max([_defensive_multiplier(m.get("type", "normal"), mon["types"]) for m in opp["moves"]] or [1.0])
                        penalty += max_mult * mon["hp_pct"]
        return penalty

    pen_p1 = calc_par_penalty(p1_alive, p2_alive)
    pen_p2 = calc_par_penalty(p2_alive, p1_alive)
    feat[f"{PREFIX}diff_paralysis_lethal_inversion_score"] = float(pen_p2 - pen_p1)

    # -------------------------------------------------------------------------
    # 3. Active Survive & Counter OHKO Margin
    # -------------------------------------------------------------------------
    counter_margin = 0.0
    if p1_active_t30 and p2_active_t30 and p1_act_hp_t30 > 0.001 and p2_act_hp_t30 > 0.001:
        idx1 = next((i for i, m in enumerate(p1_alive) if m["name"] == p1_active_t30), None)
        idx2 = next((j for j, m in enumerate(p2_alive) if m["name"] == p2_active_t30), None)
        if idx1 is not None and idx2 is not None:
            mon1 = p1_alive[idx1]
            mon2 = p2_alive[idx2]
            d12 = D1[idx1, idx2]
            d21 = D2[idx2, idx1]
            if mon1["spe_eff"] > mon2["spe_eff"]:
                s_p2 = mon2["hp_pct"] - d12
                if s_p2 > 0 and d21 >= mon1["hp_pct"]:
                    counter_margin = -float(s_p2)
            else:
                s_p1 = mon1["hp_pct"] - d21
                if s_p1 > 0 and d12 >= mon2["hp_pct"]:
                    counter_margin = float(s_p1)
    feat[f"{PREFIX}active_survive_and_counter_ohko_margin"] = counter_margin

    # -------------------------------------------------------------------------
    # 4. Latent Boom Leverage (Explosion / Self-Destruct Gen 2)
    # -------------------------------------------------------------------------
    def calc_boom_leverage(att_team, def_team):
        max_lev = 0.0
        for i, att in enumerate(att_team):
            if att["is_boom_user"] and att["hp_pct"] <= 0.45:
                # Calculate boom damage on each defender
                for j, dfn in enumerate(def_team):
                    d_boom = _calc_damage_fraction(att, dfn, is_explosion=True)
                    lev = min(dfn["hp_pct"], d_boom) - att["hp_pct"]
                    if lev > max_lev:
                        max_lev = lev
        return max_lev

    boom_p1 = calc_boom_leverage(p1_alive, p2_alive)
    boom_p2 = calc_boom_leverage(p2_alive, p1_alive)
    feat[f"{PREFIX}diff_latent_boom_leverage"] = float(boom_p1 - boom_p2)

    # -------------------------------------------------------------------------
    # 5. Infinite Stall Locks (Threshold < 0.31)
    # -------------------------------------------------------------------------
    def count_infinite_stall_locks(def_team, D_incoming):
        locks = 0
        for j, dfn in enumerate(def_team):
            if dfn["has_rest"] or dfn["has_instant_heal"]:
                max_in = float(np.max(D_incoming[:, j])) if D_incoming.shape[0] > 0 else 0.0
                if max_in < 0.31:  # Rest + Leftovers 3-turn unkillable wall
                    locks += 1
        return float(locks)

    stall_p1 = count_infinite_stall_locks(p1_alive, D2)
    stall_p2 = count_infinite_stall_locks(p2_alive, D1)
    feat[f"{PREFIX}diff_infinite_stall_locks"] = stall_p1 - stall_p2

    # -------------------------------------------------------------------------
    # 6. Asymmetric Breach & Minimax Saddle Duel Value
    # -------------------------------------------------------------------------
    def calc_asymmetric_breach(att_team, def_team):
        breaches = []
        for dfn in def_team:
            hp_cur = max(dfn["hp_pct"], 0.05) * dfn["hp_lv100"]
            b_phys = hp_cur * dfn["def_eff"]
            b_spec = hp_cur * dfn["spd_eff"]

            max_b_ratio = 0.0
            for att in att_team:
                for m in att["moves"]:
                    m_type = m.get("type", "normal")
                    is_phys = m_type in PHYSICAL_TYPES
                    b_axis = b_phys if is_phys else b_spec
                    off_stat = att["atk_eff"] if is_phys else att["spa_eff"]
                    stab = 1.5 if m_type in att["types"] else 1.0
                    t_mult = _defensive_multiplier(m_type, dfn["types"])
                    ratio = (off_stat * stab * t_mult) / max(b_axis, 10.0)
                    if ratio > max_b_ratio:
                        max_b_ratio = ratio
            breaches.append(max_b_ratio)
        return float(np.mean(breaches)) if breaches else 0.0

    p1_breach = calc_asymmetric_breach(p1_alive, p2_alive)
    p2_breach = calc_asymmetric_breach(p2_alive, p1_alive)
    feat[f"{PREFIX}p1_asymmetric_phys_spec_breach"] = p1_breach
    feat[f"{PREFIX}p2_asymmetric_phys_spec_breach"] = p2_breach
    feat[f"{PREFIX}diff_asymmetric_phys_spec_breach"] = p1_breach - p2_breach

    # Minimax Duel Matrix
    TTK_1 = np.zeros((len(p1_alive), len(p2_alive)))
    TTK_2 = np.zeros((len(p2_alive), len(p1_alive)))

    for i, p1_mon in enumerate(p1_alive):
        for j, p2_mon in enumerate(p2_alive):
            net_d1 = max(D1[i, j] - 0.0625, 0.01)
            net_d2 = max(D2[j, i] - 0.0625, 0.01)

            t1 = math.ceil(max(p2_mon["hp_pct"], 0.05) / net_d1) - (0.5 if p1_mon["spe_eff"] > p2_mon["spe_eff"] else 0.0)
            t2 = math.ceil(max(p1_mon["hp_pct"], 0.05) / net_d2) - (0.5 if p2_mon["spe_eff"] > p1_mon["spe_eff"] else 0.0)

            TTK_1[i, j] = max(t1, 0.5)
            TTK_2[j, i] = max(t2, 0.5)

    V_12 = np.log2(np.clip(TTK_2.T / TTK_1, 0.0625, 16.0))
    V_21 = np.log2(np.clip(TTK_1 / TTK_2.T, 0.0625, 16.0))

    p1_saddle = float(np.mean(np.max(V_12, axis=0)))
    p2_saddle = float(np.mean(np.max(V_21, axis=1)))
    feat[f"{PREFIX}p1_bipartite_minimax_duel_value"] = p1_saddle
    feat[f"{PREFIX}p2_bipartite_minimax_duel_value"] = p2_saddle
    feat[f"{PREFIX}diff_bipartite_minimax_duel_value"] = p1_saddle - p2_saddle

    # -------------------------------------------------------------------------
    # 7. Realizable HP (Anti-HP Illusion)
    # -------------------------------------------------------------------------
    def calc_realizable_hp(my_team, opp_team, D_opp):
        real_hp = 0.0
        for i, mon in enumerate(my_team):
            if mon["hp_pct"] <= 0.001:
                continue
            max_threat = 0.0
            for j, opp in enumerate(opp_team):
                if opp["spe_eff"] > mon["spe_eff"]:
                    d_in = D_opp[j, i]
                    if d_in > max_threat:
                        max_threat = d_in
            real_hp += mon["hp_pct"] * math.exp(-max_threat)
        return real_hp

    r_hp_p1 = calc_realizable_hp(p1_alive, p2_alive, D2)
    r_hp_p2 = calc_realizable_hp(p2_alive, p1_alive, D1)
    feat[f"{PREFIX}p1_realizable_hp"] = float(r_hp_p1)
    feat[f"{PREFIX}p2_realizable_hp"] = float(r_hp_p2)
    feat[f"{PREFIX}diff_realizable_hp"] = float(r_hp_p1 - r_hp_p2)
    feat[f"{PREFIX}ratio_realizable_hp"] = float(r_hp_p1 / (r_hp_p1 + r_hp_p2 + 0.001))

    # -------------------------------------------------------------------------
    # 8. Micro-Rollout v2 (Non-Greedy with Rest/Curse & Turn 31 Switch)
    # -------------------------------------------------------------------------
    sim_p1_hp = [m["hp_pct"] for m in p1_alive]
    sim_p2_hp = [m["hp_pct"] for m in p2_alive]
    curr_1 = next((i for i, m in enumerate(p1_alive) if m["name"] == p1_active_t30), 0)
    curr_2 = next((j for j, m in enumerate(p2_alive) if m["name"] == p2_active_t30), 0)

    # Pillar 1.5: Proactive Switch at Turn 31
    if curr_1 < len(p1_alive) and curr_2 < len(p2_alive):
        if D1[curr_1, curr_2] < 0.18:
            for b_idx, b_mon in enumerate(p1_alive):
                if b_idx != curr_1 and b_mon["hp_pct"] > 0.001:
                    incoming_b = D2[curr_2, b_idx] + (0.125 if (spikes_p1 and b_mon["is_grounded"]) else 0.0)
                    if incoming_b < 0.35 and D1[b_idx, curr_2] > 0.45:
                        curr_1 = b_idx
                        break

    for _ in range(15):
        if curr_1 >= len(p1_alive) or curr_2 >= len(p2_alive):
            break
        if sim_p1_hp[curr_1] <= 0.001:
            curr_1 += 1
            if curr_1 >= len(p1_alive):
                break
        if sim_p2_hp[curr_2] <= 0.001:
            curr_2 += 1
            if curr_2 >= len(p2_alive):
                break

        m1 = p1_alive[curr_1]
        m2 = p2_alive[curr_2]
        d1 = D1[curr_1, curr_2]
        d2 = D2[curr_2, curr_1]

        # Pillar 1.2: Active Recovery & Rest/Curse Loop
        # Check P1 heal
        if m1["has_instant_heal"] and sim_p1_hp[curr_1] <= 0.55 and d2 < 0.46:
            sim_p1_hp[curr_1] = min(1.0, sim_p1_hp[curr_1] + 0.50)
        elif m1["has_rest"] and sim_p1_hp[curr_1] <= 0.35 and sim_p1_hp[curr_1] > d2:
            sim_p1_hp[curr_1] = 1.0  # Rest to 100%
            if m1["has_curse"]:
                d1 *= 1.3  # Curse boost

        # Check P2 heal
        if m2["has_instant_heal"] and sim_p2_hp[curr_2] <= 0.55 and d1 < 0.46:
            sim_p2_hp[curr_2] = min(1.0, sim_p2_hp[curr_2] + 0.50)
        elif m2["has_rest"] and sim_p2_hp[curr_2] <= 0.35 and sim_p2_hp[curr_2] > d1:
            sim_p2_hp[curr_2] = 1.0
            if m2["has_curse"]:
                d2 *= 1.3

        if m1["spe_eff"] >= m2["spe_eff"]:
            sim_p2_hp[curr_2] -= d1
            if sim_p2_hp[curr_2] > 0.001:
                sim_p1_hp[curr_1] -= d2
        else:
            sim_p1_hp[curr_1] -= d2
            if sim_p1_hp[curr_1] > 0.001:
                sim_p2_hp[curr_2] -= d1

    rollout_margin = float(sum(max(h, 0.0) for h in sim_p1_hp) - sum(max(h, 0.0) for h in sim_p2_hp))
    feat[f"{PREFIX}greedy_endgame_rollout_margin"] = rollout_margin

    # -------------------------------------------------------------------------
    # 9. Pillar 2.1: Regime-Switching Non-Linear Signals (HP vs Physics)
    # -------------------------------------------------------------------------
    p1_total_hp = sum(p1_hp.values())
    p2_total_hp = sum(p2_hp.values()) + (p2_unrevealed * 1.0)
    delta_hp_tot = p1_total_hp - p2_total_hp

    # Comeback override: P1 is down in HP, but Rollout physics predicts P1 wins!
    feat[f"{PREFIX}comeback_override_signal"] = float(max(0.0, -delta_hp_tot) * max(0.0, rollout_margin) * (1.0 + p1_chain))

    # Throw collapse: P1 is up in HP, but Rollout physics predicts P2 collapses P1!
    feat[f"{PREFIX}throw_collapse_signal"] = float(max(0.0, delta_hp_tot) * max(0.0, -rollout_margin))

    # Anti-meta viability inversion: low viability team has positive minimax duel advantage
    p1_bench_v = sum(_get_stats(m["name"]).get("base_hp", 80) for m in p1_alive)
    p2_bench_v = sum(_get_stats(m["name"]).get("base_hp", 80) for m in p2_alive)
    delta_viab = (p1_bench_v - p2_bench_v) / 100.0
    feat[f"{PREFIX}anti_meta_viability_inversion"] = float(-delta_viab * (p1_saddle - p2_saddle))

    # Post-T30 bench vs active delta
    p1_bench_only = [m for m in p1_alive if m["name"] != p1_active_t30]
    p2_bench_only = [m for m in p2_alive if m["name"] != p2_active_t30]
    bench_minimax = 0.0
    if p1_bench_only and p2_bench_only:
        b_d1 = np.zeros((len(p1_bench_only), len(p2_bench_only)))
        for bi, bm1 in enumerate(p1_bench_only):
            for bj, bm2 in enumerate(p2_bench_only):
                b_d1[bi, bj] = _calc_damage_fraction(bm1, bm2)
        bench_minimax = float(np.mean(np.max(b_d1, axis=0)))
    active_duel = D1[0, 0] if D1.shape[0] > 0 and D1.shape[1] > 0 else 0.0
    feat[f"{PREFIX}post_t30_bench_vs_active_delta"] = float(bench_minimax - active_duel)

    # -------------------------------------------------------------------------
    # 10. Prior Vulnerability & Speed Ceiling
    # -------------------------------------------------------------------------
    p2_unrevealed_prior_vuln = 0.0
    if p2_unrevealed > 0:
        seen_p2_species = {m["name"] for m in p2_alive}
        unseen_ou = [p for p in _top_ou_priors() if p["name"] not in seen_p2_species]
        if unseen_ou:
            p1_top_spe = max([m["spe_eff"] for m in p1_alive] or [80.0])
            total_freq = sum(p["freq"] for p in unseen_ou)
            exp_d = 0.0
            speed_sup = 0.0
            for p in unseen_ou:
                w = p["freq"] / max(total_freq, 0.001)
                dummy = build_combat_record(p["name"], 1.0, "nostatus", {}, [])
                d_val = max([_calc_damage_fraction(m, dummy) for m in p1_alive] or [0.1])
                exp_d += w * d_val
                if p1_top_spe > (2.0 * p["spe"] + 98.0):
                    speed_sup += w
            p2_unrevealed_prior_vuln = exp_d * speed_sup * float(p2_unrevealed)
    feat[f"{PREFIX}p2_unrevealed_prior_vulnerability"] = float(p2_unrevealed_prior_vuln)

    p1_max_spe = max([m["spe_eff"] for m in p1_alive] or [80.0])
    top_ou_speeds = [2.0 * p["spe"] + 98.0 for p in _top_ou_priors()]
    slower_count = sum(1 for s in top_ou_speeds if s < p1_max_spe)
    q_ou = slower_count / max(len(top_ou_speeds), 1)
    feat[f"{PREFIX}p1_speed_ceiling_vs_unrevealed_ou_quantile"] = float(q_ou ** max(float(p2_unrevealed), 1.0))

    return feat
