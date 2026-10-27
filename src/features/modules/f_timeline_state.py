"""Timeline state & end-game dynamics feature extractor (Gen 2 OU).

Extracts accurate unique fainted counts, total estimated team HP,
P2 unrevealed count, status conditions, and active boosts at turn 30.
Prefix: tl_
"""

from typing import Dict, Any, Set
import numpy as np

PREFIX = "tl_"


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract fine-grained timeline and end-state features for a single battle."""
    feat: Dict[str, float] = {}

    p1_team_names = [p.get("name", f"p1_{i}") for i, p in enumerate(data.get("p1_team_details", []))]
    p2_lead_name = data.get("p2_lead_details", {}).get("name", "p2_lead")

    # Track last observed HP percentage per Pokémon
    # P1 starts with all 6 at 1.0
    p1_last_hp: Dict[str, float] = {name: 1.0 for name in p1_team_names}
    p2_last_hp: Dict[str, float] = {p2_lead_name: 1.0}

    # Track status conditions at last turn
    p1_last_status: Dict[str, str] = {name: "nostatus" for name in p1_team_names}
    p2_last_status: Dict[str, str] = {p2_lead_name: "nostatus"}

    p1_fainted: Set[str] = set()
    p2_fainted: Set[str] = set()
    p2_seen_names: Set[str] = {p2_lead_name}

    p1_switches = 0
    p2_switches = 0
    prev_p1_active = None
    prev_p2_active = None

    timeline = data.get("battle_timeline", [])

    for turn_idx, turn in enumerate(timeline):
        p1_st = turn.get("p1_pokemon_state", {})
        p2_st = turn.get("p2_pokemon_state", {})

        p1_name = p1_st.get("name")
        p2_name = p2_st.get("name")

        if p1_name:
            if prev_p1_active and p1_name != prev_p1_active:
                p1_switches += 1
            prev_p1_active = p1_name
            p1_hp = float(p1_st.get("hp_pct", 1.0))
            p1_last_hp[p1_name] = p1_hp
            p1_last_status[p1_name] = p1_st.get("status", "nostatus")
            if p1_hp == 0.0 or p1_st.get("status") == "fnt":
                p1_fainted.add(p1_name)

        if p2_name:
            p2_seen_names.add(p2_name)
            if prev_p2_active and p2_name != prev_p2_active:
                p2_switches += 1
            prev_p2_active = p2_name
            p2_hp = float(p2_st.get("hp_pct", 1.0))
            p2_last_hp[p2_name] = p2_hp
            p2_last_status[p2_name] = p2_st.get("status", "nostatus")
            if p2_hp == 0.0 or p2_st.get("status") == "fnt":
                p2_fainted.add(p2_name)

    # 1. Alive / Fainted counts
    p1_fainted_count = len(p1_fainted)
    p2_fainted_count = len(p2_fainted)
    p1_alive_count = max(0, 6 - p1_fainted_count)
    p2_alive_count = max(0, 6 - p2_fainted_count)

    feat[f"{PREFIX}p1_alive_count"] = float(p1_alive_count)
    feat[f"{PREFIX}p2_alive_count"] = float(p2_alive_count)
    feat[f"{PREFIX}diff_alive_count"] = float(p1_alive_count - p2_alive_count)
    feat[f"{PREFIX}p1_fainted_count"] = float(p1_fainted_count)
    feat[f"{PREFIX}p2_fainted_count"] = float(p2_fainted_count)

    # 2. Information Asymmetry: P2 unrevealed count
    p2_unrevealed = max(0, 6 - len(p2_seen_names))
    feat[f"{PREFIX}p2_revealed_count"] = float(len(p2_seen_names))
    feat[f"{PREFIX}p2_unrevealed_count"] = float(p2_unrevealed)

    # 3. Total Team HP % (known + assumed 1.0 for unrevealed)
    p1_total_hp = sum(p1_last_hp.values())
    p2_total_hp = sum(p2_last_hp.values()) + (p2_unrevealed * 1.0)

    feat[f"{PREFIX}p1_total_hp"] = float(p1_total_hp)
    feat[f"{PREFIX}p2_total_hp"] = float(p2_total_hp)
    feat[f"{PREFIX}diff_total_hp"] = float(p1_total_hp - p2_total_hp)

    # 4. Momentum & Switching
    feat[f"{PREFIX}p1_switches"] = float(p1_switches)
    feat[f"{PREFIX}p2_switches"] = float(p2_switches)
    feat[f"{PREFIX}diff_switches"] = float(p1_switches - p2_switches)

    # 5. Active state at turn 30 (boosts & status)
    if timeline:
        last_t = timeline[-1]
        p1_act = last_t.get("p1_pokemon_state", {})
        p2_act = last_t.get("p2_pokemon_state", {})

        p1_boosts = p1_act.get("boosts", {})
        p2_boosts = p2_act.get("boosts", {})

        p1_tot_boosts = sum(p1_boosts.values()) if p1_boosts else 0
        p2_tot_boosts = sum(p2_boosts.values()) if p2_boosts else 0

        feat[f"{PREFIX}p1_active_atk_boost"] = float(p1_boosts.get("atk", 0))
        feat[f"{PREFIX}p1_active_def_boost"] = float(p1_boosts.get("def", 0))
        feat[f"{PREFIX}p1_active_spe_boost"] = float(p1_boosts.get("spe", 0))
        feat[f"{PREFIX}p2_active_atk_boost"] = float(p2_boosts.get("atk", 0))
        feat[f"{PREFIX}p2_active_def_boost"] = float(p2_boosts.get("def", 0))
        feat[f"{PREFIX}p2_active_spe_boost"] = float(p2_boosts.get("spe", 0))
        feat[f"{PREFIX}diff_active_total_boosts"] = float(p1_tot_boosts - p2_tot_boosts)

    # 6. Major Gen 2 Status conditions across seen Pokémon
    p1_slp_frz = sum(1 for st in p1_last_status.values() if st in ("slp", "frz"))
    p2_slp_frz = sum(1 for st in p2_last_status.values() if st in ("slp", "frz"))
    feat[f"{PREFIX}p1_slp_or_frz_count"] = float(p1_slp_frz)
    feat[f"{PREFIX}p2_slp_or_frz_count"] = float(p2_slp_frz)
    feat[f"{PREFIX}diff_slp_or_frz"] = float(p1_slp_frz - p2_slp_frz)

    p1_major_st = sum(1 for st in p1_last_status.values() if st in ("slp", "frz", "par", "tox", "psn", "brn"))
    p2_major_st = sum(1 for st in p2_last_status.values() if st in ("slp", "frz", "par", "tox", "psn", "brn"))
    feat[f"{PREFIX}p1_major_status_count"] = float(p1_major_st)
    feat[f"{PREFIX}p2_major_status_count"] = float(p2_major_st)
    feat[f"{PREFIX}diff_major_status"] = float(p1_major_st - p2_major_st)

    return feat
