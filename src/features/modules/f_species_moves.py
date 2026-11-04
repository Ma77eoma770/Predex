"""Feature su specie presenti e mosse usate (persona D).

Prefisso: sm_

Due gruppi di feature, per entrambi i giocatori:
  1. SPECIE: per ognuna delle specie più comuni, il suo stato al turno 30:
       -1   = non presente (P1) / mai vista in campo (P2)
        0   = presente ma KO
       0..1 = presente, con quella frazione di HP residui (1 = piena salute)
     Un solo numero dice "c'è?" e "come sta?". Gli alberi separano facilmente -1, 0 e il resto.
  2. MOSSE: per ognuna delle mosse più usate, quante volte il giocatore l'ha usata nei 30 turni.
     Descrive lo stile di gioco (es. molte Rest = squadra difensiva, Explosion = sacrificio).

Il vocabolario (quali specie e mosse sono "comuni") si costruisce contando le apparizioni
nel SOLO file di train. Il conteggio non usa l'esito delle battaglie (player_won),
quindi non porta informazione sulla risposta: non è data leakage.
"""

import json
from collections import Counter
from typing import Any, Dict, List

from src.config import TRAIN_JSONL

PREFIX = "sm_"

# Soglie minime di frequenza nel train. Specie o mosse più rare darebbero colonne quasi
# sempre vuote, su cui il modello potrebbe solo imparare a memoria pochi casi.
MIN_SPECIES_COUNT = 100   # apparizioni nelle squadre di P1 (su 60.000 slot)
MIN_MOVE_COUNT = 300      # usi totali di P1 + P2 (su circa 480.000 mosse)


def _build_vocab():
    species, moves = Counter(), Counter()
    if not TRAIN_JSONL.exists():
        return [], []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            for p in b.get("p1_team_details", []):
                species[p["name"]] += 1
            for t in b.get("battle_timeline", []):
                for side in ("p1", "p2"):
                    m = t.get(f"{side}_move_details")
                    if m:
                        moves[m["name"]] += 1
    top_species = sorted(n for n, c in species.items() if c >= MIN_SPECIES_COUNT)
    top_moves = sorted(n for n, c in moves.items() if c >= MIN_MOVE_COUNT)
    return top_species, top_moves


TOP_SPECIES, TOP_MOVES = _build_vocab()


def _species_state(timeline, side: str, roster: List[str]) -> Dict[str, float]:
    """Stato finale (-1 / 0 / HP) delle specie in `roster` per un giocatore."""
    state = {name: 1.0 for name in roster}    # P1: tutta la squadra parte al 100%
    for t in timeline:
        s = t.get(f"{side}_pokemon_state", {})
        name = s.get("name")
        if name:
            hp = float(s.get("hp_pct", 1.0))
            state[name] = 0.0 if s.get("status") == "fnt" else hp
    return state


def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    timeline = data.get("battle_timeline", [])

    p1_roster = [p.get("name") for p in data.get("p1_team_details", [])]
    p1_state = _species_state(timeline, "p1", p1_roster)
    p2_state = _species_state(timeline, "p2", [])   # di P2 conosciamo solo chi è sceso in campo

    for sp in TOP_SPECIES:
        feat[f"{PREFIX}p1_{sp}"] = p1_state.get(sp, -1.0)
        feat[f"{PREFIX}p2_{sp}"] = p2_state.get(sp, -1.0)

    counts = {"p1": Counter(), "p2": Counter()}
    for t in timeline:
        for side in ("p1", "p2"):
            m = t.get(f"{side}_move_details")
            if m:
                counts[side][m["name"]] += 1
    for mv in TOP_MOVES:
        feat[f"{PREFIX}mv_p1_{mv}"] = float(counts["p1"][mv])
        feat[f"{PREFIX}mv_p2_{mv}"] = float(counts["p2"][mv])

    return feat
