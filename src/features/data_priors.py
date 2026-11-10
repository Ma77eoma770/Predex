"""Statistiche "a priori" calcolate SOLO da train.jsonl (persona D, pulizia del 04/11).

Il regolamento vieta "usage statistics, and manually collected tables". Alcuni moduli usavano
tabelle scritte a mano (i 20 Pokémon più usati in OU con le loro frequenze, la probabilità che
ci sia Snorlax, le liste di chi usa Explosion, Rapid Spin, Roar...). Qui le stesse informazioni
vengono ricavate dai dati della gara:
  - frequenza di una specie = quota di squadre di P1 del train che la contengono;
  - "utilizzatori" di un gruppo di mosse = specie che, nel train, usano almeno una di quelle mosse
    in almeno il 10% delle battaglie in cui compaiono (con almeno 30 apparizioni);
  - probabilità che una specie usi una mossa (per le "mosse probabili" di f_bipartite_physics).
Non si usano etichette (player_won) né il test, quindi niente leakage. Tutto è calcolato una volta
e tenuto in memoria.
"""

import json
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List

from src.config import TRAIN_JSONL

MIN_APPEARANCES = 30
MIN_USE_SHARE = 0.10


@lru_cache(maxsize=1)
def _scan():
    """Una passata sul train: squadre di P1, apparizioni delle specie, mosse usate per specie."""
    team_count = Counter()           # in quante squadre di P1 compare la specie
    n_teams = 0
    appear = Counter()               # battaglie (per lato) in cui la specie scende in campo
    used = defaultdict(Counter)      # battaglie in cui la specie usa la mossa
    move_info = {}                   # nome mossa -> (base_power, type, category) come nei dati
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            n_teams += 1
            for p in b.get("p1_team_details", []):
                team_count[p["name"].strip().lower()] += 1
            for side in ("p1", "p2"):
                moves_by_species = defaultdict(set)
                for t in b.get("battle_timeline", []):
                    name = t[f"{side}_pokemon_state"]["name"].strip().lower()
                    moves_by_species[name]
                    m = t.get(f"{side}_move_details")
                    if m:
                        moves_by_species[name].add(m["name"])
                        move_info.setdefault(m["name"], (float(m.get("base_power", 0) or 0), m.get("type", ""), m.get("category", "")))
                for name, ms in moves_by_species.items():
                    appear[name] += 1
                    for mv in ms:
                        used[name][mv] += 1
    return team_count, n_teams, appear, used, move_info


def species_frequency(name: str) -> float:
    team_count, n_teams, *_ = _scan()
    return team_count.get(name, 0) / max(n_teams, 1)


def top_species(k: int = 20) -> List[str]:
    team_count, *_ = _scan()
    return [s for s, _ in team_count.most_common(k)]


def team_share_with_any(species: Iterable[str]) -> float:
    """Quota di squadre di P1 del train che contengono almeno una delle specie indicate."""
    return _team_share(frozenset(species))


@lru_cache(maxsize=None)
def _team_share(species: FrozenSet[str]) -> float:
    n = hit = 0
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            n += 1
            hit += any(p["name"].strip().lower() in species for p in b.get("p1_team_details", []))
    return hit / max(n, 1)


@lru_cache(maxsize=None)
def move_users(moves: FrozenSet[str]) -> FrozenSet[str]:
    """Specie che nel train usano spesso almeno una delle mosse indicate."""
    _, _, appear, used, _ = _scan()
    out = set()
    for s, n in appear.items():
        if n < MIN_APPEARANCES:
            continue
        share = max((used[s][mv] for mv in moves), default=0) / n
        if share >= MIN_USE_SHARE:
            out.add(s)
    return frozenset(out)


@lru_cache(maxsize=1)
def move_priors() -> Dict[str, List[Dict]]:
    """Per ogni specie: le mosse che usa, con probabilità = quota di apparizioni in cui la usa."""
    _, _, appear, used, move_info = _scan()
    out = {}
    for s, n in appear.items():
        if n < MIN_APPEARANCES:
            continue
        rows = []
        for mv, c in used[s].most_common():
            power, mtype, cat = move_info[mv]
            rows.append({"name": mv, "prob": c / n, "power": power, "type": mtype, "category": cat})
        out[s] = rows
    return out


@lru_cache(maxsize=1)
def bulky_species(min_bulk: float = 270.0) -> FrozenSet[str]:
    """Specie "muro": base_hp + base_def + base_spd >= min_bulk, prese dalle squadre di P1 del train
    (dove le statistiche base sono nei dati) e con almeno MIN_APPEARANCES squadre."""
    team_count, *_ = _scan()
    stats = {}
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            for p in json.loads(line).get("p1_team_details", []):
                stats.setdefault(p["name"].strip().lower(), p["base_hp"] + p["base_def"] + p["base_spd"])
    return frozenset(s for s, b in stats.items() if b >= min_bulk and team_count[s] >= MIN_APPEARANCES)


@lru_cache(maxsize=1)
def average_team_member() -> Dict[str, float]:
    """Il "Pokémon medio" delle squadre di P1 del train: serve come valore a priori per i Pokémon di P2
    non ancora rivelati (prima erano costanti scritte a mano: 85 per statistica, 80% non volanti, ...)."""
    rows = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            for p in json.loads(line).get("p1_team_details", []):
                rows.append(p)
    n = max(len(rows), 1)
    mean = lambda fn: sum(fn(p) for p in rows) / n
    bst = lambda p: p["base_hp"] + p["base_atk"] + p["base_def"] + p["base_spa"] + p["base_spd"] + p["base_spe"]
    return {
        **{s: mean(lambda p, s=s: float(p[f"base_{s}"])) for s in ("hp", "atk", "def", "spa", "spd", "spe")},
        "grounded_share": mean(lambda p: float("flying" not in p["types"])),
        "phys_bulk": mean(lambda p: p["base_hp"] * p["base_def"] / 100.0),
        "spec_bulk": mean(lambda p: p["base_hp"] * p["base_spd"] / 100.0),
        "effective_bulk": mean(lambda p: p["base_hp"] * (p["base_def"] + p["base_spd"]) / 200.0),
        "firepower": mean(lambda p: float(max(p["base_atk"], p["base_spa"]))),
        "viability": mean(lambda p: min(80.0, max(20.0, (bst(p) - 300.0) / 3.0))),
    }
