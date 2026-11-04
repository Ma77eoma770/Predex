"""Feature di "bravura dei giocatori" ricavate dal record delle squadre (persona D).

Una squadra = insieme delle 6 specie. Il record di una squadra si costruisce dalle battaglie di
TRAIN con etichetta, in entrambi i ruoli:
  - come P1 vince se player_won = 1;
  - come P2 (solo se P2 ha rivelato tutti e 6 i Pokémon) vince se player_won = 0.
Per una battaglia:
  sk1, n1 = tasso di vittoria smussato e partite della squadra di P1;
  sk2, n2 = lo stesso per le squadre compatibili con i Pokémon rivelati da P2
            (se P2 ne ha rivelati 6 è la squadra esatta; con meno di 3 rivelati nessuna stima);
  skd     = logit(sk1) - logit(sk2).
Smussamento: (vittorie + m * 0,5) / (partite + m), con m = 5.

Queste feature USANO LE ETICHETTE: vanno sempre calcolate dentro la cross-validation.
  - battaglie da valutare/prevedere: record costruito su tutte le battaglie di training;
  - battaglie di training: record costruito con una divisione interna in 5 parti, così
    nessuna battaglia vede il proprio risultato.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

SKILL_COLUMNS = ["sk1", "n1", "sk2", "n2", "skd"]
SMOOTHING = 5.0


def read_teams(jsonl_path: Path) -> Dict[int, Tuple[frozenset, frozenset]]:
    """battle_id -> (specie della squadra P1, specie di P2 viste nei 30 turni)."""
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            out[b["battle_id"]] = (frozenset(p["name"] for p in b["p1_team_details"]),
                                   frozenset(t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]))
    return out


class PlayerSkill:
    """Record delle squadre appreso da (squadre, etichette) di training; si applica a qualsiasi battaglia."""

    def __init__(self, m: float = SMOOTHING):
        self.m = m

    def fit(self, teams: List[Tuple[frozenset, frozenset]], y: np.ndarray):
        self.wins, self.games = defaultdict(float), defaultdict(float)
        for (p1, p2), yi in zip(teams, y):
            self.wins[p1] += yi; self.games[p1] += 1
            if len(p2) == 6:
                self.wins[p2] += 1 - yi; self.games[p2] += 1
        self.by_species = defaultdict(set)
        for t in self.games:
            for s in t:
                self.by_species[s].add(t)
        return self

    def _rate(self, w, g):
        return (w + self.m * 0.5) / (g + self.m)

    def transform(self, teams: List[Tuple[frozenset, frozenset]]) -> np.ndarray:
        rows = []
        for p1, p2 in teams:
            w1, g1 = self.wins.get(p1, 0.0), self.games.get(p1, 0.0)
            w2 = g2 = 0.0
            if len(p2) >= 3:
                cand = None
                for s in p2:
                    cand = set(self.by_species.get(s, ())) if cand is None else cand & self.by_species.get(s, set())
                    if not cand:
                        break
                for t in cand or ():
                    w2 += self.wins[t]; g2 += self.games[t]
            sk1, sk2 = self._rate(w1, g1), self._rate(w2, g2)
            rows.append([sk1, g1, sk2, g2, np.log(sk1 / (1 - sk1)) - np.log(sk2 / (1 - sk2))])
        return np.array(rows)


def skill_features(fit_teams, fit_y, apply_teams, m=SMOOTHING, inner=5, seed=0):
    """(feature per le battaglie di training senza leakage, feature per le battaglie da prevedere)."""
    fit_y = np.asarray(fit_y)
    F_fit = np.zeros((len(fit_teams), len(SKILL_COLUMNS)))
    for a, b in KFold(inner, shuffle=True, random_state=seed).split(fit_teams):
        model = PlayerSkill(m).fit([fit_teams[i] for i in a], fit_y[a])
        F_fit[b] = model.transform([fit_teams[i] for i in b])
    F_apply = PlayerSkill(m).fit(fit_teams, fit_y).transform(apply_teams)
    return F_fit, F_apply
