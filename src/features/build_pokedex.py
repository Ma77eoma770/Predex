"""Costruisce data/pokedex.json SOLO da train.jsonl (persona D, pulizia del 05/11).

Il Pokédex usato dai moduli f_bipartite_physics, f_mitigations e f_random_forest contiene, per ogni
specie, i tipi e le 6 statistiche base. Nei dati della gara queste informazioni ci sono già: in
p1_team_details (le 6 specie di P1) e in p2_lead_details (il primo Pokémon di P2). Qui le
raccogliamo dalle battaglie di train, nell'ordine in cui compaiono. Le specie che non compaiono mai
con le statistiche nel train (8 specie viste solo nel test) restano fuori: i moduli usano per loro
i valori di riserva (statistiche 80, tipo normal). Il test non viene letto.

    python src/features/build_pokedex.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, TRAIN_JSONL

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]


def build_pokedex(jsonl_path=TRAIN_JSONL):
    dex = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            entries = list(b.get("p1_team_details", []))
            if b.get("p2_lead_details"):
                entries.append(b["p2_lead_details"])
            for p in entries:
                name = p["name"].strip().lower()
                if name not in dex:
                    dex[name] = {"types": list(p["types"]), **{s: float(p[s]) for s in STATS}}
    return dex


if __name__ == "__main__":
    dex = build_pokedex()
    with open(DATA_DIR / "pokedex.json", "w", encoding="utf-8") as f:
        json.dump(dex, f, indent=2)
    print(f"Pokédex scritto con {len(dex)} specie (solo train)")
