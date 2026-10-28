"""Build data/species_types.csv: species -> types and base stats, from the competition data only.

The JSON gives types and base stats for Player 1's six Pokémon and for Player 2's lead, but
only the NAME of the other Player 2 Pokémon revealed in the timeline. Since a species' types
and base stats are fixed game data, every appearance with details (train + test inputs) is
collected into one table. The target `player_won` is never read.

Usage:
    python -m src.features.build_species_types
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_JSONL, TRAIN_JSONL
from src.features.type_data import BASE_STATS, SPECIES_CSV


def _record(p: dict) -> Tuple:
    return (p["types"][0], p["types"][1], *(p[s] for s in BASE_STATS))


def build_species_table() -> None:
    table: Dict[str, Tuple] = {}
    revealed_p2 = set()

    for path in (TRAIN_JSONL, TEST_JSONL):
        with open(path, encoding="utf-8") as f:
            for line in f:
                battle = json.loads(line)
                with_details = battle["p1_team_details"] + [battle["p2_lead_details"]]
                for p in with_details:
                    rec = _record(p)
                    if p["name"] in table and table[p["name"]] != rec:
                        # Fail loudly: two different records for one species would make the lookup ambiguous.
                        raise ValueError(f"Conflicting details for {p['name']}: {table[p['name']]} vs {rec}")
                    table[p["name"]] = rec
                revealed_p2.update(t["p2_pokemon_state"]["name"] for t in battle["battle_timeline"])

    SPECIES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SPECIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "type1", "type2", *BASE_STATS])
        for name in sorted(table):
            writer.writerow([name, *table[name]])

    missing = sorted(revealed_p2 - set(table))
    print(f"Saved {len(table)} species -> {SPECIES_CSV}")
    print(f"Revealed P2 species without details anywhere ({len(missing)}): {missing}")


if __name__ == "__main__":
    build_species_table()
