"""Gen 2 type chart and species lookup shared by the type-based feature modules.

Both tables are static game data: the type chart is the one supplied with the competition,
and the species table is derived from the competition inputs by
`python -m src.features.build_species_types` (names, types and base stats only, never the target).
"""

import csv
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

from src.config import DATA_DIR, TYPE_CHART_CSV

SPECIES_CSV = DATA_DIR / "species_types.csv"
BASE_STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]

# "notype" fills the second slot of single-typed Pokémon; "threequestionmarks" is the
# "???" type of Curse. Neither changes damage, so both are neutral (multiplier 1).
NEUTRAL_TYPES = {"notype", "threequestionmarks"}


@lru_cache(maxsize=None)
def load_type_chart() -> Dict[Tuple[str, str], float]:
    """(attacking_type, defending_type) -> multiplier. The CSV lists only values != 1."""
    with open(TYPE_CHART_CSV, newline="", encoding="utf-8") as f:
        return {(r["attacking_type"], r["defending_type"]): float(r["multiplier"]) for r in csv.DictReader(f)}


@lru_cache(maxsize=None)
def all_types() -> Tuple[str, ...]:
    """The 17 Gen 2 types appearing in the chart."""
    chart = load_type_chart()
    return tuple(sorted({t for pair in chart for t in pair}))


def clean_types(types: Iterable[str]) -> Tuple[str, ...]:
    return tuple(t for t in types if t not in NEUTRAL_TYPES)


def multiplier(attacking_type: str, defending_types: Iterable[str]) -> float:
    """Damage multiplier of one attacking type against a (possibly dual-typed) defender."""
    if attacking_type in NEUTRAL_TYPES:
        return 1.0
    chart = load_type_chart()
    result = 1.0
    for def_type in clean_types(defending_types):
        result *= chart.get((attacking_type, def_type), 1.0)
    return result


def best_stab_multiplier(attacker_types: Iterable[str], defender_types: Iterable[str]) -> float:
    """Best multiplier the attacker gets from its own types (STAB moves) against the defender."""
    att = clean_types(attacker_types)
    if not att:
        return 1.0
    return max(multiplier(t, defender_types) for t in att)


@lru_cache(maxsize=None)
def species_table() -> Dict[str, Dict[str, object]]:
    """species -> {"types": tuple, base stats...}, read from data/species_types.csv."""
    table: Dict[str, Dict[str, object]] = {}
    with open(SPECIES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            entry: Dict[str, object] = {"types": clean_types((r["type1"], r["type2"]))}
            entry.update({s: float(r[s]) for s in BASE_STATS})
            table[r["species"]] = entry
    return table


def species_types(name: str) -> Optional[Tuple[str, ...]]:
    """Types of a species, or None if it never appears with its details in the competition data."""
    entry = species_table().get(name)
    return entry["types"] if entry else None  # type: ignore[return-value]


def species_stat(name: str, stat: str) -> Optional[float]:
    entry = species_table().get(name)
    return entry[stat] if entry else None  # type: ignore[return-value]
