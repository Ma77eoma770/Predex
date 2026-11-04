"""Feature di Max (copiate da mypredex-main/src/features.py, senza modifiche alle funzioni).

Usate nel Passo 19: i gruppi field+threat entrano in LightGBM, tutte le 122 (material+status+
moves+field+threat) nella sua regressione logistica. Dipendono solo dalla singola battaglia.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(name):
    with open(ROOT / "raw_data" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def simple_features(data):
    rows = []
    for b in data:
        r = {"battle_id": b["battle_id"]}
        team = b.get("p1_team_details", [])
        if team:
            for s in ("hp", "spe", "atk", "def"):
                r[f"p1_mean_{s}"] = np.mean([p.get(f"base_{s}", 0) for p in team])
        lead = b.get("p2_lead_details")
        if lead:
            r["p2_lead_hp"] = lead.get("base_hp", 0)
            r["p2_lead_spe"] = lead.get("base_spe", 0)
        rows.append(r)
    return pd.DataFrame(rows).fillna(0)


# ---------------------------------------------------------------------------
# Gruppo 1: vantaggio materiale (KO, HP, Pokémon rimasti)
# (p1_left/p2_left/left_diff rimosse: erano 6 - ko, cioè duplicati esatti di p*_ko e ko_diff)
# NB: i KO di p2 sono sottostimati nei dati (la riga 'fnt' compare quasi solo per p1).
# ---------------------------------------------------------------------------
def material_features(data):
    rows = []
    for b in data:
        tl = b["battle_timeline"]
        r = {"battle_id": b["battle_id"]}
        ko = {"p1": set(), "p2": set()}
        last_hp = {"p1": {}, "p2": {}}  # ultimo hp noto per ogni Pokémon visto
        for t in tl:
            for p in ("p1", "p2"):
                s = t[f"{p}_pokemon_state"]
                last_hp[p][s["name"]] = s["hp_pct"]
                if s["status"] == "fnt":
                    ko[p].add(s["name"])
        for p in ("p1", "p2"):
            r[f"{p}_ko"] = len(ko[p])
            r[f"{p}_seen"] = len(last_hp[p])
            r[f"{p}_hp_sum"] = sum(last_hp[p].values())
        for turn in (10, 20, 30):
            s1 = tl[turn - 1]["p1_pokemon_state"]["hp_pct"]
            s2 = tl[turn - 1]["p2_pokemon_state"]["hp_pct"]
            r[f"hp_diff_t{turn}"] = s1 - s2
        r["hp_mean_diff"] = np.mean(
            [t["p1_pokemon_state"]["hp_pct"] - t["p2_pokemon_state"]["hp_pct"] for t in tl]
        )
        r["ko_diff"] = r["p1_ko"] - r["p2_ko"]
        r["hp_sum_diff"] = r["p1_hp_sum"] - r["p2_hp_sum"]
        rows.append(r)
    return pd.DataFrame(rows)


# NOTA (esperimento scartato): stima dei KO di p2. Nei dati la riga 'fnt' compare quasi solo
# per p1 (KO medi 1.69 vs 0.50). Ho provato a contare come KO anche i Pokémon usciti dal campo
# con hp <= soglia (0.15/0.3/0.5): CV logistica 0.6559 -> 0.6559-0.6573, entro il rumore tra
# fold (~0.003). Non aggiunta al set di feature.

# ---------------------------------------------------------------------------
# Gruppo 2: status, effetti e boost (turni in cui il Pokémon attivo ha una condizione)
# ---------------------------------------------------------------------------
BAD_STATUS = ("slp", "par", "tox", "psn", "brn", "frz")
BOOSTS = ("atk", "def", "spa", "spd", "spe")


def status_features(data):
    rows = []
    for b in data:
        tl = b["battle_timeline"]
        r = {"battle_id": b["battle_id"]}
        for p in ("p1", "p2"):
            states = [t[f"{p}_pokemon_state"] for t in tl]
            for st in BAD_STATUS:
                r[f"{p}_turns_{st}"] = sum(s["status"] == st for s in states)
            r[f"{p}_turns_badstatus"] = sum(s["status"] in BAD_STATUS for s in states)
            r[f"{p}_status_t30"] = int(states[-1]["status"] in BAD_STATUS)
            r[f"{p}_n_statused"] = len({s["name"] for s in states if s["status"] in BAD_STATUS})
            for ef in ("confusion", "leechseed", "trapped", "substitute"):
                r[f"{p}_turns_{ef}"] = sum(ef in s["effects"] for s in states)
            for bst in BOOSTS:
                r[f"{p}_boost_{bst}"] = np.mean([s["boosts"][bst] for s in states])
        r["badstatus_diff"] = r["p2_turns_badstatus"] - r["p1_turns_badstatus"]  # >0 = p2 più bloccato
        for bst in BOOSTS:
            r[f"boost_{bst}_diff"] = r[f"p1_boost_{bst}"] - r[f"p2_boost_{bst}"]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gruppo 3: mosse e cambi
# ---------------------------------------------------------------------------
KEY_MOVES = ("rest", "protect", "toxic", "spikes", "explosion", "selfdestruct", "substitute", "curse")


def moves_features(data):
    rows = []
    for b in data:
        tl = b["battle_timeline"]
        r = {"battle_id": b["battle_id"]}
        for p in ("p1", "p2"):
            mv = [t[f"{p}_move_details"] for t in tl]
            used = [m for m in mv if m is not None]
            n = max(len(used), 1)
            att = [m for m in used if m["base_power"] > 0]
            r[f"{p}_n_nomove"] = len(mv) - len(used)
            names = [t[f"{p}_pokemon_state"]["name"] for t in tl]
            r[f"{p}_n_switch"] = sum(a != c for a, c in zip(names, names[1:]))
            for cat in ("physical", "special", "status"):
                r[f"{p}_share_{cat}"] = sum(m["category"] == cat for m in used) / n
            r[f"{p}_mean_power"] = np.mean([m["base_power"] for m in att]) if att else 0.0
            r[f"{p}_mean_acc"] = np.mean([m["accuracy"] for m in used]) if used else 0.0
            r[f"{p}_n_priority"] = sum(m["priority"] != 0 for m in used)
            for k in KEY_MOVES:
                r[f"{p}_mv_{k}"] = sum(m["name"] == k for m in used)
        r["switch_diff"] = r["p1_n_switch"] - r["p2_n_switch"]
        r["nomove_diff"] = r["p1_n_nomove"] - r["p2_n_nomove"]
        r["power_diff"] = r["p1_mean_power"] - r["p2_mean_power"]
        rows.append(r)
    return pd.DataFrame(rows)


# NOTA (esperimenti scartati), CV logistica rispetto a material+status+moves (0.6677):
# - "damage": danno inflitto/subito tra turni consecutivi, per finestre di 10 turni e trend -> 0.6673
# - "typematch": efficacia dei tipi delle mosse usate e vantaggio di tipo tra gli attivi -> 0.6713
#   (+0.004, entro il rumore di ~0.005; peggiora il gradient boosting)
# - entrambi -> 0.6704. Nessuno aggiunto al set di feature.
# - "team": one-hot di 132 specie (team di p1, Pokémon di p2 visti) e 12 oggetti di p1, più
#   statistiche base medie dei due team (386 colonne in totale) -> 0.6649 con C=1, 0.6669 con C=0.01,
#   0.6646 con gradient boosting regolarizzato. Nessun guadagno.
# Rispetto a material+status+moves+field+threat (logistica C=0.01, 0.6766):
# - "movebag": conteggio per lato e differenza di ognuna delle ~100 mosse con almeno 200 usi
#   (~300 colonne) -> 0.6723 con C=0.01, 0.6754 con C=1, 0.6699 con C=0.003, 0.6694 con boosting.
#   Nessun guadagno e più varianza tra fold: i segnali sparsi delle singole mosse (Blizzard,
#   Thief, Pursuit nell'audit dei campi) non si sommano.

# ---------------------------------------------------------------------------
# Gruppo 4: condizioni del campo (spikes, rapid spin, schermi).
# In Gen 2 Spikes resta finché il lato colpito non usa Rapid Spin: ogni cambio successivo
# costa HP, quindi è un vantaggio che pesa soprattutto dopo il turno 30.
# Ordine dentro un turno ignoto: applico prima spikes, poi rapid spin.
# ---------------------------------------------------------------------------
def _move_name(t, p):
    m = t[f"{p}_move_details"]
    return m["name"] if m else None


def field_features(data):
    rows = []
    for b in data:
        tl = b["battle_timeline"]
        r = {"battle_id": b["battle_id"]}
        for p, q in (("p1", "p2"), ("p2", "p1")):
            active, turns_on, switch_in = False, 0, 0
            prev_name = None
            for t in tl:
                name = t[f"{p}_pokemon_state"]["name"]
                if active and prev_name is not None and name != prev_name:
                    switch_in += 1  # cambio con spikes avversarie sul proprio lato
                if _move_name(t, q) == "spikes":
                    active = True
                if _move_name(t, p) == "rapidspin":
                    active = False
                turns_on += active
                prev_name = name
            names = [_move_name(t, p) for t in tl]
            r[f"{p}_spikes_on_side"] = int(active)  # spikes avversarie attive al turno 30
            r[f"{p}_turns_spikes_on_side"] = turns_on
            r[f"{p}_switch_into_spikes"] = switch_in
            r[f"{p}_rapidspin"] = names.count("rapidspin")
            r[f"{p}_screens"] = names.count("reflect") + names.count("lightscreen")
            r[f"{p}_screens_last5"] = sum(n in ("reflect", "lightscreen") for n in names[-5:])
        for k in ("spikes_on_side", "turns_spikes_on_side", "switch_into_spikes", "screens_last5"):
            r[f"{k}_diff"] = r[f"p1_{k}"] - r[f"p2_{k}"]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gruppo 5: minaccia attuale del Pokémon attivo (boost negli ultimi turni).
# La media dei boost su 30 turni (gruppo status) non dice chi è potenziato ADESSO.
# ---------------------------------------------------------------------------
def threat_features(data):
    rows = []
    for b in data:
        tl = b["battle_timeline"]
        r = {"battle_id": b["battle_id"]}
        for p in ("p1", "p2"):
            s = tl[-1][f"{p}_pokemon_state"]
            bo = s["boosts"]
            off = bo["atk"] + bo["spa"] + bo["spe"]
            r[f"{p}_boost_t30"] = sum(bo.values())
            r[f"{p}_off_boost_t30"] = off
            r[f"{p}_def_boost_t30"] = bo["def"] + bo["spd"]
            r[f"{p}_max_boost_t30"] = max(bo.values())
            r[f"{p}_threat_t30"] = off * s["hp_pct"]  # un Pokémon potenziato conta se ha ancora HP
            r[f"{p}_boost_last3"] = np.mean([sum(t[f"{p}_pokemon_state"]["boosts"].values())
                                             for t in tl[-3:]])
        for k in ("boost_t30", "off_boost_t30", "def_boost_t30", "threat_t30", "boost_last3"):
            r[f"{k}_diff"] = r[f"p1_{k}"] - r[f"p2_{k}"]
        rows.append(r)
    return pd.DataFrame(rows)

def build(data, groups=("material", "status", "moves", "field", "threat")):
    """Unisce i gruppi di feature richiesti (per battle_id) in un unico DataFrame."""
    fns = {"simple": simple_features, "material": material_features,
           "status": status_features, "moves": moves_features,
           "field": field_features, "threat": threat_features}
    df = None
    for g in groups:
        part = fns[g](data)
        df = part if df is None else df.merge(part, on="battle_id")
    return df
