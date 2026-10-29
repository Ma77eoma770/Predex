"""Diagnose where the ~33% error of the team baseline comes from (analysis, not an experiment).

Two hypotheses: (a) information that is in the data is not captured by v1;
(b) the outcome is not predictable from the first 30 turns.

Part 1  Learning curve: does more training data still help?
Part 2  Error analysis on out-of-fold probabilities: where are the errors, is the model calibrated,
        are there confident errors beyond what calibration predicts?
Part 3  KO asymmetry: P2 KOs seem to be recorded as HP 0 much less often than P1 KOs.
        Counts on train AND test (inputs only), plus "vanished" Pokémon at low HP.
Part 4  Exploratory check: v1 + one "vanished at low HP" column, paired against v1.

Nothing is written to results/experiments.csv and the feature store is not touched.
Tables are printed and saved as CSV in outputs/diagnostics/.

Usage:
    python src/models/diagnose_baseline.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split

from src.config import ID_COL, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL, TEST_JSONL, TRAIN_JSONL
from src.evaluation import get_cv_splits, paired_comparison
from src.features import filter_by_prefix, load_master_features
from src.models.eval_feature_groups import T_CRIT, dropped_columns
from src.models.train_linear_svc import FEATURE_PREFIXES, N_V1_FEATURES, build_team_baseline

DIAG_DIR = OUTPUTS_DIR / "diagnostics"
FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CONFIDENT = 0.8  # p > 0.8 or p < 0.2
VANISHED_MAX_HP = 0.3

# Reading rules, fixed before looking at the results (see the plan).
FLAT_GAIN_PT = 0.2
MEMORIZATION_GAP_PT = 1.0
BALANCED_WIN_RATE = (0.35, 0.65)
CALIBRATION_TOL = 0.03
MIN_BIN_N = 100

SWITCH_BINS = ["0", "(0,0.1]", "(0.1,0.3]", ">0.3"]
VANISHED_BINS = ["(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.5]", ">0.5"]


def save(df: pd.DataFrame, name: str) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIAG_DIR / f"{name}.csv", index=False)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Battle-level KO summary (Parts 3 and 4 share the same definitions)
# ---------------------------------------------------------------------------

def _switch_bin(hp: float) -> str:
    if hp == 0:
        return "0"
    return "(0,0.1]" if hp <= 0.1 else ("(0.1,0.3]" if hp <= 0.3 else ">0.3")


def _vanished_bin(hp: float) -> str:
    for upper, label in ((0.1, "(0,0.1]"), (0.2, "(0.1,0.2]"), (0.3, "(0.2,0.3]"), (0.5, "(0.3,0.5]")):
        if hp <= upper:
            return label
    return ">0.5"


def vanished_pokemon(battle: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    """Pokémon of one side that were active, are NOT active at the last turn, never appear at HP 0.

    For each: species, last observed HP and the turn index of its last appearance. A Pokémon that
    leaves at low HP and never comes back is either saved for later or knocked out without the KO
    being recorded; P1, whose KOs are recorded, is the term of comparison.
    """
    timeline = battle["battle_timeline"]
    state_key = f"{side}_pokemon_state"
    last_hp: Dict[str, float] = {}
    last_turn: Dict[str, int] = {}
    ever_zero = set()
    for i, turn in enumerate(timeline):
        st = turn[state_key]
        last_hp[st["name"]] = st["hp_pct"]
        last_turn[st["name"]] = i
        if st["hp_pct"] == 0:
            ever_zero.add(st["name"])
    final_active = timeline[-1][state_key]["name"]
    return [
        {"name": n, "last_hp": last_hp[n], "last_turn": last_turn[n]}
        for n in last_hp
        if n != final_active and n not in ever_zero
    ]


def vanished_low_hp(battle: Dict[str, Any], side: str, max_hp: float = VANISHED_MAX_HP) -> int:
    return sum(v["last_hp"] <= max_hp for v in vanished_pokemon(battle, side))


def battle_summary(battle: Dict[str, Any]) -> Dict[str, Any]:
    timeline = battle["battle_timeline"]
    row: Dict[str, Any] = {ID_COL: battle[ID_COL]}
    for side in ("p1", "p2"):
        key = f"{side}_pokemon_state"
        row[f"{side}_ko"] = len({t[key]["name"] for t in timeline if t[key]["hp_pct"] == 0})
        row[f"{side}_any_hp0"] = int(row[f"{side}_ko"] > 0)
        for b in SWITCH_BINS:
            row[f"{side}_switch_{b}"] = 0
        for t0, t1 in zip(timeline, timeline[1:]):
            if t0[key]["name"] != t1[key]["name"]:
                row[f"{side}_switch_{_switch_bin(t0[key]['hp_pct'])}"] += 1
        vanished = vanished_pokemon(battle, side)
        for b in VANISHED_BINS:
            row[f"{side}_vanished_{b}"] = sum(_vanished_bin(v["last_hp"]) == b for v in vanished)
        row[f"{side}_vanished_low_hp"] = vanished_low_hp(battle, side)
    # Context of the first P2 Pokémon vanished at low HP, to open the battle by hand.
    low = [v for v in vanished_pokemon(battle, "p2") if v["last_hp"] <= VANISHED_MAX_HP]
    if low:
        v = min(low, key=lambda x: x["last_turn"])
        turn = timeline[v["last_turn"]]
        nxt = timeline[v["last_turn"] + 1] if v["last_turn"] + 1 < len(timeline) else None
        row["ex_turn"] = turn["turn"]
        row["ex_p2_species"] = v["name"]
        row["ex_last_hp"] = v["last_hp"]
        row["ex_p1_move"] = (turn.get("p1_move_details") or {}).get("name", "none")
        row["ex_p2_next_active"] = nxt["p2_pokemon_state"]["name"] if nxt else "none"
    if TARGET_COL in battle:
        row[TARGET_COL] = int(battle[TARGET_COL])
    return row


def summarize_jsonl(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([battle_summary(json.loads(line)) for line in f]).set_index(ID_COL)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def part1_learning_curve(X: pd.DataFrame, y: pd.Series, splits) -> dict:
    print("\n" + "=" * 78 + "\nPART 1 - Learning curve (team baseline, validation = whole fold)\n" + "=" * 78)
    rows = []
    for f in FRACTIONS:
        val_acc, train_acc, n_train = [], [], []
        for train_idx, val_idx in splits:
            if f < 1.0:
                # Stratified subsample of the training fold; the validation fold is never reduced.
                sub_idx, _ = train_test_split(train_idx, train_size=f, stratify=y.iloc[train_idx],
                                              random_state=RANDOM_STATE)
            else:
                sub_idx = train_idx
            model = build_team_baseline().fit(X.iloc[sub_idx], y.iloc[sub_idx])
            train_acc.append(model.score(X.iloc[sub_idx], y.iloc[sub_idx]))
            val_acc.append(model.score(X.iloc[val_idx], y.iloc[val_idx]))
            n_train.append(len(sub_idx))
        rows.append({"fraction": f, "n_train": int(np.mean(n_train)),
                     "val_mean": np.mean(val_acc), "val_std": np.std(val_acc),
                     "train_mean": np.mean(train_acc), "gap_pt": (np.mean(train_acc) - np.mean(val_acc)) * 100})
    curve = pd.DataFrame(rows)
    save(curve, "learning_curve")
    fmt = curve.assign(val=lambda d: (d.val_mean * 100).map("{:.2f}".format) + " +/- " + (d.val_std * 100).map("{:.2f}".format),
                       train=lambda d: (d.train_mean * 100).map("{:.2f}".format),
                       gap_pt=lambda d: d.gap_pt.map("{:.2f}".format))[["fraction", "n_train", "val", "train", "gap_pt"]]
    show("Accuracy (%) by fraction of the training fold:", fmt)
    gain = (curve.val_mean.iloc[-1] - curve.val_mean.iloc[-2]) * 100
    gap = curve.gap_pt.iloc[-1]
    return {"gain_80_100_pt": gain, "flat": gain < FLAT_GAIN_PT, "gap_pt": gap, "no_memorization": gap < MEMORIZATION_GAP_PT,
            "val_full": curve.val_mean.iloc[-1]}


def part2_errors(X: pd.DataFrame, y: pd.Series, splits) -> dict:
    print("\n" + "=" * 78 + "\nPART 2 - Error analysis on out-of-fold probabilities\n" + "=" * 78)
    p = cross_val_predict(build_team_baseline(), X, y, cv=splits, method="predict_proba", n_jobs=-1)[:, 1]
    pred = (p > 0.5).astype(int)
    err = pred != y.values
    df = pd.DataFrame({ID_COL: X.index, TARGET_COL: y.values, "p": p, "pred": pred, "error": err,
                       "diff_alive": X["tl_diff_alive_count"].values, "diff_total_hp": X["tl_diff_total_hp"].values})
    save(df.drop(columns="error"), "oof_probs")
    print(f"OOF accuracy: {1 - err.mean():.4%} | log-loss {log_loss(y, p):.4f} | Brier {brier_score_loss(y, p):.4f}")

    # 1. By advantage at turn 30
    g = df.groupby("diff_alive").agg(n=("error", "size"), accuracy=("error", lambda e: 1 - e.mean()),
                                     p1_win_rate=(TARGET_COL, "mean"), errors=("error", "sum")).reset_index()
    g["error_share"] = g.errors / err.sum()
    g["battle_share"] = g.n / len(df)
    save(g, "errors_by_diff_alive")
    show("1. By tl_diff_alive_count (P1 alive - P2 alive at turn 30):", g.round(4))
    balanced = g.p1_win_rate.between(*BALANCED_WIN_RATE)
    bal_err, bal_n = g.error_share[balanced].sum(), g.battle_share[balanced].sum()
    print(f"   Balanced groups (P1 win rate in {BALANCED_WIN_RATE}): {g.diff_alive[balanced].tolist()} -> "
          f"{bal_n:.1%} of battles, {bal_err:.1%} of errors")

    zero = df[df.diff_alive == 0].copy()
    zero["hp_bin"] = pd.qcut(zero.diff_total_hp, 5)
    z = zero.groupby("hp_bin", observed=True).agg(n=("error", "size"), accuracy=("error", lambda e: 1 - e.mean()),
                                                  p1_win_rate=(TARGET_COL, "mean")).reset_index()
    z["hp_bin"] = z.hp_bin.astype(str)
    save(z, "errors_alive0_by_total_hp")
    show("   Deep dive, diff_alive = 0, by quintile of tl_diff_total_hp:", z.round(4))

    # 2. Calibration
    df["p_bin"] = pd.cut(df.p, np.linspace(0, 1, 11), include_lowest=True)
    c = df.groupby("p_bin", observed=False).agg(n=("p", "size"), mean_p=("p", "mean"),
                                                observed_win_rate=(TARGET_COL, "mean")).reset_index()
    c["p_bin"] = c.p_bin.astype(str)
    c["gap"] = c.observed_win_rate - c.mean_p
    c["reliable"] = np.where(c.n >= MIN_BIN_N, "yes", "NO (<100)")
    save(c, "calibration")
    show("2. Calibration (predicted probability vs observed P1 win rate):", c.round(4))
    reliable = c[c.n >= MIN_BIN_N]
    max_gap = reliable.gap.abs().max()

    # 3. Confident errors vs what a calibrated model would make
    conf = df[(df.p > CONFIDENT) | (df.p < 1 - CONFIDENT)]
    q = np.minimum(conf.p, 1 - conf.p)
    expected, sd = q.sum(), np.sqrt((q * (1 - q)).sum())
    observed = int(conf.error.sum())
    conf_err = conf[conf.error].sort_values("p")
    save(conf_err.drop(columns=["p_bin", "error"]), "confident_errors")
    print(f"\n3. Confident predictions (p > {CONFIDENT} or p < {1 - CONFIDENT:.1f}): {len(conf)} battles | "
          f"errors observed {observed} vs expected if calibrated {expected:.1f} +/- {sd:.1f}")
    ex = pd.concat([conf_err.head(5), conf_err.tail(5)])[[ID_COL, "p", TARGET_COL, "diff_alive", "diff_total_hp"]]
    show("   10 examples (most confident in each direction):", ex.round(3))

    return {"balanced_groups": g.diff_alive[balanced].tolist(), "bal_err": bal_err, "bal_n": bal_n,
            "concentrated": bal_err > bal_n, "max_calib_gap": max_gap, "calibrated": max_gap <= CALIBRATION_TOL,
            "conf_obs": observed, "conf_exp": expected, "conf_sd": sd, "conf_far_above": observed > expected + 2 * sd}


def part3_ko_asymmetry(train_sum: pd.DataFrame, test_sum: pd.DataFrame) -> dict:
    print("\n" + "=" * 78 + "\nPART 3 - KO asymmetry (raw JSONL, inputs only)\n" + "=" * 78)

    # 1 + 3b. Recorded KOs and switches, per battle, train vs test
    rows = []
    for side in ("p1", "p2"):
        rows.append({"measure": f"{side}: battles with a Pokémon at HP 0",
                     "train_count": int(train_sum[f"{side}_any_hp0"].sum()),
                     "train_per_battle": train_sum[f"{side}_any_hp0"].mean(), "test_per_battle": test_sum[f"{side}_any_hp0"].mean()})
        rows.append({"measure": f"{side}: unique KOs recorded",
                     "train_count": int(train_sum[f"{side}_ko"].sum()),
                     "train_per_battle": train_sum[f"{side}_ko"].mean(), "test_per_battle": test_sum[f"{side}_ko"].mean()})
        for b in SWITCH_BINS:
            col = f"{side}_switch_{b}"
            rows.append({"measure": f"{side}: switches after HP {b}", "train_count": int(train_sum[col].sum()),
                         "train_per_battle": train_sum[col].mean(), "test_per_battle": test_sum[col].mean()})
    t1 = pd.DataFrame(rows)
    save(t1, "ko_recorded_train_test")
    show("1/3b. Recorded KOs and switches by HP of the previous turn (per battle: train vs test):", t1.round(3))

    # 2. P1 win rate by recorded KO difference
    d = (train_sum.p2_ko - train_sum.p1_ko).clip(-3, 2)
    t2 = train_sum.assign(ko_diff=d).groupby("ko_diff").agg(n=(TARGET_COL, "size"), p1_win_rate=(TARGET_COL, "mean")).reset_index()
    save(t2, "win_rate_by_ko_diff")
    show("2. P1 win rate by (P2 KOs - P1 KOs recorded), clipped to [-3, +2]:", t2.round(4))

    # 3 + 3b. Vanished Pokémon by last HP
    rows = []
    for b in VANISHED_BINS:
        rows.append({"last_hp": b,
                     "p1_train": train_sum[f"p1_vanished_{b}"].mean(), "p2_train": train_sum[f"p2_vanished_{b}"].mean(),
                     "p1_test": test_sum[f"p1_vanished_{b}"].mean(), "p2_test": test_sum[f"p2_vanished_{b}"].mean()})
    t3 = pd.DataFrame(rows)
    t3["p2_over_p1_train"] = t3.p2_train / t3.p1_train
    t3["p2_over_p1_test"] = t3.p2_test / t3.p1_test
    save(t3, "vanished_by_last_hp")
    show("3/3b. Vanished Pokémon per battle by last observed HP (not active at turn 30, never at HP 0):", t3.round(3))

    k = train_sum.p2_vanished_low_hp.clip(upper=3)
    t4 = train_sum.assign(p2_vanished_low=k).groupby("p2_vanished_low").agg(
        n=(TARGET_COL, "size"), p1_win_rate=(TARGET_COL, "mean")).reset_index()
    save(t4, "win_rate_by_p2_vanished_low")
    show(f"   P1 win rate by number of P2 Pokémon vanished with last HP <= {VANISHED_MAX_HP} (3 = 3 or more):", t4.round(4))

    # 4. Examples to open by hand
    ex = train_sum[train_sum.ex_turn.notna()].head(5).reset_index()
    ex = ex[[ID_COL, "ex_turn", "ex_p2_species", "ex_last_hp", "ex_p1_move", "ex_p2_next_active", TARGET_COL]]
    save(ex, "vanished_examples")
    show("4. Examples of P2 Pokémon vanished at low HP (turn of last appearance, P1 move that turn, next P2 active):", ex)

    low_bins = VANISHED_BINS[:3]
    ratio_train = t3[t3.last_hp.isin(low_bins)].p2_train.sum() / t3[t3.last_hp.isin(low_bins)].p1_train.sum()
    ratio_test = t3[t3.last_hp.isin(low_bins)].p2_test.sum() / t3[t3.last_hp.isin(low_bins)].p1_test.sum()
    return {"low_ratio_train": ratio_train, "low_ratio_test": ratio_test,
            "win_by_vanished": dict(zip(t4.p2_vanished_low, t4.p1_win_rate))}


def part4_correction(X: pd.DataFrame, y: pd.Series, splits, train_sum: pd.DataFrame, v1_scores: np.ndarray) -> dict:
    print("\n" + "=" * 78 + "\nPART 4 - Exploratory correction: v1 + one vanished-at-low-HP column\n" + "=" * 78)
    n_train, n_test = len(splits[0][0]), len(splits[0][1])
    extra = pd.DataFrame({
        "p2_vanished_low_hp": train_sum.p2_vanished_low_hp,
        "diff_vanished_low_hp": train_sum.p2_vanished_low_hp - train_sum.p1_vanished_low_hp,
    }).loc[X.index]
    assert not extra.isna().any().any(), "vanished columns must cover every training battle"

    print(f"v1 alone: {v1_scores.mean():.4%} +/- {v1_scores.std():.4%}")
    results = {}
    for col in extra.columns:
        Xc = X.join(extra[[col]])  # appended last: DropCollinear cannot remove a v1 column because of it
        cols = list(Xc.columns)
        out = cross_validate(build_team_baseline(), Xc, y, cv=splits, scoring="accuracy", return_estimator=True, n_jobs=-1)
        pc = paired_comparison(out["test_score"], v1_scores, n_train=n_train, n_test=n_test)
        keep = pc["mean_d"] > 0 and pc["t_corrected"] >= T_CRIT
        dropped = [dropped_columns(est, cols, Xc.iloc[tr]) for est, (tr, _) in zip(out["estimator"], splits)]
        n_dropped = sum(col in d for d in dropped)
        print(f"\nv1 + {col}: {out['test_score'].mean():.4%} +/- {out['test_score'].std():.4%}")
        print(f"    d = {pc['mean_d'] * 100:+.2f} +/- {pc['std_d'] * 100:.2f} pt | W/T/L {pc['wins']}/{pc['ties']}/{pc['losses']} | "
              f"t_corr = {pc['t_corrected']:.2f} (crit {T_CRIT:.3f}), p = {pc['p_value']:.4f} | {'KEEP' if keep else 'DROP'}")
        print(f"    column dropped by the baseline preprocessing in {n_dropped}/{len(splits)} folds")
        results[col] = {**pc, "keep": keep, "acc": out["test_score"].mean()}
    print("\nLimits: the 0.3 threshold was chosen after looking at the exploration data, and the columns are "
          "evaluated on the same folds used for that choice.")
    return results


def run() -> None:
    print("=" * 78)
    print("Predex - diagnosis of the team logistic baseline on feature set v1 (person B)")
    print("=" * 78)
    X, y = load_master_features("train")
    X = filter_by_prefix(X, include_prefixes=FEATURE_PREFIXES)
    assert X.shape[1] == N_V1_FEATURES, f"v1 must have {N_V1_FEATURES} columns, got {X.shape[1]}"
    splits = get_cv_splits(X)

    r1 = part1_learning_curve(X, y, splits)
    r2 = part2_errors(X, y, splits)

    train_sum = summarize_jsonl(TRAIN_JSONL)
    test_sum = summarize_jsonl(TEST_JSONL)
    r3 = part3_ko_asymmetry(train_sum, test_sum)

    v1_scores = cross_validate(build_team_baseline(), X, y, cv=splits, scoring="accuracy", n_jobs=-1)["test_score"]
    r4 = part4_correction(X, y, splits, train_sum, v1_scores)

    print("\n" + "=" * 78 + "\nREADING RULES (fixed before the run)\n" + "=" * 78)
    print(f"Learning curve: gain 80%->100% = {r1['gain_80_100_pt']:+.2f} pt -> {'FLAT' if r1['flat'] else 'STILL RISING'}; "
          f"train-val gap = {r1['gap_pt']:.2f} pt -> {'no memorization' if r1['no_memorization'] else 'memorization'}")
    print(f"Errors: balanced groups {r2['balanced_groups']} hold {r2['bal_n']:.1%} of battles and {r2['bal_err']:.1%} of errors "
          f"-> {'CONCENTRATED' if r2['concentrated'] else 'not concentrated'}")
    print(f"Calibration: max |gap| in bins with >= {MIN_BIN_N} battles = {r2['max_calib_gap']:.3f} "
          f"-> {'CALIBRATED' if r2['calibrated'] else 'NOT calibrated'}")
    print(f"Confident errors: {r2['conf_obs']} observed vs {r2['conf_exp']:.1f} +/- {r2['conf_sd']:.1f} expected "
          f"-> {'FAR ABOVE expected' if r2['conf_far_above'] else 'in line with calibration'}")
    print(f"KO asymmetry: P2/P1 ratio of Pokémon vanished at HP <= 0.3 = {r3['low_ratio_train']:.2f} (train), "
          f"{r3['low_ratio_test']:.2f} (test)")
    for col, r in r4.items():
        print(f"Correction v1+{col}: d = {r['mean_d'] * 100:+.2f} pt, t_corr = {r['t_corrected']:.2f}, "
              f"p = {r['p_value']:.4f} -> {'KEEP' if r['keep'] else 'DROP'}")
    print(f"\nCSV tables saved in {DIAG_DIR}")


if __name__ == "__main__":
    run()
