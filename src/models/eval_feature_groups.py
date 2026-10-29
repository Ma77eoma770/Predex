"""Evaluate feature groups on top of feature set v1 with the team's logistic baseline.

Each group is added to v1 alone (v1+T, v1+O, v1+D, v1+A) and compared with v1 fold by fold
on the frozen 10 folds. A group is kept if its mean gain is positive and the Nadeau-Bengio
corrected paired t-test gives t >= T_CRIT (one-sided p < 0.10). The kept groups are then
evaluated together against v1 with the same test.

The groups come from src/features/experimental (not in the default feature store): they are
computed on the fly from train.jsonl with the registry code and joined to v1 by battle_id.

For every set the script also reports how many new columns survive the baseline's own
preprocessing (VarianceThreshold + DropCollinear 0.95) and, for each dropped one, why.

Usage:
    python src/models/eval_feature_groups.py          # print only
    python src/models/eval_feature_groups.py --log    # also append one row per set to results/experiments.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_validate

from src.config import TRAIN_JSONL
from src.evaluation import get_cv_splits, log_result, paired_comparison
from src.features import load_master_features
from src.features.experimental import register_experimental
from src.features.registry import build_feature_dataframe
from src.models.train_linear_svc import FEATURE_PREFIXES, N_V1_FEATURES, PERSON, build_team_baseline

GROUPS = {
    "T": ["tm_"],      # Player 1 team composition
    "O": ["ty_off_"],  # offensive matchup of the moves used
    "D": ["ty_def_"],  # P1 team vs the types P2 has shown
    "A": ["ty_act_"],  # active Pokémon matchup at the last turn
}
SIGNIFICANCE = 0.10
# One-sided critical value of Student's t with k - 1 = 9 degrees of freedom (= 1.383).
T_CRIT = float(stats.t.ppf(1 - SIGNIFICANCE, df=9))
COLLINEAR_THRESHOLD = 0.95  # same value as the team baseline


def columns_with(X: pd.DataFrame, prefixes: List[str]) -> List[str]:
    return [c for c in X.columns if c.startswith(tuple(prefixes))]


def evaluate(X: pd.DataFrame, y: pd.Series, cols: List[str], splits) -> dict:
    out = cross_validate(build_team_baseline(), X[cols], y, cv=splits, scoring="accuracy",
                         return_estimator=True, n_jobs=-1)
    return {"scores": out["test_score"], "estimators": out["estimator"], "cols": cols}


def dropped_columns(estimator, cols: List[str], X_train: pd.DataFrame) -> Dict[str, str]:
    """Columns removed by the baseline preprocessing of one fitted fold, with the reason."""
    prep = estimator["preprocessor"]
    after_variance = [c for c, keep in zip(cols, prep["variance_filter"].get_support()) if keep]
    reasons = {c: "variance ~0" for c in cols if c not in after_variance}

    # DropCollinearFeatures sees the numpy output of VarianceThreshold, so its labels are positions.
    to_drop = prep["collinear_filter"].to_drop_
    drop_names = [after_variance[i] if isinstance(i, (int, np.integer)) else i for i in to_drop]
    if drop_names:
        # Same rule as DropCollinearFeatures: a column is dropped if it correlates above the
        # threshold with any EARLIER column; report the strongest such partner.
        corr = X_train[after_variance].corr()
        for name in drop_names:
            earlier = after_variance[: after_variance.index(name)]
            partner = corr.loc[earlier, name].abs().idxmax()
            reasons[name] = f"collinear with {partner} (r={corr.loc[partner, name]:+.3f})"
    return reasons


def report_survival(result: dict, v1_drops_per_fold: List[set], X: pd.DataFrame, splits, new_cols: List[str]) -> List[str]:
    """Print how many new columns survive the preprocessing and why the others are dropped."""
    survived, drop_counts, drop_reasons = [], defaultdict(int), {}
    extra_v1_drops = set()
    for fold, (est, (train_idx, _)) in enumerate(zip(result["estimators"], splits)):
        reasons = dropped_columns(est, result["cols"], X.iloc[train_idx])
        new_dropped = [c for c in reasons if c in new_cols]
        survived.append(len(new_cols) - len(new_dropped))
        for c in new_dropped:
            drop_counts[c] += 1
            drop_reasons[c] = reasons[c]
        # Adding columns must not remove any v1 column that v1 alone keeps.
        extra_v1_drops |= {c for c in reasons if c not in new_cols} - v1_drops_per_fold[fold]

    lo, hi = min(survived), max(survived)
    span = f"{lo}" if lo == hi else f"{lo}-{hi}"
    print(f"    new columns surviving preprocessing: {span}/{len(new_cols)} (over {len(splits)} folds)")
    for c, n in sorted(drop_counts.items()):
        print(f"      dropped {c} in {n}/{len(splits)} folds: {drop_reasons[c]}")
    if extra_v1_drops:
        print(f"    WARNING: v1 columns dropped only because of the new group: {sorted(extra_v1_drops)}")
    return sorted(drop_counts)


def print_comparison(name: str, result: dict, paired: dict, keep: bool) -> None:
    s = result["scores"]
    print(f"\n{name}: {len(result['cols'])} features | accuracy {s.mean():.4%} +/- {s.std():.4%}")
    print(f"    vs v1: d = {paired['mean_d'] * 100:+.2f} +/- {paired['std_d'] * 100:.2f} pt | "
          f"W/T/L {paired['wins']}/{paired['ties']}/{paired['losses']} | "
          f"t_corr = {paired['t_corrected']:.2f} (crit {T_CRIT:.3f}), p = {paired['p_value']:.3f} | "
          f"{'KEEP' if keep else 'DROP'}")


def run(log: bool) -> None:
    print("=" * 78)
    print("Predex - feature groups on top of v1 (team logistic baseline, person B)")
    print("=" * 78)

    X, y = load_master_features("train")
    # The groups live in src/features/experimental, outside the default feature store:
    # compute them on the fly with the same registry code and join them by battle_id.
    X_exp, _ = build_feature_dataframe(TRAIN_JSONL, is_train=True, modules=register_experimental(),
                                       show_progress=False)
    X = X.join(X_exp, how="inner", validate="one_to_one")
    assert len(X) == len(y), "experimental features must cover exactly the training battles"
    splits = get_cv_splits(X)
    n_train, n_test = len(splits[0][0]), len(splits[0][1])

    # Explicit column order: v1 first, then the new group. The parquet stores modules in
    # alphabetical order (tm_ before tl_), and DropCollinear removes the LATER column of a
    # collinear pair, so with the parquet order it could drop a v1 column instead of a new one.
    v1_cols = columns_with(X, FEATURE_PREFIXES)
    assert len(v1_cols) == N_V1_FEATURES, f"v1 must have {N_V1_FEATURES} columns, got {len(v1_cols)}"

    v1 = evaluate(X, y, v1_cols, splits)
    v1_drops = [set(dropped_columns(est, v1_cols, X.iloc[tr])) for est, (tr, _) in zip(v1["estimators"], splits)]
    print(f"\nv1: {len(v1_cols)} features | accuracy {v1['scores'].mean():.4%} +/- {v1['scores'].std():.4%} "
          f"(reference: 66.56%) | v1 columns dropped by the baseline itself: {sorted(set().union(*v1_drops))}")

    rows = [("v1", v1, None, None)]
    kept_groups = []
    for group, prefixes in GROUPS.items():
        new_cols = columns_with(X, prefixes)
        result = evaluate(X, y, v1_cols + new_cols, splits)
        paired = paired_comparison(result["scores"], v1["scores"], n_train=n_train, n_test=n_test)
        keep = paired["mean_d"] > 0 and paired["t_corrected"] >= T_CRIT
        print_comparison(f"v1+{group}", result, paired, keep)
        report_survival(result, v1_drops, X, splits, new_cols)
        rows.append((f"v1+{group}", result, paired, keep))
        if keep:
            kept_groups.append(group)

    print("\n" + "-" * 78)
    if not kept_groups:
        print("No group passes the criterion: feature set v1 stays as it is.")
    else:
        # Final check: all kept groups together. Same folds used for the selection, so optimistic.
        name = "v1+" + "+".join(kept_groups)
        new_cols = columns_with(X, [p for g in kept_groups for p in GROUPS[g]])
        result = evaluate(X, y, v1_cols + new_cols, splits)
        paired = paired_comparison(result["scores"], v1["scores"], n_train=n_train, n_test=n_test)
        keep = paired["mean_d"] > 0 and paired["t_corrected"] >= T_CRIT
        print(f"Final check with all kept groups ({', '.join(kept_groups)}):")
        print_comparison(name, result, paired, keep)
        report_survival(result, v1_drops, X, splits, new_cols)
        rows.append((name, result, paired, keep))
    print("Note: groups are selected and checked on the same folds, so the final gain is optimistic.")

    if not log:
        print("\n[--log not set] results/experiments.csv left untouched.")
        return

    # The v1 reference is already in the table (team baseline row), so only the new sets are logged.
    logged = [r for r in rows if r[2] is not None]
    for name, result, paired, keep in logged:
        notes = (
            f"{'tenuto' if keep else 'scartato'}, t corretta = {paired['t_corrected']:.2f}; "
            f"appaiato vs v1 d={paired['mean_d'] * 100:+.2f}±{paired['std_d'] * 100:.2f} pt, "
            f"{paired['wins']}W/{paired['ties']}P/{paired['losses']}L, p={paired['p_value']:.3f}")
        log_result(
            model="LogisticRegression (team baseline)",
            feature_version=name.replace("v1", "v1_bs_tl", 1),
            hyperparams={"C": 1.0, "solver": "lbfgs", "max_iter": 5000,
                         "prep": "create_linear_preprocessing_pipeline(drop_collinear=True, 0.95)",
                         "n_features": len(result["cols"])},
            cv_mean_acc=result["scores"].mean(),
            cv_std_acc=result["scores"].std(),
            person=PERSON,
            n_folds=len(splits),
            notes=notes,
        )
    print(f"\n[--log] {len(logged)} rows appended to results/experiments.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate feature groups on top of v1")
    parser.add_argument("--log", action="store_true", help="Append the results to results/experiments.csv")
    args = parser.parse_args()
    run(log=args.log)
