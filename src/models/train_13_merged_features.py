"""PASSO 13 — LightGBM sul feature store del team + feature sm_ (persona D).

Domanda: le feature del team (rf_, mit_: meccaniche di gioco) e le nostre (sm_: specie e mosse)
si sommano? Stesso modello (LightGBM con i parametri di Optuna), stessi fold, cambia solo il set:

  v2           = bs_ + tl_ + sm_                         (il nostro set attuale, 324)
  team         = bs_ + tl_ + rf_ + mit_                  (il set del team, 522)
  team+sm      = team + sm_                              (794)
  team+sm-ext  = team+sm senza le colonne di viability Smogon (dato esterno, vietato)

Ogni set viene valutato sui fold del team e sulla seconda divisione (seed 2026).

Esecuzione, dalla cartella Predex-team, con il venv di Predex-main attivo:
    python -m src.features.registry --force       # deve dire "Saved 794 features"
    python src/models/train_13_merged_features.py
Durata indicativa: 5-10 minuti.
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import ID_COL, RANDOM_STATE, TARGET_COL
from src.cross_validation import load_folds
from src.features import load_master_features
from src.models.evaluation import log_result, run_cv, save_oof

warnings.filterwarnings("ignore")

# Parametri trovati da Optuna al Passo 10 (sulle feature v2)
LGBM_OPTUNA_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.026094297167793386, "num_leaves": 20,
    "min_child_samples": 28, "subsample": 0.8325152611459203, "subsample_freq": 1,
    "colsample_bytree": 0.746083812111282, "reg_lambda": 0.22914798719789617,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
}


def feature_sets(columns):
    pre = lambda p: [c for c in columns if c.startswith(p)]
    external = [c for c in columns if "viability" in c]          # indice Smogon: dato esterno
    team = pre("bs_") + pre("tl_") + pre("rf_") + pre("mit_")
    return {
        "v2": pre("bs_") + pre("tl_") + pre("sm_"),
        "team": team,
        "team+sm": team + pre("sm_"),
        "team+sm-ext": [c for c in team + pre("sm_") if c not in external],
    }, external


def alt_folds(index, y):
    alt = pd.DataFrame({ID_COL: index, TARGET_COL: y.values, "fold": -1})
    for k, (_, va) in enumerate(StratifiedKFold(10, shuffle=True, random_state=2026).split(alt, alt[TARGET_COL])):
        alt.loc[va, "fold"] = k
    return alt


def main():
    X, y = load_master_features("train")
    sets, external = feature_sets(list(X.columns))
    if not sets["v2"] or len(sets["team+sm"]) < 700:
        raise SystemExit(f"Feature mancanti ({X.shape[1]} colonne): rigenera il Parquet con "
                         "'python -m src.features.registry --force' (servono anche le sm_).")
    print({k: len(v) for k, v in sets.items()}, "| colonne esterne escluse:", external, "\n")

    folds = {"team": load_folds(), "alt": alt_folds(X.index, y)}
    print(f"{'feature set':<14}{'n':>5}{'fold team: acc':>18}{'logloss':>9}{'2a divisione: acc':>20}{'logloss':>9}{'distanza':>10}")
    for name, cols in sets.items():
        res = {k: run_cv(lambda: LGBMClassifier(**LGBM_OPTUNA_PARAMS), X[cols], y,
                         name=f"13_lgbm_{name}_{k}", folds_df=fd, verbose=False)
               for k, fd in folds.items()}
        save_oof(res["team"], X[cols], y)
        log_result(res["team"], params=f"LightGBM Optuna (passo 10), feature set {name}")
        t, a = res["team"], res["alt"]
        print(f"{name:<14}{len(cols):>5}{t['acc_mean']:>12.4f} ± {t['acc_std']:.3f}{t['logloss']:>9.4f}"
              f"{a['acc_mean']:>14.4f} ± {a['acc_std']:.3f}{a['logloss']:>9.4f}{t['train_acc'] - t['acc_mean']:>10.3f}",
              flush=True)


if __name__ == "__main__":
    main()
