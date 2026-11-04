"""PASSO 19 — Modello finale con le feature di Max (persona D, con il lavoro di Max).

Rispetto al Passo 16:
  1. LightGBM (5 seed, parametri del Passo 14) riceve anche i gruppi "field" e "threat" di Max
     (33 feature: spikes, rapid spin, schermi, boost del Pokémon attivo al turno 30).
  2. La media ha tre membri invece di due, con pesi fissati a priori:
        0,70 LightGBM + 0,15 nostra logistica (786 + bravura)
                      + 0,15 logistica di Max (sue 122 feature + bravura, StandardScaler, C=0,01)
     La logistica di Max sbaglia su battaglie diverse dalla nostra (correlazione 0,78), quindi
     la media guadagna.
Le 5 feature di bravura si calcolano sempre dentro i fold (usano le etichette).

Esecuzione dalla cartella Predex-team (venv attivo):
    python src/models/train_19_final_max.py 2>&1 | tee outputs/log_19.txt
    python src/models/train_19_final_max.py --no-cv     # solo submission
Scrive in outputs/:
    submission_19_blend3.csv          (media a tre)
    submission_19_lgbm5_max.csv    (solo LightGBM 5 seed)
"""

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ID_COL, OUTPUTS_DIR, TARGET_COL, TEST_JSONL, TRAIN_JSONL
from src.cross_validation import load_folds
from src.features import load_master_features
from src.features.max_features import build as max_build
from src.features.player_skill import SKILL_COLUMNS, read_teams, skill_features
from src.models.train_13_merged_features import alt_folds, feature_sets
from src.models.train_15_final import make_lgbm, make_logit, write_submission

warnings.filterwarnings("ignore")

MAX_LGBM = ("field", "threat")  # versione usata per submission_19 (solo "field" dava risultati equivalenti in explore_batch7)
MAX_ALL = ("material", "status", "moves", "field", "threat")
WEIGHTS = {"lgbm": 0.70, "logit": 0.15, "logit_fl": 0.15}
FL_IN_LGBM: list = []  # colonne di Max passate a LightGBM (riempita in main)


def max_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, random_state=42))


def max_frame(jsonl_path, index):
    """Le 122 feature di Max, nell'ordine di index, con prefisso fl_; più i nomi per gruppo."""
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]
    groups = {}
    for g in MAX_ALL:
        cols = [c for c in max_build(data[:1], groups=(g,)).columns if c != "battle_id"]
        groups[g] = [f"fl_{c}" for c in cols]
    F = max_build(data, groups=MAX_ALL).set_index("battle_id").loc[index]
    F.columns = [f"fl_{c}" for c in F.columns]
    return F, groups


def with_skill(X, S):
    return X.assign(**{c: S[:, i] for i, c in enumerate(SKILL_COLUMNS)})


def fit_predict(Xg_tr, Xg_va, Fl_tr, Fl_va, y_tr, S_tr, S_va):
    """Allena i tre membri e restituisce le loro probabilità sul blocco di validazione."""
    return {
        "lgbm": make_lgbm().fit(with_skill(Xg_tr, S_tr), y_tr).predict_proba(with_skill(Xg_va, S_va))[:, 1],
        "logit": make_logit().fit(with_skill(Xg_tr.drop(columns=FL_IN_LGBM), S_tr), y_tr)
                             .predict_proba(with_skill(Xg_va.drop(columns=FL_IN_LGBM), S_va))[:, 1],
        "logit_fl": max_logit().fit(with_skill(Fl_tr, S_tr), y_tr).predict_proba(with_skill(Fl_va, S_va))[:, 1],
    }


def blend(p):
    return sum(w * p[m] for m, w in WEIGHTS.items())


def cv(Xg, Fl, y, teams, folds):
    yv = y.to_numpy()
    oof = {m: np.zeros(len(Xg)) for m in WEIGHTS}
    for k in range(10):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        S_tr, S_va = skill_features([teams[i] for i in tr], yv[tr], [teams[i] for i in va])
        p = fit_predict(Xg.iloc[tr], Xg.iloc[va], Fl.iloc[tr], Fl.iloc[va], yv[tr], S_tr, S_va)
        for m in WEIGHTS:
            oof[m][va] = p[m]
        print(f"    fold {k} fatto", flush=True)
    oof["media 0,75/0,25 (come il 16)"] = 0.75 * oof["lgbm"] + 0.25 * oof["logit"]
    oof["MEDIA A TRE"] = blend(oof)
    return oof


def report(split, oof, y, folds):
    yv = y.to_numpy()
    for m, p in oof.items():
        accs = [((p[folds == k] >= 0.5) == yv[folds == k]).mean() for k in range(10)]
        print(f"  {split:<18}{m:<30} acc {np.mean(accs):.4f} ± {np.std(accs):.4f} "
              f"log-loss {log_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)):.4f}", flush=True)


def main():
    global FL_IN_LGBM
    X_all, y = load_master_features("train")
    X_test_all, _ = load_master_features("test")
    cols = feature_sets(list(X_all.columns))[0]["team+sm-ext"]
    X, X_test = X_all[cols], X_test_all[cols]
    Fl, groups = max_frame(TRAIN_JSONL, X.index)
    Fl_test, _ = max_frame(TEST_JSONL, X_test.index)
    FL_IN_LGBM = [c for g in MAX_LGBM for c in groups[g]]
    Xg, Xg_test = pd.concat([X, Fl[FL_IN_LGBM]], axis=1), pd.concat([X_test, Fl_test[FL_IN_LGBM]], axis=1)

    tr_map, te_map = read_teams(TRAIN_JSONL), read_teams(TEST_JSONL)
    teams = [tr_map[b] for b in X.index]
    teams_test = [te_map[b] for b in X_test.index]
    print(f"LightGBM: {X.shape[1]} + {len(FL_IN_LGBM)} di Max + {len(SKILL_COLUMNS)} di bravura | "
          f"logistica di Max: {Fl.shape[1]} + {len(SKILL_COLUMNS)}\n", flush=True)

    if "--no-cv" not in sys.argv:
        for split, folds in (("fold del team", load_folds()["fold"].to_numpy()),
                             ("seconda divisione", alt_folds(X.index, y)["fold"].to_numpy())):
            oof = cv(Xg, Fl, y, teams, folds)
            report(split, oof, y, folds)
            if split.startswith("fold"):
                pd.DataFrame({ID_COL: X.index, TARGET_COL: y.values,
                              **{f"p_{m}": oof[m] for m in WEIGHTS}, "p_blend3": oof["MEDIA A TRE"]}
                             ).to_csv(OUTPUTS_DIR / "oof_19_max.csv", index=False)

    # Riallenamento su tutto il train: bravura del train con divisione interna, del test dal train intero
    S_fit, S_test = skill_features(teams, y.to_numpy(), teams_test)
    p = fit_predict(Xg, Xg_test, Fl, Fl_test, y.to_numpy(), S_fit, S_test)
    p_blend = blend(p)
    s1 = write_submission(p_blend, X_test.index, "submission_19_blend3.csv")
    s2 = write_submission(p["lgbm"], X_test.index, "submission_19_lgbm5_max.csv")
    pd.DataFrame({ID_COL: X_test.index, **p, "blend3": p_blend}).to_csv(OUTPUTS_DIR / "test_probs_19.csv", index=False)
    old = OUTPUTS_DIR / "submission_16_blend_skill.csv"
    if old.exists():
        same = (pd.read_csv(old).set_index(ID_COL).loc[X_test.index, TARGET_COL].to_numpy() == s1[TARGET_COL].to_numpy()).mean()
        print(f"\nPrevisioni uguali alla submission 16 (0,696 sul pubblico): {same:.1%}")
    print(f"Submission scritte. Quota 'vince P1': media a tre {s1[TARGET_COL].mean():.1%}, lgbm {s2[TARGET_COL].mean():.1%}")


if __name__ == "__main__":
    main()
