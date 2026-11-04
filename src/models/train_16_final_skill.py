"""PASSO 16 — Modello finale con le feature di bravura dei giocatori (persona D).

Come il Passo 15 (786 feature, LightGBM 5 seed + logistica, media 0,75/0,25), con in più le
5 feature di src/features/player_skill.py, calcolate DENTRO ogni fold (usano le etichette).
Valuta sulle due divisioni dei fold, poi riallena su tutto il train e scrive in outputs/:
  submission_16_blend_skill.csv    (LightGBM 5 seed + logistica)
  submission_16_lgbm5_skill.csv    (solo LightGBM 5 seed)

Esecuzione dalla cartella Predex-team (venv attivo):
    python src/models/train_16_final_skill.py            # validazione + submission
    python src/models/train_16_final_skill.py --no-cv    # solo submission
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.config import ID_COL, OUTPUTS_DIR, TARGET_COL, TEST_JSONL, TRAIN_JSONL
from src.cross_validation import load_folds
from src.features import load_master_features
from src.features.player_skill import SKILL_COLUMNS, read_teams, skill_features
from src.models.train_13_merged_features import alt_folds, feature_sets
from src.models.train_15_final import W_LGBM, make_lgbm, make_logit, write_submission

warnings.filterwarnings("ignore")


def with_skill(X, F):
    return X.assign(**{c: F[:, i] for i, c in enumerate(SKILL_COLUMNS)})


def cv(X, y, teams, folds):
    yv = y.to_numpy()
    oof = {"lgbm": np.zeros(len(X)), "logit": np.zeros(len(X))}
    for k in range(10):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        Ftr, Fva = skill_features([teams[i] for i in tr], yv[tr], [teams[i] for i in va])
        Xtr, Xva = with_skill(X.iloc[tr], Ftr), with_skill(X.iloc[va], Fva)
        oof["lgbm"][va] = make_lgbm().fit(Xtr, y.iloc[tr]).predict_proba(Xva)[:, 1]
        oof["logit"][va] = make_logit().fit(Xtr, y.iloc[tr]).predict_proba(Xva)[:, 1]
        print(f"    fold {k} fatto", flush=True)
    oof["blend"] = W_LGBM * oof["lgbm"] + (1 - W_LGBM) * oof["logit"]
    return oof


def report(name, oof, y, folds):
    yv = y.to_numpy()
    for m, p in oof.items():
        accs = [((p[folds == k] >= 0.5) == yv[folds == k]).mean() for k in range(10)]
        print(f"  {name:<18}{m:<7} acc {np.mean(accs):.4f} ± {np.std(accs):.4f} "
              f"log-loss {log_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)):.4f}", flush=True)


def main():
    X_all, y = load_master_features("train")
    X_test_all, _ = load_master_features("test")
    cols = feature_sets(list(X_all.columns))[0]["team+sm-ext"]
    X, X_test = X_all[cols], X_test_all[cols]
    tr_map, te_map = read_teams(TRAIN_JSONL), read_teams(TEST_JSONL)
    teams = [tr_map[b] for b in X.index]
    teams_test = [te_map[b] for b in X_test.index]
    print(f"Feature: {X.shape[1]} + {len(SKILL_COLUMNS)} di bravura\n")

    if "--no-cv" not in sys.argv:
        for split, folds in (("fold del team", load_folds()["fold"].to_numpy()),
                             ("seconda divisione", alt_folds(X.index, y)["fold"].to_numpy())):
            oof = cv(X, y, teams, folds)
            report(split, oof, y, folds)
            if split.startswith("fold"):
                pd.DataFrame({ID_COL: X.index, TARGET_COL: y.values, **{f"p_{k}": v for k, v in oof.items()}}).to_csv(
                    OUTPUTS_DIR / "oof_16_skill.csv", index=False)

    # Riallenamento finale: feature di bravura del train con divisione interna, del test dal train intero
    F_fit, F_test = skill_features(teams, y.to_numpy(), teams_test)
    Xf, Xt = with_skill(X, F_fit), with_skill(X_test, F_test)
    p_lgbm = make_lgbm().fit(Xf, y).predict_proba(Xt)[:, 1]
    p_logit = make_logit().fit(Xf, y).predict_proba(Xt)[:, 1]
    p_blend = W_LGBM * p_lgbm + (1 - W_LGBM) * p_logit
    s1 = write_submission(p_blend, X_test.index, "submission_16_blend_skill.csv")
    s2 = write_submission(p_lgbm, X_test.index, "submission_16_lgbm5_skill.csv")
    pd.DataFrame({ID_COL: X_test.index, "lgbm5": p_lgbm, "logit": p_logit, "blend": p_blend}).to_csv(
        OUTPUTS_DIR / "test_probs_16.csv", index=False)
    print(f"\nSubmission scritte. Quota 'vince P1': blend {s1[TARGET_COL].mean():.1%}, lgbm {s2[TARGET_COL].mean():.1%}")
    print(f"Test: P1 con squadra già vista {np.mean(F_test[:, 1] > 0):.1%} | P2 con stima {np.mean(F_test[:, 3] > 0):.1%}")


if __name__ == "__main__":
    main()
