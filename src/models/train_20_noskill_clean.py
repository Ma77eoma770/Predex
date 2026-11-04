"""PASSO 20 — "Assicurazione" senza bravura dei giocatori, sul feature store pulito (persona D).

Come il Passo 19 (786 + 33 feature di Max in LightGBM 5 seed, media a tre con la nostra logistica
e quella di Max, pesi 0,70/0,15/0,15) ma SENZA le 5 feature di bravura. Serve come terza scelta
finale: se nel test privato le squadre si ripetessero meno che nel train, la bravura aiuterebbe
meno e questo modello reggerebbe meglio. Sostituisce la submission 12, costruita su un feature
store che conteneva ancora dati esterni (classifica Smogon, 3 specie aggiunte a pokedex.json).

    python src/models/train_20_noskill_clean.py 2>&1 | tee outputs/log_20.txt
Scrive outputs/submission_20_blend3_noskill_clean.csv
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

import src.models.train_19_final_max as t19
from src.config import ID_COL, OUTPUTS_DIR, TARGET_COL, TEST_JSONL, TRAIN_JSONL
from src.cross_validation import load_folds
from src.features import load_master_features
from src.models.train_13_merged_features import alt_folds, feature_sets
from src.models.train_15_final import make_lgbm, make_logit, write_submission

warnings.filterwarnings("ignore")


def fit_predict(Xg_tr, Xg_va, Fl_tr, Fl_va, y_tr, fl_cols):
    return {
        "lgbm": make_lgbm().fit(Xg_tr, y_tr).predict_proba(Xg_va)[:, 1],
        "logit": make_logit().fit(Xg_tr.drop(columns=fl_cols), y_tr).predict_proba(Xg_va.drop(columns=fl_cols))[:, 1],
        "logit_fl": t19.max_logit().fit(Fl_tr, y_tr).predict_proba(Fl_va)[:, 1],
    }


def main():
    X_all, y = load_master_features("train")
    X_test_all, _ = load_master_features("test")
    cols = feature_sets(list(X_all.columns))[0]["team+sm-ext"]
    X, X_test = X_all[cols], X_test_all[cols]
    Fl, groups = t19.max_frame(TRAIN_JSONL, X.index)
    Fl_test, _ = t19.max_frame(TEST_JSONL, X_test.index)
    fl_cols = [c for g in t19.MAX_LGBM for c in groups[g]]
    Xg, Xg_test = pd.concat([X, Fl[fl_cols]], axis=1), pd.concat([X_test, Fl_test[fl_cols]], axis=1)
    yv = y.to_numpy()
    print(f"LightGBM: {X.shape[1]} + {len(fl_cols)} di Max (senza bravura)\n", flush=True)

    if "--no-cv" not in sys.argv:
        for split, folds in (("fold del team", load_folds()["fold"].to_numpy()),
                             ("seconda divisione", alt_folds(X.index, y)["fold"].to_numpy())):
            oof = {m: np.zeros(len(X)) for m in t19.WEIGHTS}
            for k in range(10):
                tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
                p = fit_predict(Xg.iloc[tr], Xg.iloc[va], Fl.iloc[tr], Fl.iloc[va], yv[tr], fl_cols)
                for m in oof:
                    oof[m][va] = p[m]
            oof["MEDIA A TRE"] = t19.blend(oof)
            for m, pr in oof.items():
                accs = [((pr[folds == k] >= 0.5) == yv[folds == k]).mean() for k in range(10)]
                print(f"  {split:<18}{m:<14} acc {np.mean(accs):.4f} ± {np.std(accs):.4f} "
                      f"log-loss {log_loss(yv, np.clip(pr, 1e-6, 1 - 1e-6)):.4f}", flush=True)

    p = fit_predict(Xg, Xg_test, Fl, Fl_test, yv, fl_cols)
    s = write_submission(t19.blend(p), X_test.index, "submission_20_blend3_noskill_clean.csv")
    print(f"\nSubmission scritta. Quota 'vince P1': {s[TARGET_COL].mean():.1%}")


if __name__ == "__main__":
    main()
