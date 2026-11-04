"""PASSO 15 — Modello finale sul feature set unito: LightGBM 5 seed, logistica e media (persona D).

Feature set: "team+sm-ext" (786 colonne: feature del team + nostre sm_, senza dati Smogon).
Modelli:
  - LightGBM con i parametri del Passo 14, media di 5 seed;
  - regressione logistica del team (filtro collinearità + standardizzazione, C=1) sullo stesso set;
  - media pesata 0,75 LightGBM + 0,25 logistica. Il peso è fissato PRIMA di guardare i risultati,
    dai pesi dello stacking del Passo 8 (logistica circa 1/4 del totale): così non lo
    "tariamo" sui fold e la stima resta onesta.
Tutto valutato sui fold del team e sulla seconda divisione (seed 2026).

Poi riallena su tutte le 10.000 battaglie e scrive tre submission in outputs/:
  submission_15_lgbm5_merged.csv     -> candidata n. 1 (miglior modello singolo)
  submission_15_blend_merged.csv     -> candidata n. 2 (ensemble)
  submission_15_logit_merged.csv     -> candidata n. 3 (opzione robusta, lineare)

Esecuzione dalla cartella Predex-team (venv attivo), circa 10-15 minuti:
    python src/models/train_15_final.py
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from src.config import ID_COL, OUTPUTS_DIR, RANDOM_STATE, SAMPLE_SUBMISSION_CSV, TARGET_COL
from src.cross_validation import get_folds, load_folds
from src.features import create_linear_preprocessing_pipeline, load_master_features
from src.models.evaluation import log_result, run_cv, save_oof
from src.models.train_13_merged_features import alt_folds, feature_sets

warnings.filterwarnings("ignore")

FEATURE_SET = "team+sm-ext"
SEEDS = [42, 1, 2, 3, 4]
W_LGBM = 0.75   # peso fissato a priori (vedi docstring)

# Miglior tentativo del Passo 14 (n. 33), copiato dall'output di Optuna
LGBM_PARAMS = {
    "n_estimators": 700, "learning_rate": 0.015734907157116045, "num_leaves": 22,
    "min_child_samples": 28, "subsample": 0.8530449054047685, "subsample_freq": 1,
    "colsample_bytree": 0.7597032332745631, "reg_lambda": 0.002619227712179358,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
}


class SeedAveragedLGBM:
    """Stessi iperparametri, un modello per seed; previsione = media delle probabilità."""

    def __init__(self, params, seeds):
        self.params, self.seeds = params, seeds

    def fit(self, X, y):
        self.models_ = [LGBMClassifier(**{**self.params, "random_state": s}).fit(X, y) for s in self.seeds]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def make_lgbm():
    return SeedAveragedLGBM(LGBM_PARAMS, SEEDS)


def make_logit():
    return Pipeline([
        ("preprocessor", create_linear_preprocessing_pipeline(drop_collinear=True, collinear_threshold=0.95, k_best=None)),
        ("classifier", LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def summarize(probs, y, folds_df):
    y_np = y.to_numpy()
    accs = [((probs[va] >= 0.5) == y_np[va]).mean() for _, _, va in get_folds(folds_df)]
    return np.mean(accs), np.std(accs), log_loss(y_np, np.clip(probs, 1e-6, 1 - 1e-6))


def write_submission(probs, test_ids, filename):
    sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    pred = pd.Series((probs >= 0.5).astype(int), index=test_ids)
    sub = sample[[ID_COL]].copy()
    sub[TARGET_COL] = sub[ID_COL].map(pred)
    assert not sub[TARGET_COL].isna().any(), "battle_id senza previsione"
    sub[TARGET_COL] = sub[TARGET_COL].astype(int)
    sub.to_csv(OUTPUTS_DIR / filename, index=False)
    return sub


def main():
    X_all, y = load_master_features("train")
    X_test_all, _ = load_master_features("test")
    cols = feature_sets(list(X_all.columns))[0][FEATURE_SET]
    X, X_test = X_all[cols], X_test_all[cols]          # stesso ordine di colonne per train e test
    print(f"Feature set {FEATURE_SET}: {X.shape[1]} colonne\n")

    print(f"{'divisione':<20}{'modello':<22}{'accuracy':>18}{'log-loss':>10}")
    for split, fd in (("fold del team", load_folds()), ("seconda divisione", alt_folds(X.index, y))):
        tag = "team" if split.startswith("fold") else "alt"
        r_lgbm = run_cv(make_lgbm, X, y, name=f"15_lgbm5_{tag}", folds_df=fd, verbose=False)
        r_logit = run_cv(make_logit, X, y, name=f"15_logit_{tag}", folds_df=fd, verbose=False)
        blend = W_LGBM * r_lgbm["oof_probs"] + (1 - W_LGBM) * r_logit["oof_probs"]
        corr = np.corrcoef(r_lgbm["oof_probs"], r_logit["oof_probs"])[0, 1]
        for label, probs in (("LightGBM 5 seed", r_lgbm["oof_probs"]), ("logistica", r_logit["oof_probs"]),
                             (f"media {W_LGBM:.2f}/{1 - W_LGBM:.2f}", blend)):
            m, s, ll = summarize(probs, y, fd)
            print(f"{split:<20}{label:<22}{m:.4f} ± {s:.4f}{ll:>10.4f}", flush=True)
        print(f"{'':<20}correlazione LightGBM-logistica: {corr:.3f}")
        if tag == "team":
            for res in (r_lgbm, r_logit):
                save_oof(res, X, y)
                log_result(res, params=f"feature set {FEATURE_SET}")
            pd.DataFrame({ID_COL: X.index, TARGET_COL: y.values, "pred_prob": blend,
                          "pred_class": (blend >= 0.5).astype(int)}).to_csv(OUTPUTS_DIR / "oof_15_blend.csv", index=False)

    # --- Riallenamento su tutte le 10.000 battaglie e submission ---
    p_lgbm = make_lgbm().fit(X, y).predict_proba(X_test)[:, 1]
    p_logit = make_logit().fit(X, y).predict_proba(X_test)[:, 1]
    p_blend = W_LGBM * p_lgbm + (1 - W_LGBM) * p_logit
    subs = {
        "submission_15_lgbm5_merged.csv": write_submission(p_lgbm, X_test.index, "submission_15_lgbm5_merged.csv"),
        "submission_15_blend_merged.csv": write_submission(p_blend, X_test.index, "submission_15_blend_merged.csv"),
        "submission_15_logit_merged.csv": write_submission(p_logit, X_test.index, "submission_15_logit_merged.csv"),
    }
    pd.DataFrame({ID_COL: X_test.index, "lgbm5": p_lgbm, "logit": p_logit, "blend": p_blend}).to_csv(
        OUTPUTS_DIR / "test_probs_15.csv", index=False)

    print("\nControlli sulle submission:")
    prev_path = PROJECT_ROOT.parent / "Predex-main" / "outputs" / "submission_12_lgbm_optuna_5seeds.csv"
    prev = pd.read_csv(prev_path) if prev_path.exists() else None
    for name, sub in subs.items():
        line = f"  {name:<34} 'vince P1' {sub[TARGET_COL].mean():.1%}"
        if prev is not None:
            line += f" | uguali alla 12 (0,690 sul pubblico): {(sub[TARGET_COL] == prev[TARGET_COL]).mean():.1%}"
        print(line)
    q = [10, 50, 90]
    print(f"  Percentili prob. LightGBM: test {np.percentile(p_lgbm, q).round(3)}")


if __name__ == "__main__":
    main()
