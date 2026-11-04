"""PASSO 14 — Nuovo tuning di LightGBM sul feature set unito (persona D).

I parametri del Passo 10 erano stati cercati sulle feature v2 (324 colonne). Ora il set migliore
è "team+sm-ext": feature del team + nostre sm_, SENZA le colonne di viability Smogon
(dato esterno). Con 786 colonne i parametri ottimali possono cambiare (per esempio conviene
far vedere a ogni albero una quota più piccola di colonne), quindi rifacciamo la ricerca.

Stesso protocollo del Passo 10:
  - obiettivo: log-loss media sui 10 fold del team;
  - primo tentativo = parametri del Passo 10, così si parte da una base nota;
  - verifica finale sulla seconda divisione (seed 2026), che il tuning non vede.
Lo studio si salva in outputs/optuna_lgbm_merged.db: se lo interrompi, riprende da lì.

Esecuzione, dalla cartella Predex-team (venv attivo). Durata indicativa: 25-40 minuti.
    python src/models/tune_14_optuna_merged.py
"""

import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

from src.config import OUTPUTS_DIR, RANDOM_STATE
from src.cross_validation import get_folds, load_folds
from src.features import load_master_features
from src.models.evaluation import log_result, run_cv, save_oof
from src.models.train_13_merged_features import LGBM_OPTUNA_PARAMS, alt_folds, feature_sets

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
FEATURE_SET = "team+sm-ext"
STORAGE = f"sqlite:///{OUTPUTS_DIR / 'optuna_lgbm_merged.db'}"
STUDY_NAME = "lgbm_team_sm_ext_logloss"
TUNED_KEYS = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples",
              "subsample", "colsample_bytree", "reg_lambda")


def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 40),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        # Con 786 colonne una quota più bassa di colonne per albero può ridurre il rumore
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
    }


def make_objective(X, y, folds_df):
    y_np = y.to_numpy()

    def objective(trial):
        params = suggest_params(trial)
        oof, done = np.zeros(len(X)), []
        for fold, tr, va in get_folds(folds_df):
            model = LGBMClassifier(**params).fit(X.iloc[tr], y.iloc[tr])
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
            done.extend(va)
            trial.report(log_loss(y_np[done], np.clip(oof[done], 1e-6, 1 - 1e-6)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("accuracy", float(((oof >= 0.5) == y_np).mean()))
        return log_loss(y_np, np.clip(oof, 1e-6, 1 - 1e-6))

    return objective


def progress(study, trial):
    if trial.state.name == "COMPLETE":
        print(f"  tentativo {trial.number:>3}: log-loss {trial.value:.4f} | accuracy "
              f"{trial.user_attrs.get('accuracy', float('nan')):.4f} | migliore {study.best_value:.4f}", flush=True)
    elif trial.state.name == "PRUNED":
        print(f"  tentativo {trial.number:>3}: interrotto", flush=True)


def main():
    X_all, y = load_master_features("train")
    sets, _ = feature_sets(list(X_all.columns))
    X = X_all[sets[FEATURE_SET]]
    print(f"Feature set {FEATURE_SET}: {X.shape[1]} colonne")
    folds_df = load_folds()

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        storage=STORAGE, study_name=STUDY_NAME, load_if_exists=True)
    if len(study.trials) == 0:
        study.enqueue_trial({k: LGBM_OPTUNA_PARAMS[k] for k in TUNED_KEYS})
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    remaining = N_TRIALS - done
    print(f"Tentativi completati: {done} | da fare: {max(0, remaining)}\n")
    t0 = time.time()
    if remaining > 0:
        study.optimize(make_objective(X, y, folds_df), n_trials=remaining, callbacks=[progress])
    print(f"\nTuning in {(time.time() - t0) / 60:.1f} minuti")

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs")).sort_values("value")
    print("\nI 5 tentativi migliori:")
    print(df.head(5).round(4).to_string(index=False))
    print("\nImportanza degli iperparametri:")
    for name, imp in optuna.importance.get_param_importances(study).items():
        print(f"  {name:<20} {imp:.2f}")

    best = {**LGBM_OPTUNA_PARAMS, **study.best_params}
    print(f"\nParametri migliori: {study.best_params}\n")
    print(f"{'divisione':<20}{'parametri':<12}{'accuracy':>18}{'log-loss':>10}{'distanza':>10}")
    for split_name, fd in (("fold del team", folds_df), ("seconda divisione", alt_folds(X.index, y))):
        for pname, params in (("passo 10", LGBM_OPTUNA_PARAMS), ("nuovi", best)):
            res = run_cv(lambda p=params: LGBMClassifier(**p), X, y,
                         name=f"14_lgbm_{pname.replace(' ', '')}_{'team' if fd is folds_df else 'alt'}",
                         folds_df=fd, verbose=False)
            if fd is folds_df and pname == "nuovi":
                save_oof(res, X, y)
                log_result(res, params=str(study.best_params), notes=f"Optuna su {FEATURE_SET}")
            print(f"{split_name:<20}{pname:<12}{res['acc_mean']:.4f} ± {res['acc_std']:.4f}"
                  f"{res['logloss']:>10.4f}{res['train_acc'] - res['acc_mean']:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
