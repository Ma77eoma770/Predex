"""Strumenti di valutazione condivisi per gli esperimenti della persona D.

Ogni esperimento (baseline, LightGBM, XGBoost, CatBoost, ...) passa da qui, così:
- tutti usano ESATTAMENTE le stesse 10 fold di data/train_folds.csv;
- tutti calcolano le metriche nello stesso modo;
- ogni risultato finisce nello stesso file di log (outputs/results_D.csv);
- le predizioni out-of-fold vengono salvate per lo stacking finale del team.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.config import ID_COL, OUTPUTS_DIR, TARGET_COL
from src.cross_validation import get_folds, load_folds

RESULTS_CSV = OUTPUTS_DIR / "results_D.csv"


def check_alignment(X: pd.DataFrame, folds_df: pd.DataFrame) -> None:
    """Verifica che la riga i di X sia la stessa battaglia della riga i di train_folds.csv.

    get_folds() restituisce POSIZIONI di riga (0, 1, 2, ...), non battle_id.
    Se X fosse ordinato diversamente, ogni battaglia finirebbe nel fold sbagliato
    e i risultati sarebbero falsi senza nessun errore visibile. Meglio fermarsi subito.
    """
    same_order = np.array_equal(X.index.to_numpy(), folds_df[ID_COL].to_numpy())
    assert same_order, "X e train_folds.csv non hanno le battaglie nello stesso ordine!"


def run_cv(
    make_model: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    folds_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Dict:
    """Esegue la 10-fold cross-validation per un modello.

    make_model: una funzione SENZA argomenti che restituisce un modello NUOVO, non allenato.
        Serve una funzione (e non un modello già creato) perché ad ogni fold dobbiamo
        ripartire da zero: riusare lo stesso oggetto porterebbe "memoria" da un fold all'altro.
    """
    if folds_df is None:
        folds_df = load_folds()
    check_alignment(X, folds_df)

    oof_probs = np.zeros(len(X))       # probabilità "P1 vince" predette per ogni battaglia
    fold_accs, train_accs, fold_times = [], [], []

    for fold, train_idx, val_idx in get_folds(folds_df):
        # .iloc seleziona per POSIZIONE di riga, coerente con get_folds()
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = make_model()
        t0 = time.time()
        model.fit(X_tr, y_tr)          # il modello vede SOLO le 9 fold di training
        fold_times.append(time.time() - t0)

        # predict_proba restituisce [P(classe 0), P(classe 1)]: prendiamo la colonna 1
        probs = model.predict_proba(X_va)[:, 1]
        preds = (probs >= 0.5).astype(int)   # soglia 0.5: sopra il 50% diciamo "vince P1"

        oof_probs[val_idx] = probs
        fold_accs.append(accuracy_score(y_va, preds))
        # Accuracy sulle STESSE battaglie usate per allenare: serve solo come diagnosi.
        # Se è molto più alta di quella di validazione, il modello sta imparando a memoria.
        train_preds = (model.predict_proba(X_tr)[:, 1] >= 0.5).astype(int)
        train_accs.append(accuracy_score(y_tr, train_preds))
        if verbose:
            print(f"  fold {fold}: train = {train_accs[-1]:.4f} | validazione = {fold_accs[-1]:.4f}")

    # Log-loss: penalizza le predizioni sicure ma sbagliate. Più bassa = meglio.
    # Le probabilità vengono "tagliate" lontano da 0 e 1 per evitare log(0) = infinito.
    clipped = np.clip(oof_probs, 1e-6, 1 - 1e-6)
    result = {
        "name": name,
        "acc_mean": float(np.mean(fold_accs)),
        "acc_std": float(np.std(fold_accs)),
        "train_acc": float(np.mean(train_accs)),
        "logloss": float(log_loss(y, clipped)),
        "fit_sec_per_fold": float(np.mean(fold_times)),
        "n_features": X.shape[1],
        "oof_probs": oof_probs,
    }
    if verbose:
        print(
            f"==> {name}: accuracy {result['acc_mean']:.4f} ± {result['acc_std']:.4f} "
            f"(train {result['train_acc']:.4f}) | "
            f"log-loss {result['logloss']:.4f} | {result['fit_sec_per_fold']:.2f}s per fold"
        )
    return result


def save_oof(result: Dict, X: pd.DataFrame, y: pd.Series) -> Path:
    """Salva le predizioni out-of-fold nel formato del team (serviranno per lo stacking)."""
    path = OUTPUTS_DIR / f"oof_{result['name']}.csv"
    pd.DataFrame({
        ID_COL: X.index,
        TARGET_COL: y.values,
        "pred_prob": result["oof_probs"],
        "pred_class": (result["oof_probs"] >= 0.5).astype(int),
    }).to_csv(path, index=False)
    return path


def log_result(result: Dict, params: str = "", notes: str = "") -> None:
    """Aggiunge una riga al file dei risultati, così niente si perde tra un esperimento e l'altro."""
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": result["name"],
        "n_features": result["n_features"],
        "acc_mean": round(result["acc_mean"], 4),
        "acc_std": round(result["acc_std"], 4),
        "train_acc": round(result["train_acc"], 4),
        "logloss": round(result["logloss"], 4),
        "fit_sec_per_fold": round(result["fit_sec_per_fold"], 3),
        "params": params,
        "notes": notes,
    }])
    # Rileggiamo il file e lo riscriviamo intero: così se aggiungiamo nuove colonne
    # (come train_acc) le righe vecchie restano allineate, con la cella vuota.
    if RESULTS_CSV.exists():
        row = pd.concat([pd.read_csv(RESULTS_CSV), row], ignore_index=True)
    row.to_csv(RESULTS_CSV, index=False)
