# Predex — Pokémon Battle Outcome Prediction (Gen 2 OU)

Progetto di Machine Learning per la predizione dell'esito di battaglie Pokémon competitive (formato **Gen 2 OU**) a partire dalle informazioni parziali osservate nei primi **30 turni**.

---

## 📌 Panoramica del Problema

- **Task**: Classificazione binaria supervisionata. Predire `player_won` ($1$ se vince Player 1, $0$ se vince Player 2).
- **Dataset**: $10.000$ battaglie di training, $5.000$ battaglie di test. Classi perfettamente bilanciate ($50\% / 50\%$).
- **Metrica di valutazione**: **Accuracy** calcolata su 10-Fold Stratified Cross-Validation congelata (`data/train_folds.csv`).
- **Vincoli**:
  - Solo **ML classico** (ammessi modelli lineari, SVM, k-NN, alberi decisionali, Random Forest, Gradient Boosting — LightGBM, XGBoost, CatBoost — e Stacking/Voting; vietate reti neurali/deep learning).
  - Solo dati ufficiali forniti dalla competizione e `type_chart_gen2.csv` (vietati database esterni o statistiche esterne).
  - Massimo 3 submission finali generate tramite notebook Kaggle collegato al repository.

---

## 📂 Architettura Modulare del Repository

Il repository è strutturato per permettere a **5 persone di collaborare in parallelo** su feature e modelli diversi senza merge conflict:

```text
Predex/
├── data/
│   ├── train_folds.csv                 # 10 Fold congelate e condivise (NON modificare)
│   └── features/                       # Master Parquet Store generato localmente (ignorato da Git)
│       ├── .gitkeep
│       ├── train_features.parquet      # Tutte le feature estratte per il train (10.000 x N)
│       └── test_features.parquet       # Tutte le feature estratte per il test (5.000 x N)
├── raw_data/                           # File grezzi originali (ignorati da Git)
│   ├── .gitkeep
│   ├── train.jsonl
│   ├── test.jsonl
│   ├── type_chart_gen2.csv
│   └── sample_submission.csv
├── outputs/                            # Predizioni OOF e submission (ignorati da Git)
│   ├── oof_logistic_master.csv
│   └── submission_logistic_master.csv
├── src/
│   ├── config.py                       # Costanti, seed=42 e percorsi Pathlib
│   ├── cross_validation.py             # Generatore delle 10 fold stratificate
│   ├── features/                       # ARCHITETTURA DEL FEATURE STORE
│   │   ├── __init__.py
│   │   ├── registry.py                 # Registry centrale: estrae, compila e carica i Parquet
│   │   ├── selectors.py                # Pipeline di Feature Selection anti-rumore e anti-leakage
│   │   └── modules/                    # Moduli indipendenti di feature (uno per persona/idea)
│   │       ├── __init__.py
│   │       ├── f_baseline.py           # Feature base (prefisso bs_): team stats, lead stats, t30 snapshot
│   │       └── f_timeline_state.py     # Feature avanzate (prefisso tl_): KO unici, HP residui, status, boost
│   └── models/                         # Script di addestramento dei vari modelli
│       ├── __init__.py
│       └── train_logistic.py           # Baseline Logistic Regression con Feature Selection automatica
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Rapido per il Team

### 1. Clonare il repository
```bash
git clone https://github.com/Ma77eoma770/Predex.git
cd Predex
```

### 2. Creare l'ambiente virtuale ed installare le dipendenze

**Opzione A (Consigliata — con `uv`, istantaneo):**
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Opzione B (Classica — con `pip` e `venv` standard):**
```bash
python3 -m venv .venv
source .venv/bin/activate       # Su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Aggiungere i dati grezzi in `raw_data/`
I dati grezzi non sono inclusi su Git per policy e best practice. Scarica i file della gara e posizionali in `raw_data/`:
- `raw_data/train.jsonl`
- `raw_data/test.jsonl`
- `raw_data/type_chart_gen2.csv`
- `raw_data/sample_submission.csv`

---

## 🧠 Il Sistema "Master Feature Store": Come Funziona

Invece di avere file sparsi o un unico script gigante che causa merge conflict:

1. **Unico "Calderone" (Master Parquet)**:
   Tutte le feature pensate dal team confluiscono nei file binari super-veloci `data/features/train_features.parquet` e `data/features/test_features.parquet` (caricamento in **~0.05 secondi**).
2. **Auto-Discovery dei Moduli**:
   Tutti i file in `src/features/modules/` che iniziano con `f_*.py` vengono rilevati automaticamente da `registry.py`.
3. **Isolamento per Modello (Feature Selection Automatica)**:
   Poiché ogni modello gradisce feature diverse (es. k-NN odia l'alta dimensionalità, i modelli lineari soffrono la multicollinearità, mentre Gradient Boosting gestisce bene tutto), ogni script in `src/models/` applica la **propria selezione/filtro** all'interno della cross-validation senza toccare i file degli altri.

---

## 🛠️ Guida per il Team: Come Lavorare in Parallelo

### A. Come Aggiungere una Nuova Feature (in 3 Step)

Se a qualsiasi membro del gruppo viene in mente una nuova feature:

#### 1. Crea il tuo file in `src/features/modules/`
Crea ad esempio `src/features/modules/f_status_effects.py` con questa struttura:

```python
from typing import Dict, Any

# 1. Definisci un prefisso univoco di 2-3 lettere per non sovrascrivere le colonne altrui
PREFIX = "st_"

def extract_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Riceve il dizionario JSON di una singola battaglia e restituisce un dizionario di feature."""
    feat: Dict[str, float] = {}
    
    # Esempio: calcola quante volte compare lo status 'slp' o 'frz'
    timeline = data.get("battle_timeline", [])
    slp_count = sum(1 for t in timeline if t.get("p1_pokemon_state", {}).get("status") == "slp")
    
    feat[f"{PREFIX}p1_sleep_turns"] = float(slp_count)
    return feat
```

#### 2. Rigenera il Master Parquet Store
Esegui dal terminale:
```bash
python -m src.features.registry --force
```
Lo script troverà in automatico il tuo file, calcolerà le feature per tutte le 10.000 battaglie in pochi secondi e aggiornerà `data/features/train_features.parquet` e `data/features/test_features.parquet`.

#### 3. Testa la Feature nel Tuo Modello
Nessun altro modello viene toccato finché gli altri membri non decidono di usare le tue feature!

---

### B. Come Allenare un Modello e Filtrare le Feature

Negli script dentro `src/models/`:

```python
from src.features import load_master_features, create_linear_preprocessing_pipeline
from src.cross_validation import load_folds, get_folds

# 1. Carica istantaneamente il Master Parquet (tutte le feature disponibili)
X, y = load_master_features("train")

# 2. (Opzionale) Seleziona solo determinati prefissi
# from src.features import filter_by_prefix
# X = filter_by_prefix(X, include_prefixes=["tl_", "st_"])

# 3. Applica la Pipeline di Feature Selection (rimuove varianza zero e multicollinearità)
pipeline = Pipeline([
    ("preprocessor", create_linear_preprocessing_pipeline(
        drop_collinear=True,       # Rimuove feature duplicate o con correlazione > 0.95
        collinear_threshold=0.95,
        k_best=25,                 # (Opzionale) Seleziona automaticamente le 25 migliori per il target
    )),
    ("classifier", LogisticRegression(C=1.0))
])
```

---

## 🛡️ Regole d'Oro per la Collaborazione

1. **Non modificare `data/train_folds.csv`**:
   Le 10 fold sono fissate e garantiscono che ogni esperimento sia confrontabile al millesimo.
2. **Lavorare su branch Git dedicati**:
   Non lavorare direttamente su `main`. Crea un branch per ogni nuova feature o modello (es. `git checkout -b feat/status-effects` o `model/catboost`).
3. **Usa sempre i prefissi nelle feature**:
   Ogni modulo deve usare il proprio prefisso (es. `bs_` per baseline, `tl_` per timeline, `st_` per status, `tm_` per team stats). Mai creare colonne senza prefisso.
4. **Zero Data Leakage**:
   Ogni normalizzazione (`StandardScaler`) o selezione feature supervisionata (`SelectKBest`) deve girare **all'interno del loop dei fold** (tramite `Pipeline`), mai globalmente prima dello split.
5. **Salva sempre le predizioni Out-Of-Fold (OOF)**:
   Ogni modello testato deve salvare `outputs/oof_<nome_modello>.csv` con le probabilità predette sul validation fold per tutte le 10.000 battaglie. Questi file serviranno per costruire lo **Stacking / Blending finale** che unirà i modelli di tutti i membri per la submission definitiva.

---

## 🚀 Eseguire la Baseline Esistente

Per verificare l'ambiente ed eseguire la Logistic Regression sul Master Feature Store:
```bash
python src/models/train_logistic.py
```
Score di riferimento attuale: **CV Accuracy ~66.56%** (contro il 50.00% del random guess).
