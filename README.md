# Predex — Pokémon Battle Outcome Prediction (Gen 2 OU)

Progetto di Machine Learning per la predizione dell'esito di battaglie Pokémon (formato **Gen 2 OU**) a partire dalle informazioni asimmetriche e parziali osservate nei primi **30 turni**.

---

## 📌 Panoramica del Problema

- **Task**: Classificazione binaria supervisionata. Predire `player_won` ($1$ se vince Player 1, $0$ se vince Player 2).
- **Dataset**: $10.000$ battaglie di training, $5.000$ battaglie di test. Classi perfettamente bilanciate ($50\% / 50\%$).
- **Metrica di valutazione**: **Accuracy** (valutata sul leaderboard privato finale).
- **Condizioni di gara**:
  - Solo **ML classico** (ammessi lineari, SVM, kNN, alberi, Random Forest, Gradient Boosting, Stacking/Voting; vietate reti neurali/MLP).
  - Solo dati ufficiali forniti dalla competizione e `type_chart_gen2.csv` (vietati database esterni o statistiche di usage esterne).
  - Massimo 3 submission finali generate tramite notebook Kaggle collegato al repository.

---

## 📂 Struttura del Progetto

```text
Predex/
├── data/
│   └── train_folds.csv          # 10-Fold Stratified CV condivise e tracciate in Git (NON cancellare/modificare)
├── raw_data/                    # Dati grezzi originali (ignorati da Git, vedi setup sotto)
│   ├── .gitkeep
│   ├── train.jsonl
│   ├── test.jsonl
│   ├── type_chart_gen2.csv
│   └── sample_submission.csv
├── outputs/                     # Predizioni Out-Of-Fold (OOF), modelli salvati e submission (ignorati da Git)
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py                # Configurazione globale, costanti (seed=42) e percorsi con pathlib
│   └── cross_validation.py      # Gestione delle 10 fold stratificate e generatori per train/val
├── requirements.txt             # Dipendenze del progetto
└── README.md                    # Documentazione del repository
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
I dati grezzi pesanti non sono inclusi su Git per policy e best practice. Scarica i file della gara e posizionali in `raw_data/`:
- `raw_data/train.jsonl`
- `raw_data/test.jsonl`
- `raw_data/type_chart_gen2.csv`
- `raw_data/sample_submission.csv`

### 4. Verifica di Sanity Check
Esegui lo script di validazione per assicurarti che l'ambiente e le fold siano caricate correttamente:
```bash
python src/cross_validation.py
```
Dovresti visualizzare il riepilogo delle **10 Fold** con $9.000$ campioni di train e $1.000$ di validation ciascuna, bilanciate esattamente al $50.00\%$.

---

## 🛡️ Regole d'Oro per la Collaborazione

Per consentire a tutti di lavorare in parallelo su feature e modelli diversi senza merge conflict:

1. **Non ricalcolare mai le fold**:
   Tutti usano sempre `data/train_folds.csv` caricandolo con `from src.cross_validation import load_folds, get_folds`. Le fold sono fissate e condivise per garantire un confronto equo al millesimo.
2. **Lavorare su branch Git dedicati**:
   Non lavorare mai direttamente su `main`. Crea un branch per ogni nuova feature o modello:
   ```bash
   git checkout -b feat/type-advantages
   # oppure
   git checkout -b model/catboost-baseline
   ```
3. **Modularizzare le Feature**:
   Evita di inserire tutto il codice in un unico script. Organizza le funzioni di feature engineering in file modulari (es. `src/features/team_stats.py`, `src/features/timeline_dynamics.py`) che prendono in input i dati grezzi e restituiscono matrici/DataFrame puliti.
4. **Validazione rigorosa (No Data Leakage)**:
   - Qualsiasi trasformazione (StandardScaler, Imputer, Target Encoding, PCA) va calcolata **esclusivamente sulla porzione di train del fold** e applicata sul fold di validazione.
   - Usa `Pipeline` di `scikit-learn` o il generatore `get_folds()`.
5. **Salvare le predizioni Out-Of-Fold (OOF)**:
   Ogni modello testato deve salvare le predizioni OOF per tutte le 10.000 battaglie in `outputs/` (es. `outputs/oof_lightgbm_v1.csv`). Questo servirà per costruire lo **Stacking/Voting finale** combinando i modelli migliori del team.
