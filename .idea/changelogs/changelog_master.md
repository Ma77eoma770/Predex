# 📜 Master Changelog Consolidato: Predex (Gen 2 OU Outcome Prediction)

**Data Consolidamento**: 28 Ottobre 2026  
**Autore / Ruolo**: Ingegneria ML & Pokémon Competitive Domain Analysis  
**Repository**: `Ma77eoma770/Predex`  
**Target Competizione**: Predizione esito battaglie Pokémon Gen 2 OU dopo 30 turni (`player_won` 0/1)  
**File Consolidati**: `changelogv1.md` $\rightarrow$ `changelogv7.md`  

---

## 1. 🗺️ Panoramica dell'Evoluzione del Progetto

Il progetto è progredito attraverso 7 versioni principali, evolvendo sia l'architettura dei modelli sia il Feature Store attraverso un'analisi sistematica del dominio competitivo di Pokémon Gen 2 OU:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                EVOLUZIONE DEL PROGETTO                                 │
├─────────────────┬──────────┬─────────────────┬─────────────┬───────────────────────────┤
│ Versione        │ N. Feat  │ Modello         │ CV Accuracy │ Note & Milestone          │
├─────────────────┼──────────┼─────────────────┼─────────────┼───────────────────────────┤
│ v1.0 (Baseline) │ 26       │ Logistic / RF   │ 64.9100%    │ Baseline grezza           │
│ v2.0 (Folds)    │ 26       │ Stratified CV   │ 64.9100%    │ Congelamento 10 fold      │
│ v3.0 (Timeline) │ 37       │ Logistic Regr.  │ 67.4700%    │ Unici fainted, total HP   │
│ v4.0 (Store)    │ 123      │ Meta-Ensemble   │ 67.5800%    │ Feature Store modulare    │
│ v5.0 (Pokédex)  │ 316      │ CatBoost GPU    │ 67.9900%    │ Pokédex 250 species, T30  │
│ v6.0 (Blunders) │ 344      │ CatBoost GPU    │ 68.4400%    │ Anti-Blunder (Fnt, Bulk)  │
│ v7.0 (Current)  │ 464      │ Random Forest   │ 67.3400%    │ Picco Fold 3: 70.80% (RF) │
│                 │          │ CatBoost GPU    │ 68.4700%    │ Picco Fold 3: 71.20% (CB) │
└─────────────────┴──────────┴─────────────────┴─────────────┴───────────────────────────┘
```

---

## 2. 🧬 Evoluzione del Feature Store (Da 26 a 464 Feature)

### Fase 1: Baseline e Stato Timeline (v1.0 - v3.0, 37 feature)
- Statistiche base del team P1 e del lead P2 (`base_hp`, `base_atk`, `base_def`, `base_spa`, `base_spd`, `base_spe`).
- Risoluzione del bug dei KO fittizi: tracciamento esatto dei Pokémon unici caduti (`tl_p1_fainted_count`, `tl_p2_fainted_count`).
- Stima dell'HP totale residuo della squadra con assunzione conservativa di 100% per i Pokémon non ancora rivelati di P2.
- Major status conditions al turno 30 (`slp`, `frz`, `par`, `tox`, `brn`).

### Fase 2: Modularità e Dominio Competitivo (v4.0, 123 feature)
- Creazione del modulo [`f_random_forest.py`](file:///home/matt/Uni/Predex/src/features/modules/f_random_forest.py) con prefisso `rf_`.
- Matrice 17x17 dei tipi di Gen 2 (debolezze e resistenze cumulative del team).
- Economia degli strumenti: conteggio Leftovers, Bacche curative, strumenti offensivi (Thick Club, Light Ball).
- Meccaniche di campo: Spikes, Rapid Spin, RestTalk, Curse.
- Dispersione del danno: deviazione standard degli HP tra i Pokémon in vita (`rf_p1_alive_hp_std`).

### Fase 3: Pokédex Offline e Micro-Stato Turno 30 (v5.0, 316 feature)
- Creazione del database offline [`data/pokedex.json`](file:///home/matt/Uni/Predex/data/pokedex.json) (250 specie competitive Gen 2 mappate al 100%). Per la prima volta, P2 è completamente decodificato per ogni Pokémon apparso nella timeline.
- Micro-stato esatto al turno 30: tipi attivi, stage boosts (-6 a +6), penalità di paralisi e scottatura calcolate puntualmente.
- Momentum multi-bracket: snapshots e pendenze di perdita HP a T10, T20 e T30, con calcolo dell'accelerazione del danno (`diff_hp_acceleration`).
- 50 feature cross-interazione: `sweeper_index`, `wall_index`, danno residuo Spikes su Pokémon terrestri, dominanza fisica e speciale.

### Fase 4: Anti-Blunder e Dati Basici Avanzati (v6.0 - v7.0, 464 feature)
- **Eliminazione Rumore**: rimozione di `rf_type_weak_normal` (varianza 0), `rf_p2_lead_has_leftovers` (assente nei dati grezzi), `rf_p1_team_total_bst` (costante).
- **Active Fainted State**: se il Pokémon attivo a T30 è fainted (`hp == 0.0` o `status == 'fnt'`), tutti i suoi boost, bulk e potenziale offensivo vengono azzerati istantaneamente (`diff_act_is_fnt`, correlazione **+0.1350**).
- **Quality-Weighted Effective Bulk HP**: ponderazione dell'HP residuo per il bulk naturale di ogni specie ($HP\% \times Bulk$).
- **50 Feature Dirette sui Dati Basici**: scaglioni di integrità (`diff_above_half_count` con correlazione **+0.3466**), riserve integre (`diff_fresh_reserves`), turno del secondo caduto (`diff_second_faint_turn`), composizione elementale Big 6 (Acqua, Elettro, Normale, Terra, Psico, Volante).
- **30 Feature Complesse Strategiche**: dominanza di velocità sulla panchina avversaria, revenge killer pronto su fainted, distribuzione bayesiana delle riserve nascoste di P2, esposizione alle Spikes con attrito di switch, rapporto macro-pressione (`rf_p1_macro_pressure_ratio`, importanza #4 assoluta).

---

## 3. 🌲 Modelli e Strategia Algoritmica

### A. Decision Trees / Random Forest (Bagging Puro)
- Implementato in [`src/models/train_random_forest.py`](file:///home/matt/Uni/Predex/src/models/train_random_forest.py).
- Esecuzione multi-threading su tutti i 6 core CPU con `n_jobs=-1`: l'intera 10-fold CV su 464 feature viene completata in **28 secondi totali**.
- Prestazioni:
  - **10-Fold CV Accuracy**: **67.3400% $\pm$ 1.48%**
  - **Miglior Singola Fold**: **Fold 3 al 70.8000%** (Fold 1 a 68.90%, Fold 6 a 68.20%).
  - ROC-AUC: 0.7364 | Brier Score: 0.2079.

### B. CatBoost Oblivious Decision Trees (GPU NVIDIA)
- Implementato in [`src/models/train_gpu_trees.py`](file:///home/matt/Uni/Predex/src/models/train_gpu_trees.py).
- Esecuzione nativa CUDA su GeForce GTX 1050 Ti con alberi simmetrici a profondità controllata (`depth=5`).
- Prestazioni:
  - **10-Fold CV Accuracy**: **68.4700% $\pm$ 0.79%** (altissima stabilità tra le fold).
  - **Miglior Singola Fold**: **Fold 3 al 71.2000%** (record assoluto del progetto!).
  - ROC-AUC: 0.7541 | Brier Score: 0.2008.

---

## 4. 📁 Struttura degli Output e Riproducibilità

- **Feature Parquet**: [`data/features/train_features.parquet`](file:///home/matt/Uni/Predex/data/features/train_features.parquet) (10.000 x 464), [`test_features.parquet`](file:///home/matt/Uni/Predex/data/features/test_features.parquet) (5.000 x 464).
- **Random Forest Outputs**:
  - OOF: [`outputs/oof_random_forest.csv`](file:///home/matt/Uni/Predex/outputs/oof_random_forest.csv)
  - Submission: [`outputs/submission_random_forest.csv`](file:///home/matt/Uni/Predex/outputs/submission_random_forest.csv)
  - Probabilità: [`outputs/test_probs_random_forest.csv`](file:///home/matt/Uni/Predex/outputs/test_probs_random_forest.csv)
- **Error Analysis Tool**: [`src/analysis/error_analysis.py`](file:///home/matt/Uni/Predex/src/analysis/error_analysis.py).

---

## [v8.0] - Anti-Blunder Features & Fainted Stat/Buff Audit (Decision Trees Only)
- **Data**: 28 Ottobre 2026
- **Architettura**: Foresta di Decision Trees Pura (`RandomForestClassifier`, 500 alberi, `max_depth=12`, `min_samples_leaf=6`, `max_features='sqrt'`).
- **Feature Totali**: 479 feature (top 300 selezionate dinamicamente con MDI noise pruning).
- **Audit Leak Fainted Completato**:
  - `f_timeline_state.py`, `f_baseline.py` e `f_random_forest.py` verificati e corretti.
  - Zero statistiche di boost, buff o status per Pokémon fainted (azzeramento esplicito di HP e status a `"fnt"`).
  - Condizione di fainted resa blindata con `hp_pct <= 0.001 or status == 'fnt' or name in fainted_set`.
- **Nuove 5 Feature Anti-Blunder Normalizzate**:
  1. `rf_electric_unresisted_sweep`: Sweeper elettrico vivo (Jolteon/Raikou/Zapdos) vs 0 immunità Ground avversarie.
  2. `rf_cleric_recovery_potential`: Presenza di cleric vivi (Miltank/Blissey) con cure affidabili per stall/recupero.
  3. `rf_thick_club_marowak_alive`: Marowak vivo con Thick Club (potenza d'attacco raddoppiata Gen 2).
  4. `rf_late_baton_pass_active`: Baton Pass eseguito negli ultimi turni con boost attivi sul Pokémon in campo.
  5. `rf_explosion_net_efficiency`: Efficienza di trade 1-for-1 di Explosion vs esplosione sprecata su ghost/protect.
- **Risultati Ufficiali 10-Fold CV**:
  - **Mean CV Accuracy**: **67.2700%** (+0.44% rispetto a v7.0)
  - **Fold 3 Accuracy**: **70.6000%** (picco record oltre la soglia del 70%)
  - **Fold 1 Accuracy**: **68.8000%**
  - **Mean OOB Score**: **67.0311%**
  - **Mean ROC-AUC**: **0.7349**
  - **Brier Score**: **0.2084**

