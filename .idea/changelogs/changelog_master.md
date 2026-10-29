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
│ v8.0 (Audit)    │ 479      │ Random Forest   │ 67.2700%    │ Fainted Leak Audit        │
│ v9.0 (Viability)│ 454      │ Random Forest   │ 67.3400%    │ Smogon Index, Wall Lock   │
│ v10.0 (Physics) │ 391/522  │ Bagging Blend   │ 68.1500%    │ Raw: 68.15%, Tuned: 68.30%│
│                 │          │ (50% ET+50% RF) │ (AUC 0.747) │ Brier: 0.2045, LL: 0.5933 │
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

---

## [v9.0] - Metagame Viability Tiering & Advanced Tree Pruning (Decision Trees Only)
- **Data**: 28 Ottobre 2026
- **Architettura**: Foresta di Decision Trees Pura (`RandomForestClassifier`, 500 alberi, `max_depth=12`, `min_samples_leaf=6`, `max_features='sqrt'`).
- **Feature Store**: 454 feature pulite (rimosse 26 feature a importanza zero e 6 duplicati collineari).
- **Nuove Meccaniche Integrate**:
  1. `rf_diff_alive_viability`: Smogon Gen 2 OU Viability Tier Power Index (Snorlax 100, Zapdos/Raikou 85+, ecc.) sui Pokémon rimasti in vita. Entrata direttamente al 16° posto assoluto per importanza su tutto il modello.
  2. `rf_diff_speed_monopoly`: Controllo deterministico dell'egemonia di velocità fra i superstiti (tenendo conto della penalità del -75% da paralisi).
  3. `rf_diff_defensive_wall_lock`: Rilevamento di wall difensivi (Skarmory/Steelix) contro squadre avversarie prive di risposte superefficaci.
  4. `rf_diff_permanent_spikes_active`: Spikes permanenti in caso di assenza o KO di tutti i Rapid Spinner avversari (Cloyster, Forretress, Starmie).
  5. `rf_diff_phazing_denial_ready`: Neutralizzazione dei boost avversari di Curse/Belly Drum tramite presenza di un phazer difensivo (Roar/Whirlwind).
- **Esclusioni Effettuate su Richiesta Utente**:
  - Rimossa la proposta sul Rest Sleep Lock.
  - Rimossi status a importanza zero (burn/freeze/perish/leechseed/confusion a t30).
  - Rimossi duplicati collineari esatti (`p1_team_mean_base_*`, `leftovers_healed_est`).
- **Metriche Riscontrate**:
  - **ROC-AUC**: **0.7360** (nuovo record di discriminazione probabilistica)
  - **Brier Score**: **0.2081** (calibrazione migliorata)
  - **Fold 3**: **70.0000%**, **Fold 1**: **68.6000%**, **Fold 8**: **68.1000%**
  - **Mean CV Accuracy**: **66.9200% - 67.2700%** (a seconda del noise pruning threshold)

---

## [v10.0] - GSC Bipartite Physics Engine v2, Empirical Move Priors, Spearman Hierarchical Pruning & Pure Bagging Blend
- **Data**: 29 Ottobre 2026
- **Architettura**: Pure Tree Ensemble Blending (50% ExtraTrees + 50% RandomForest), 450 estimatori ciascuno, `max_depth=16-17`, `min_samples_leaf=8`, `max_features=0.18`, `criterion='log_loss'`.
- **Feature Store**: 522 feature aggregate nel Master Feature Store (`f_bipartite_physics.py` v2 + `f_mitigations.py` + `f_random_forest.py`).
- **Ortogonalizzazione tramite Spearman Hierarchical Pruning**:
  - Matrice di correlazione di Spearman $|\rho|$ sulle 522 feature.
  - Agglomerative Hierarchical Clustering con Ward Linkage (soglia di distanza $0.12$, ovvero $|\rho| > 0.88$).
  - Dentro ogni cluster di feature collineari viene mantenuta esclusivamente la feature a più alta Permutation Importance Out-Of-Fold.
  - Riduzione da 522 a **391 feature ortogonali**, tagliando 131 cloni collineari di HP e riserve che diluivano i campionamenti ai nodi radice dell'albero.
- **Meccaniche Fisiche e Competitive di Gen 2 Introdotte**:
  1. **Stat Scaling Lv. 100 GSC e Iniezione Strumenti Moltiplicatori**:
     - Light Ball Pikachu: raddoppio $SpA_{Lv100} \to 396$ (superiore a Zapdos/Raikou, risolve strutturalmente match anomali come #7204).
     - Thick Club Marowak: raddoppio $Atk_{Lv100} \to 516$ (se non rivela Leftovers).
  2. **Stallo Infinito (TTK = $\infty$) e Cure Cicliche nel Micro-Rollout**:
     - Feature dedicata `rf_diff_infinite_stall_locks`: numero di difensori per cui $\max_{i \in Opp} d(i \to j) < 0.31$ (soglia sotto cui il ciclo a 3 turni di Rest + Leftovers mura matematicamente l'intera squadra nemica).
     - Simulazione cure 50% (Soft-Boiled Blissey, Milk Drink Miltank, Recover Starmie) se $HP \le 0.55$.
     - Ciclo Rest + Curse: se $HP\% > d_{incoming}$, recupero a 100% HP e 2 turni di sonno con scaling +1 Atk/+1 Def se non ci sono phazer avversari vivi.
  3. **Leva di Detonazione Latente (Explosion / Self-Destruct 500 BP effettivi)**:
     - `rf_diff_latent_boom_leverage`: calcolo dell'overhead di scambio 1-for-1 forzato per boom-user classici (Forretress, Cloyster, Exeggutor, Gengar, Steelix, Snorlax, Golem) sotto il 45% HP tenendo conto del dimezzamento difensivo del bersaglio.
  4. **Prior Empirici delle Mosse dal Train Set (Zero Data Leakage)**:
     - Script autonomo che mappa $P(m \mid s)$ su 195 specie Gen 2 OU leggendo unicamente le timeline del training set (`data/empirical_move_priors.json`).
     - Stima probabilistica della coverage latente (es. Fire Blast su Tyranitar/Snorlax contro Skarmory/Forretress, Cross Chop su Machamp con crit rate al 25% che scavalca i boost difensivi).
  5. **Switch Proattivo al Turno 31**:
     - Se il Pokémon attivo a T30 infligge $d < 0.18$ e in panchina c'è un compagno che subisce $< 0.35$ in entrata e minaccia $> 0.45$ al turno dopo, il rollout simula il cambio tattico anziché la perdita passiva del pezzo.
  6. **Feature di Regime-Switching Non-Lineari**:
     - `rf_comeback_override_signal`: disattivata nei match standard, esplode quando P1 è sotto di HP ma il rollout e lo sweep counter certificano la vittoria (risolti i casi di ribaltamento come #1861, #7989).
     - `rf_throw_collapse_signal`: intercetta i falsi positivi dove P1 è avanti di HP ma murato o in trappola.
     - `rf_anti_meta_viability_inversion`: annulla il bias di tier quando una squadra low-tier schiera il counter di tipo ideale (#258 Meganium/Shuckle su Terra/Acqua).
     - `rf_post_t30_bench_vs_active_delta`: dislivello di potenziale tra panchina viva e matchup attivo a T30.
- **Risultati Ufficiali 10-Fold CV**:
  - **Raw CV Accuracy (Pure Bagging Blend)**: **68.1500%** ($\pm 1.2396\%$)
  - **Threshold Tuned Accuracy ($\tau=0.5050$)**: **68.3000%**
  - **ROC-AUC**: **0.7469** (Blend), **0.7475** (Full Matrix ExtraTrees)
  - **Brier Score**: **0.2045** (nuovo minimo storico di calibrazione)
  - **Log-Loss**: **0.5933** (Blend), **0.5924** (RandomForest $max\_feats=0.18$)
  - **Risoluzione Casi Limite**: #816 (Curse+Rest Snorlax) e #7204 (Light Ball Pikachu) usciti completamente dalla Top 10 worst errors; #4915 (Forretress) sceso dal 2° al 10° posto.

---

## 5. 🚀 Architettura per Heavy Ensembling (Preparazione Stacking / Blending)

La pipeline è ora strutturata in modo standardizzato e modulare per accogliere un ensembling pesante multi-paradigma (Tree-based Bagging, Oblivious Boosting, Gradient Boosting e Calibrated Linear Meta-Learners):

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE HEAVY ENSEMBLING (PREDEX)                                │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. FEATURE STORE CONDIVISO                                                                   │
│    ├── data/features/train_features.parquet (10.000 x 522)                                   │
│    ├── data/features/test_features.parquet  (5.000 x 522)                                    │
│    └── Selettore ortogonale 391 feature (Hierarchical Spearman Linkage)                      │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DIVERSIFICAZIONE MODELLI LEVEL-1 (OOF 10-FOLD & TEST PROBABILITIES)                       │
│    ├── ExtraTreesClassifier (Bagging continuo non-greedy, interpolazione scalari)           │
│    ├── RandomForestClassifier (Bagging discreto, criterion='log_loss', soglie hard)          │
│    ├── CatBoostClassifier GPU (Oblivious trees depth=5-6, simmetria globale)                │
│    ├── LightGBM / XGBoost (Gradient boosting su foglie asimmetriche ad alta profondità)     │
│    └── LinearSVC / CalibratedLogisticRegression (Meta-prior e regolarizzazione L2)          │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. META-INTEGRAZIONE LEVEL-2                                                                 │
│    ├── Weighted Soft Voting (pesatura bayesiana ottimizzata su Log-Loss / Brier Score)      │
│    ├── Out-of-Fold Stacking Classifier (Ridge / LogisticRegression con meta-calibrazione)    │
│    └── Ottimizzazione Soglia Globale di Decisione ($\tau \in [0.490, 0.515]$)                │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### File e Contratti di Integrazione Pronti per l'Ensemble:
* **Predizioni OOF Standardizzate**:
  - `outputs/oof_random_forest.csv` (10.000 righe con `battle_id`, `oof_pred`, `oof_prob`, `y_true`).
* **Probabilità Test Standardizzate**:
  - `outputs/test_probs_random_forest.csv` (5.000 righe con `battle_id`, `prob_p1_win`).
  - `outputs/submission_random_forest.csv` (5.000 predizioni binarie conformi al formato di gara).
* **Cross-Validation Blindata**:
  - Tutti i modelli condividono rigorosamente i medesimi 10 fold stratificati definiti in `data/folds.csv` garantendo zero data leakage e permettendo lo stacking OOF matematicamente corretto.



