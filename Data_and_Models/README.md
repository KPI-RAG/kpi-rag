# Data and Models

This folder holds the data and machine-learning part of KPI-RAG, a system that detects faults in
5G network measurements, names the fault and explains it with the relevant 3GPP standard. The
models here are the first two steps of that system:

- **Model 1** looks at a 12.8-second window of 16 network KPIs and decides: normal or fault.
- **Model 2** runs on the windows Model 1 flags and names one of 11 fault types
  (for example Jamming, Antenna Failure, Buffer Overflow).

The work went through three stages, one folder each, in the order it was done.

## Final results

Measured on 5 recording sessions that were never used for training or for choosing a model
(6,312 windows, 300 of them faults).

| | First prototype | Final system |
|---|---|---|
| False alarms (Model 1 raises an alarm on a normal window) | 1,105 | 116 |
| Faults detected by Model 1 | 269 of 300 | 266 of 300 |
| Model 1 precision / recall / F1 | 0.20 / 0.90 / 0.32 | 0.70 / 0.89 / 0.78 |
| Model 2 accuracy on the 300 faults | 0.677 | 0.703 |
| Model 2 macro-F1, cross-validation (25 folds grouped by session) | 0.291 | 0.389 |
| Windows passed on as "fault" | 1,374 (1,105 wrong) | 382 (116 wrong) |

## Folder map

```
Data_and_Models/
├── 1_First_prototype/     notebooks of the first version (01 to 06)
├── 2_Model_improvement/   three experiments run after it (07 to 09)
└── 3_Final_system/        final models, experiment scripts (steps 01 to 12) and all results
```

## Data and evaluation

- **Dataset:** TelecomTS, a public dataset recorded on a real 5G lab network: 32,000 windows,
  1,235 of them faults, 11 fault types (10 injected faults and one real jamming event).
  It is not stored here because of its size (1.2 GB).
- **Split by recording session:** neighbouring windows overlap by 75 %, so a random split by
  window leaks near-copies into the test set (31.8 % of test windows had a near-identical
  training window). All results use a split by recording session instead: 33 sessions, rebuilt
  from the timestamps with a 300-second gap; 28 for training (25,688 windows, 935 faults) and
  5 for testing (sessions 1, 3, 4, 13 and 17).
- **Choosing models:** cross-validation on the training sessions only, grouped by session
  (5 folds × 5 seeds). A change was kept only if it was statistically significant
  (corrected paired t-test, p < 0.05), large enough (at least +0.03 macro-F1) and held on the
  test sessions. The test sessions were never used to choose anything.

## 1_First_prototype

| Notebook | What it does | Main result |
|---|---|---|
| `01_data_loading_and_exploration` | loads the dataset, checks its structure, class balance | 96.1 % normal windows; fault types range from 48 to 279 windows |
| `02_feature_engineering` | builds a 582-number summary of each window (statistics, patch statistics, differences, protocol) | the feature vector used by the ablation |
| `02b_split_analysis` | compares ways of splitting train and test | a split by window leaks near-copies (31.8 %) |
| `03_scale_ablation` | compares three feature sets (64, 320 and 582 features) | the smallest set (64 statistics, "Config A") scored best (saved without outputs; result from the Phase 1 report) |
| `03b_feature_investigation` | looks into why the smallest set won | too few fault windows per feature for the larger sets (saved without outputs; result from the Phase 1 report) |
| `03c_algorithm_comparison` | CatBoost, LightGBM, XGBoost, Random Forest on every feature set | CatBoost chosen |
| `03d_smote_kbest` | SMOTE oversampling and feature selection | SMOTE did not help in cross-validation, not kept (saved without outputs; result from the Phase 1 report) |
| `03e_hyperparameter_tuning` | tuning to rescue the 582-feature set | did not help, not kept |
| `04_session_split` | rebuilds the 33 recording sessions and makes the session-level split | the split used by every later result |
| `05_data_pipeline` | builds the training and test arrays | inputs for 06 |
| `06_modeling_pipeline` | trains Model 1 and Model 2, SHAP explanations, output for the explanation layer | 1,105 false alarms; Model 2 accuracy 0.677; 82.2 % of false alarms were named "CCI-Mild" |

## 2_Model_improvement

| Notebook | What it does | Result | Kept? |
|---|---|---|---|
| `07_false_alarm_filter` | gives Model 2 a 12th "Normal" class so it can send false alarms back | on the first prototype the share of false alarms named CCI-Mild fell from 82.2 % to 16.3 %, with 4.8 % of real faults sent back | no: after the Model 1 fixes only 116 false alarms remain, and a fair re-test removed only 16 of them, and in cross-validation it also sent back about 20 % of real faults |
| `08_fft_features` | adds frequency features (how much each KPI oscillates) | macro-F1 0.294 → 0.344 in 4 of 5 folds | yes, part of the final Model 2 |
| `09_synthetic_data` | creates extra fault windows by copying the effect of a real fault onto a normal window | version 1 (constant shift) and version 2 failed; version 3, with the larger feature set (patch statistics + FFT), lifted macro-F1 from 0.291 to 0.374 (better in 25 of 25 folds, p = 0.006); a model trained only on version 3 data beat one trained on real data (0.393 vs 0.311) | version 3, yes |

Why versions 1 and 2 failed: a constant shift cannot change how much a KPI varies, it changed
the whole window instead of the faulty part, and fault and normal windows come from different
recordings, so comparing them mixed the fault with the recording itself. Version 3 measures each
fault's effect inside the same window and fixes all three problems.

## 3_Final_system

The final models and every experiment behind them, as numbered scripts. Each script answers one
question and writes its results, notes and log to `results/`. Details and how to run:
`3_Final_system/README.md`. The full write-up: `3_Final_system/docs/phase2_results.md`.

**Final models**

- **Model 1:** CatBoost on 64 statistics per window, threshold 0.5. Two defects of the first
  prototype were fixed: the class imbalance was corrected twice, and a parsing bug read values
  like `1.e-05` as two numbers. These two fixes cut false alarms from 1,105 to 116.
- **Model 2:** CatBoost trained on 935 real and 4,250 synthetic fault windows (version 3), with
  496 features: statistics, patch statistics, FFT features and trend features.
- **Output:** one record per test window for the explanation layer (fault type, confidence, the
  three KPIs that mattered most according to SHAP, window statistics), in the same format as the
  first prototype.

**Tested and not kept** (none held on the test sessions or met the rule above): other algorithms
(QUANT, LightGBM, XGBoost, MiniRocket, Hydra, catch22, voting ensembles), more synthetic data
(1,000 or 2,000 per class), level-jitter augmentation for Model 1, the false-alarm filter, a
specialist for the two congestion faults, and probability calibration of Model 1.

**What still limits the results**

- Only 10 of the 28 training sessions contain faults, so small gains cannot be proven yet;
  about 2 to 3 times more fault recordings would be needed.
- In session 0 a normal idle link looks the same as a link blocked by jamming in another
  recording.
- High Network Congestion (Gradual) and Buffer Overflow are often confused: the dataset lists
  the same KPIs for both, and 30 of 37 test windows of the first are named as the second.
  Counting the three congestion faults as one family, the system names the right family for
  75 % of the faults.

## Running the code

The notebooks were run on Google Colab and on a JupyterHub server, so their file paths point to
those places; change them to your own copy of TelecomTS. The final system reads its paths from
`3_Final_system/src/config.py` (set `KPI_RAG_ROOT` to your data folder). Library versions are in
`3_Final_system/requirements.txt`.
