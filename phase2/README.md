# KPI-RAG Phase 2 — Data & Models

Reproducible evaluation and improvement of the two KPI-RAG classifiers on the TelecomTS dataset:

- **Model 1** — binary anomaly detector (Normal vs fault): the "alarm".
- **Model 2** — 11-class fault-type classifier: the "diagnosis".

Author: Farida El Ghabary (Data & Models workstream), Group 15 · Supervisor: Prof. Yuanzhu Chen.
New here? Read `PROJECT_STORY.md` first (plain-language overview), then `docs/final_handoff.md`.

## Final result in one table (5 held-out recordings, 6,312 windows, 300 true faults)

| | Phase 1 | Phase 2 final |
|---|---|---|
| Model 1 false alarms | 1,105 | **116** |
| Model 1 faults caught | 269 / 300 | 266 / 300 |
| Model 2 accuracy on the true faults | 0.677 | **0.703** |
| Model 2 cross-validation macro-F1 (25 session folds) | 0.291 | **0.389** |
| Windows handed to the explanation layer as "fault" | 1,374 (1,105 wrong) | **382 (116 wrong)** |

Final configuration: Model 1 = CatBoost on 64 statistics, correct parser, imbalance corrected once, threshold 0.5,
raw probability as confidence (calibration tested and rejected). Model 2 = CatBoost, real + Farida's v3 synthetic data (500/class), Config B + FFT + trend
features, 11 classes, no reject class. Full numbers and the reasoning: `docs/phase2_results.md` (and `.docx`).

## Where things are

```
phase2/
├── PROJECT_STORY.md      beginner-friendly story of the whole project
├── PROJECT_STATUS.md     current state, Category A/B/C priorities
├── STATUS.md             step table + decisions
├── EXPERIMENT_LOG.md     every experiment: hypothesis, before/after, decision
├── DECISIONS.md          dated list of decisions and why
├── requirements.txt      pinned library versions (+ notes on aeon/shap)
├── src/                  shared code (imported by every script)
│   ├── config.py         paths, seeds, hyper-parameters, protocol constants
│   ├── data.py           session-split loading, integrity checks, Config A features, Phase 1 parser replica
│   ├── models.py         Model 1 / Model 2 recipes, 80/20 rebalancing
│   ├── features.py       new feature groups (segment shift, shape, co-movement)
│   ├── v3.py             Farida's synthetic-data method v3 + FFT + Config B (verbatim port) + cache helper
│   ├── evaluation.py     metrics, threshold rules, Nadeau-Bengio corrected paired t-test
│   ├── pipeline.py       final cascade -> Layer 2 JSON (Phase 1 schema), idle flag, validation
│   └── utils.py          per-step logging and result saving
├── scripts/              one numbered script per step (see table below)
├── results/              one folder per step: NOTES.md, log.txt, folds.csv, results.json, ...
│   └── final/            FINAL ARTIFACTS: model1.cbm, model2.cbm, feature_spec.json,
│                         normal_means.json, layer2_output_sessionsplit_phase2.json, validation.json
└── docs/                 phase2_results.md/.docx (report), future_work.md, final_handoff.md, research notes
```

## Inputs (read-only, never modified)

On the course server (`KPI_RAG_ROOT`, default `/home/jovyan/kpi_rag`):
`Models Improvement/Data Splits/` (Raneem's session split: `train_X/y/idx`, `test_idx`, `session_id_by_row`),
`Models Improvement/Synthetic/` (`raw_kpis.npy` and `raw_meta.csv` = the REAL dataset parsed once,
plus Farida's `signal_cache_v3.npz`, `effects_v3.pkl`, `fault_recipes.json`, `context.csv`), and `data/telecomts.csv`.
Step 01 stores the SHA-256 of each input in `results/01_baseline/manifest.json`.
No synthetic windows are ever used for validation or testing.

## Steps (run from the `phase2/` folder: `python scripts/<script>.py`)

| Script | Question it answers | Needs first | Time |
|---|---|---|---|
| `01_freeze_data_and_reproduce_baseline` | Can we reproduce the Phase 1 Model 2 baseline (0.2908)? | — | 10 min |
| `02_model1_threshold` | Is 0.5 a good Model 1 threshold? | — | 1 min |
| `03_model1_fixes` | Do the two Model 1 bug fixes help? | — | 3 min |
| `04_diagnose_sessions` | Why does Model 1 fail on some recordings? | — | 1 min |
| `05a_model1_new_features` / `05b_model2_new_features` | Do the new trend/co-movement features help? | 01 (for 05b) | 5 / 40 min |
| `06_model2_farida_final_and_next_steps` | Reproduce Farida's final model; her next steps 1–2 | 01 | 2 h |
| `07_model2_algorithms` | Is CatBoost the right algorithm? (9 methods) | 01, aeon | 2 h |
| `08_model2_best_algorithms_plus_v3` | Best algorithms + synthetic data | 06, 07 | 1.5 h |
| `09a_synthetic_amount_sweep` | Does more synthetic data help? | — | 2.5 h |
| `09z_build_synth_cache` | Build the synthetic-window cache (25 folds) | — | 10 min |
| `09c_model2_candidates_and_ensemble` | Final Model 2 candidates, per-class + per-session results; also writes the full-training synthetic cache `all_s42_n500_bug.npz` | 09z | 1.5 h |
| `10a_model1_shift_experiments` | Model 1 alternatives for the cross-session failure | — | 30 min |
| `10b_session0_diagnosis` | Why every detector fails on session 0 | — | 1 min |
| `10_cascade_reject_option` | Complete cascade + Raneem's reject option | 09z, 09c, (10a) | 1.5 h |
| `10c_reject_policy` | Reject by probability threshold instead of argmax | 10 | 10 min |
| `12a_idle_ambiguity` | Idle-link flag; Model 1 calibration tested and rejected | 10 | 5 min |
| `11_final_pipeline` | **Train final models, write the Layer 2 JSON and `validation.json`** | 09c, 10, 12a, shap | 5 min |
| `12b_pair_specialist` | Specialist for HNC-Gradual vs Buffer Overflow (not adopted) | 09c | 30 min |
| `12c_session_statistics` | Session-level confidence intervals and power analysis | 03, 06, 09c, 10a | 1 min |
| `_run_after.py` | Helper: start a script when another one has finished (one job at a time) | — | — |

**Shortest path to the final deliverables** (about 4 h): `01` → `09z` → `09c` → `10` → `12a` → `11`.
Long scripts checkpoint per fold and resume if interrupted.

## Environment

Python 3.12, 16 GB RAM container (run **one training job at a time**). Versions in `requirements.txt`.
`aeon` (QUANT, MiniRocket, Hydra) and `shap` are installed with `--no-deps` into `~/.kpi_rag_vendor` so they cannot change
the core libraries; the scripts add that folder to `sys.path` themselves. To run elsewhere: `export KPI_RAG_ROOT=/path/to/kpi_rag`.
All randomness is seeded (`src/config.py`): re-running gives identical numbers.

## Evaluation protocol (applies to every script)

- Session-level split only (Raneem's split: 33 sessions; test = {1, 3, 4, 13, 17}); cross-validation grouped by session,
  5 folds × 5 seeds. The paper's random split of overlapping windows is **not** used (it leaks).
- Methods are compared with the Nadeau-Bengio corrected paired t-test; a change is adopted only if p < 0.05, the gain is
  material (≥ 0.03 macro-F1) and it holds on the held-out sessions.
- The test sessions were already used in Phase 1, so every test number is reported as **post-hoc** and never used to choose.
- Synthetic windows enter training folds only, rebuilt inside each fold from training-fold sessions.

## Validation of the final output

`results/final/validation.json` (written by step 11): 6,312 records with the exact Phase 1 schema, 0 invalid fields
(every `channel` is one of the 16 KPIs), saved models reload and reproduce the same predictions, and a side-by-side with
the Phase 1 JSON. There is no separate unit-test suite; correctness is checked by the reproduction steps (01, 03, 06) and
by this validation file.
