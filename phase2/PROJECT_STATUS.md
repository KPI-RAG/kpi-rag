# PROJECT_STATUS — KPI-RAG Phase 2 (Data & Models)

_Updated 2026-10-15. Companions: `PROJECT_STORY.md` (beginner story), `STATUS.md` (step table + decisions),
`EXPERIMENT_LOG.md`, `DECISIONS.md`, `docs/phase2_results.md` (report), `docs/future_work.md`._

## Verified current state (source of truth = files in `results/`)
| Component | Final configuration | Verified numbers |
|---|---|---|
| Model 1 | CatBoost, 64 Config A features (correct parser), 80/20 rebalance, **no** class weights, t = 0.5; calibrated confidence (isotonic, OOF) | test TP 266 / FP 116 / FN 34 (Phase 1: 269 / 1,105 / 31) |
| Model 2 | CatBoost, real 935 + v3 synthetic 500/class, Config B + FFT + G1 G2 G3 (496 features), 11 classes, no reject class | CV macro-F1 0.389 (baseline 0.291); test acc 0.703 (Phase 1 0.677) |
| Cascade output | `results/final/layer2_output_sessionsplit_phase2.json`, Phase 1 schema | 6,312 records, 382 fault labels (116 false), 0 invalid fields, 50 idle-flagged, reload reproduces |
| Reporting complements | idle-link review flag (note field), family-level metric (congestion) | flag: 45 FP + 5 TP of 116 FP; family accuracy 0.75 vs 11-class 0.663 end-to-end |

## Priorities (Category A/B/C)
| Cat. | Task | Status |
|---|---|---|
| A | Reproduce baselines; fix parser + double weighting; freeze protocol | ✅ steps 01–03 |
| A | Diagnose Model 1 errors per session; test justified alternatives | ✅ steps 04, 05a, 10a, 10b |
| A | Validate synthetic data (leakage, nesting, amount) | ✅ steps 06, 09a, 09z |
| A | Model 2 candidates, per-class results; cascade + reject option | ✅ steps 07–09c, 10, 10c |
| A | Final pipeline, Layer 2 JSON, validation, NaN fix | ✅ step 11 (re-run 2026-10-15) |
| A | Session-level uncertainty + power analysis | ✅ step 12c-1 |
| A | Idle-link ambiguity flag + calibrated confidence | ✅ step 12a |
| B | Specialist for HNC-Gradual vs Buffer Overflow | ✅ step 12b: not adopted (pair F1 0.30 → 0.25; test 30/37 unchanged) → future work #3 |
| B | Trend features for Model 1 (adopted+G) | tested (10a): promising, p = 0.23 → not adopted |
| C | LOSO evaluation, donor/host alternatives, QUANT routing, TabPFN/Mantis, threshold tuning, more recordings | `docs/future_work.md` |

## Blockers / constraints
- Server container 16 GB → one job at a time (`scripts/_run_after.py` chains); laptop memory kills local watchers.
- Data: 10 fault sessions, 2 contradictory jamming recordings, 1 mixed recording.

## Next
All Category A and B items are closed. Remaining: thesis chapter writing by Farida from `docs/phase2_results.md`; team hand-over of `results/final/`.
