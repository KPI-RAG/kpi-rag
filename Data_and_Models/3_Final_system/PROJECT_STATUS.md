# PROJECT_STATUS — KPI-RAG Phase 2 (Data & Models)

_Updated 2026-10-15. Companions: `PROJECT_STORY.md` (beginner story), `STATUS.md` (step table + decisions),
`EXPERIMENT_LOG.md`, `DECISIONS.md`, `docs/phase2_results.md` (report), `docs/future_work.md`._

## Verified current state (source of truth = files in `results/`)
| Component | Final configuration | Verified numbers |
|---|---|---|
| Model 1 | CatBoost, 64 Config A features (correct parser), 80/20 rebalance, **no** class weights, t = 0.5; raw probability as confidence (isotonic calibration rejected, 12a) | test TP 266 / FP 116 / FN 34 (Phase 1: 269 / 1,105 / 31) |
| Model 2 | CatBoost, real 935 + v3 synthetic 500/class, Config B + FFT + G1 G2 G3 (496 features), 11 classes, no false-alarm filter | CV macro-F1 0.389 (baseline 0.291); test acc 0.703 (Phase 1 0.677) |
| Cascade output | `results/final/layer2_output_sessionsplit_phase2.json`, Phase 1 schema | 6,312 records, 382 fault labels (116 false), 0 invalid fields, 50 idle-flagged, reload reproduces |
| Reporting complements | idle-link review flag (note field), family-level metric (congestion) | flag: 45 FP + 5 TP of 116 FP; family accuracy 0.75 vs 11-class 0.663 end-to-end |

## Work items
| Priority | Task | Status |
|---|---|---|
| Core | Reproduce baselines; fix parser + double weighting; freeze protocol | ✅ steps 01–03 |
| Core | Diagnose Model 1 errors per session; test justified alternatives | ✅ steps 04, 05a, 10a, 10b |
| Core | Validate synthetic data (leakage, nesting, amount) | ✅ steps 06, 09a, 09z |
| Core | Model 2 candidates, per-class results; cascade + false-alarm filter | ✅ steps 07–09c, 10, 10c |
| Core | Final pipeline, Layer 2 JSON, validation, NaN fix | ✅ step 11 (re-run 2026-10-15) |
| Core | Session-level uncertainty + power analysis | ✅ step 12c-1 |
| Core | Idle-link ambiguity flag (adopted); Model 1 calibration (rejected: flat 8.5 %) | ✅ step 12a |
| Extra | Specialist for HNC-Gradual vs Buffer Overflow | ✅ step 12b: not adopted (pair F1 0.30 → 0.25; test 30/37 unchanged) → future work #3 |
| Extra | Trend features for Model 1 (adopted+G) | tested (10a): promising, p = 0.23 → not adopted |
| Future | LOSO evaluation, donor/host alternatives, QUANT routing, TabPFN/Mantis, threshold tuning, more recordings | `docs/future_work.md` |

## Constraints
- 16 GB of RAM, so one training job runs at a time (`scripts/_run_after.py` starts the next one).
- Data: 10 fault sessions, 2 contradictory jamming recordings, 1 mixed recording.

## Next
All core and extra items are done. The final outputs are in `results/final/`; open ideas are in `docs/future_work.md`.
