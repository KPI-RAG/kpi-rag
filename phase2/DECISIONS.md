# DECISIONS — KPI-RAG Phase 2

| Date | Decision | Why / evidence | Owner |
|---|---|---|---|
| 2026-10-13 | Continue from Farida's Final Experiments (26-Sep); her report's Next Steps are the roadmap | her instruction | Farida |
| 2026-10-13 | Raneem's session split is authoritative (300 s gap, 33 sessions, test {1,3,4,13,17}); never the paper's random window split | later work, matches Phase 1 report; paper's `run.py` line 26 leaks | Farida |
| 2026-10-13 | Model 2 baseline to beat = 0.2908 ± 0.0902 (5×5 grouped CV) | reproduced exactly from her harness | — |
| 2026-10-13 | Model 1 threshold stays 0.5 | per-fold optimum ranges 0.008–0.927 (step 02) | — |
| 2026-10-13 | Adopt fixed Model 1 (correct parser, no double weighting) | both are bug fixes; FP 1105 → 116 on test, 25/25 folds fewer FA | Farida |
| 2026-10-13 | Algorithm/setting changes need p < 0.05 AND ≥ 0.03 AND hold on test; LightGBM (+0.04, p=0.37) not adopted | "tiny differences do not matter" | Farida |
| 2026-10-13 | Keep 500 synthetic windows per class | 09a: no effect for CatBoost | — |
| 2026-10-13 | Final Model 2 = her recipe + trend features (CatBoost, real + v3, Config B + FFT + G) | 09c: consistent in CV (18/25) and test (acc 0.703 vs Phase 1 0.677) | Farida |
| 2026-10-13 | One training job at a time; no parallel workers | container limit 16 GB; two restarts | — |
| 2026-10-13 | Not pursued: TabPFN, Mantis, more algorithms, deep nets, HIVE-COTE 2 | evidence review: no expected gain at n=935 vs cost | — |
| 2026-10-14 | Autonomous 10-hour run: Model 1 shift experiments (10a) → cascade + reject (10) → final pipeline (11) → report (12) | Farida's mandate | Farida |
| 2026-10-14 | Model 1 final = step-03 detector (no jitter, no QUANT, no G) | 10a: jitter/QUANT gains are CV-only (test FP 805 / 2,512); +G p = 0.23 → reported as promising | — |
| 2026-10-14 | Session-0 failure classified as a data limit (idle link ≈ blocked link) | 10b: model-agnostic, channel medians | — |
| 2026-10-14 | No reject class in the final Model 2 | 10/10c: catch-all rule not met (68 %), no τ removes FPs without rejecting faults; Model 1 fix already removed 90 % of the original catch-all | — |
| 2026-10-14 | Layer 2 JSON: SHAP source chosen by model output, not ground truth; cross-KPI features excluded from `shap_top3` so `channel` is always a KPI | Phase 1 used ground truth (not available at inference); schema unchanged | — |
| 2026-10-15 | NaN in co-movement features kept as "missing" for CatBoost; `normal_means` computed NaN-safe; undefined features never written to `shap_top3` | replacing NaN by 0 would invent a correlation; CatBoost handles missing values natively | — |
| 2026-10-15 | Idle-link windows flagged (`note: idle_link_ambiguous`) and Model 1 confidence calibrated; labels unchanged | 10b/12a: the ambiguity is in the data; the flag has 90 % precision on false alarms (45/50) | Farida |
| 2026-10-15 | Scope split: A = done (01–12a, 12c-1); B = 12b specialist (run once, adopt only by rule); C = `docs/future_work.md` | Farida's instruction: finish must-do items, keep optional improvements for future work | Farida |
