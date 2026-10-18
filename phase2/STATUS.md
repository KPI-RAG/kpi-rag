# KPI-RAG Phase 2 — Project Status (central brain)

_Last updated: 2026-10-13 · Owner: Farida El Ghabary (Data & Models) · Deadline: ~2026-10-16_

## Goal
Improve the two KPI-RAG classifiers, with organized, reproducible code to submit to
the supervisor:
- **Model 1:** binary anomaly detector (Normal vs fault). Main problem: false alarms.
- **Model 2:** 11-class fault-type classifier. Main problem: low macro-F1 (~0.29 CV).

Baseline = Phase 1 report (submitted 2026-09-12): CatBoost cascade, Config A (64 dims),
session-level split. Model 1 test P 0.196 / R 0.897; Model 2 test accuracy 0.677.

## Rules (apply to every step)
- Inputs in `kpi_rag/Models Improvement/` and `kpi_rag/data/` are **read-only**.
  All Phase 2 outputs live in `kpi_rag/phase2/`.
- One numbered script per step in `scripts/`; shared code in `src/`; outputs in `results/NN_*/`.
- Selection: session-grouped CV, 5 folds x 5 seeds; Nadeau-Bengio corrected t-test vs
  the current model. Test set (5 sessions: 1, 3, 4, 13, 17) = **post-hoc only**, never used to choose.
- A step is "done" only when its script runs end to end and the result is checked.

## Status

| Step | Task | Status | Result |
|---|---|---|---|
| 01 | Freeze data, reproduce Model 2 baseline | ✅ done | 0.2908 ± 0.0902 reproduced exactly (twice) |
| 02 | Model 1 threshold, out-of-fold | ✅ done | Phase 1 test CM reproduced exactly; keep t = 0.5 (unstable across sessions) |
| 03 | Fix 2 Model 1 bugs | ✅ done, **adopted** | FP −13% OOF (25/25 folds); test F1 0.321 → 0.780; p ≈ 0.24 |
| 0  | Status doc + per-step notes | ✅ done | this file, `results/*/NOTES.md` |
| 04 | Diagnose sessions 0/6/12 (Model 1 near random) | ✅ done | 18 sessions all-Normal, 9 all-fault, 1 mixed (s0). ~90% of false alarms = whole Normal sessions 0 and 5 (level shift), not spikes |
| 05 | New features G1 shift / G2 shape / G3 co-movement (`src/features.py`) | ✅ done | Model 1: PR-AUC 0.492 → 0.532, recall 0.499 → 0.545 (p = 0.23). Model 2 real-only: 0.291 → 0.332 (20/25, Wilcoxon p = 0.001) |
| 06 | **Farida's Final Experiments reproduced** (25/25 folds exact, 0.3743) + her next steps 1 & 2 | ✅ done | Next step 1 (exclude pair): no gain. **Next step 2 (trend features): 0.389, best Model 2 so far** (p = 0.0018 vs baseline) |
| 07 | Algorithm comparison for Model 2 (9 methods), real-only, same folds | ✅ done | **QUANT 0.428** (p = 0.008), CatBoost+catch22 0.402, LightGBM 0.391 all beat baseline significantly; CatBoost-A 0.291 |
| 08 | Top-3 algorithms (07) + Farida's v3 synthetic data + trend features (06) | ✅ done | **LightGBM + v3 + trend: CV 0.412, test 0.609 / acc 0.733** (beats real-only 0.586 and Phase 1 0.677 on test). Synthetic data hurts QUANT (0.428 → 0.381) |
| 09a | More synthetic data: 500 vs 1000 vs 2000 per class × CatBoost / LightGBM / QUANT (2 seeds × 5 folds, like Farida's cell-23 sweep) | ✅ done | **No gain for CatBoost (0.379 → 0.378); keep 500/class.** QUANT worsens with more synthetic data |
| 09z | Synthetic-window cache for all 25 folds + equivalence check | ✅ done | 25 folds, 769 MB, **identical to inline generation (max diff 0.0)**; ~8 s/fold |
| 09c | Model 2 candidates for real: CatBoost+v3 (her recipe) ± trend, QUANT real-only, CatBoost+catch22, soft votes; per-class F1, confusion matrices, per-session test | ✅ done | **Final Model 2 = her recipe + trend (CV 0.389, test acc 0.703 vs Phase 1 0.677)** — Farida's decision. QUANT complementary (Jamming F1 0.99 vs 0.00) |
| 10a | Model 1 alternatives for the cross-session failure: +G, level-jitter augmentation, QUANT, QUANT+CB | ✅ done | jitter best in CV (0.550, 22/25) but FP 116 → 805 on test; QUANT FP 2,512 → **Model 1 stays step-03 adopted**; +G promising (test PR-AUC 0.917) but p = 0.23 |
| 10b | Why session 0 fails for every detector | ✅ done | s0 "Normal" = idle link ≈ s2 Jamming; s0 Jamming = loaded link; every Normal session fingerprintable (AUC 1.0) → data limit |
| 10 | Cascade + Raneem's reject option (hard negatives OOF vs random vs none) | ✅ done | Model 1 cascade reproduced (TP 266 / FP 116). Reject class: 16/116 FP rejected, residual catch-all 68 % (rule 41 % failed), 20 % of real faults rejected in CV → **not promoted** |
| 10c | Reject policy by P(Normal) ≥ τ | ✅ done | no τ removes FPs without rejecting faults → **final has no reject class** |
| 11 | Final pipeline + Layer 2 JSON (Phase 1 schema) + validation | ✅ done (re-run 2026-10-15) | 6,312 records, 382 fault labels (116 false), 0 invalid fields, NaN-safe normal means, raw Model 1 confidence (calibration rejected, 12a), 50 idle-flagged, reload reproduces |
| 12a | Idle-link ambiguity flag (adopted); Model 1 isotonic calibration REJECTED (flat 8.5 %) | ✅ done | flag = 45 FP + 5 TP of 116 FP |
| 12b | Specialist HNC-Gradual vs Buffer Overflow | ✅ done, not adopted | pair F1 0.30 → 0.25; test 30/37 unchanged |
| 12c-1 | Session-level bootstrap CIs + power analysis | ✅ done | small gains need ≈ 31 folds (2–3× more fault sessions) |
| 12 | Report `docs/phase2_results.md` + docx, `PROJECT_STORY.md`, `docs/future_work.md` | ✅ done | — |
| — | TabPFN / Mantis / more algorithms | dropped | evidence: no expected gain at n = 935 vs cost |
| 09 | Her next step 4: reject option + fixed Model 1, measured on the full cascade | ✅ done (steps 10, 10c) | not promoted |
| 10 | Final cascade + one pipeline (Layer 2 JSON format unchanged) | ✅ done (step 11) | see above |
| 11 | SHAP vs Alignment Table; report (incl. her next step 3: class grouping) | ✅ report + family metric done (12c); SHAP-vs-table comparison left to the evaluation workstream | — |
| 11b | Audit 2026-10-18: patch-feature names in `shap_top3` were patch-major but values are channel-major (names only; models unaffected) | ✅ fixed in `src/pipeline.py`, JSON regenerated | 121 of 382 fault records had a wrong KPI in `shap_top3` |
| — | Her next step 5: explanation-layer evaluation (G-Eval, TeleQnA) | out of scope | Evaluation workstream (Rodina) |

## Key decisions
| Date | Decision | Why |
|---|---|---|
| 2026-10-13 | Model 2 baseline to beat = 0.2908 ± 0.0902 (5x5 grouped CV) | Reproduced exactly from Farida's harness |
| 2026-10-13 | Model 1 threshold stays 0.5 | Best threshold per fold ranges 0.008–0.927 |
| 2026-10-13 | **Adopt fixed Model 1** (`MODEL1_FIXED_FEATURES=True`, `MODEL1_CLASS_WEIGHTING=False`) | Both are bug fixes; consistent gains, no harm except −0.03 recall |
| 2026-09 (Farida) | Keep real-only Model 2 (no synthetic v3) | CV gain did not transfer to test |
| 2026-10-13 | Steps 01–05 use real data only | `raw_kpis.npy` / `raw_meta.csv` (in the `Synthetic/` folder) are a cache of the real `telecomts.csv` |
| 2026-10-13 | **Algorithm/setting changes need to be significant AND material**: adopt a change over the CatBoost pipeline only if corrected p < 0.05 and ≥ 0.03 macro-F1 in CV, confirmed on the held-out sessions. LightGBM's +0.04 (p = 0.37) is noise → **not adopted**; CatBoost stays | Farida's instruction: tiny differences do not matter |
| 2026-10-13 | **Final Model 2 = Farida's recipe + trend features** (CatBoost, real + v3 500/class, Config B + FFT + G1 G2 G3; CV 0.389, test acc 0.703). QUANT reported as complementary (Jamming/Doppler/RF-Filters), not deployed | Farida's decision after step 09c |
| 2026-10-13 | **Session-level split only, always** (Raneem's split; CV grouped by session). No experiment uses the paper's random window split, not even for comparison | Farida's instruction. The paper's leakage is reported from its code (`src/run.py` line 26: random `train_test_split` of overlapping windows), not re-run |
| 2026-10-13 | **Phase 2 continues from Farida's Final Experiments (26-Sep)**, executing her report's Next Steps; her v3 synthetic method is the Model 2 starting point (step 06 onward) | Farida's instruction |
| 2026-10-13 | **Raneem's session split is authoritative** (`data split[2]`: 300 s gap, 33 sessions, test = {1, 3, 4, 13, 17}) | Later work, matches the Phase 1 report. The "2 of 33 sessions" statement in Farida's earlier `02b_split_analysis` is superseded |

## Known defects found in Phase 1 (fixed in Phase 2)
1. Imbalance corrected twice in Model 1 (80/20 rebalance + `auto_class_weights="Balanced"`).
2. Regex parser reads `1.e-05` as `1` and `-5`, which corrupts BLER features in 1.9% of
   windows (BLER must be in [0, 1], TelecomTS Appendix p. 21).

## Blockers / risks
- Sessions 0, 6, 12 score PR-AUC ≈ 0.02 for every Model 1 variant; this one fold
  dominates every statistical comparison.
- Only 28 sessions (10 with faults), so real gains struggle to reach p < 0.05.
- Phase 1 report issues to correct (step 08): Table 6 from unexecuted notebooks;
  4.05% near-duplicate figure unreproduced; Config A chosen on the test set.
- New findings missing from the Phase 1 report (to add in step 08): every injected-fault
  session is 100% faults (only session 0 is mixed); Model 1's double imbalance correction;
  the BLER parsing bug; ~90% of false alarms from Normal sessions 0 and 5.

## Next actions
1. Read 05a/05b results; adopt a feature set only if it meets the decision rule.
2. Step 06: Model 2 re-evaluation with the adopted features.

## Environment incident (2026-10-13, resolved)
`pip install --user aeon==1.0.0` silently downgraded numpy, scikit-learn, scipy, pandas and
numba in the user site. All shadowing copies were removed; active versions were re-checked
and match `requirements.txt` exactly. Jobs running at the time had already loaded their
libraries and were unaffected. aeon 1.6.0 now lives in an isolated folder (`~/.kpi_rag_vendor`,
installed with `--no-deps`). See `requirements.txt`.

## Server limits (learned 2026-10-13, the hard way)
The JupyterHub container has **16 GB RAM** (`/sys/fs/cgroup/memory.max`), although the host
shows 1 TB. Running steps 08 and 09a concurrently exceeded it and the container restarted,
killing both jobs (checkpoints made the loss ~30 min). **Rule: one training job at a time;**
QUANT/ExtraTrees with `n_jobs <= 8`; CatBoost `thread_count` as in `src/v3.py`.

## How long jobs are run
Long scripts run as detached server processes (`tools/sync_phase2.py launch`), which
survive network drops. Their live output is in `results/_runs/<script>.out`.

## How to resume
Read this file, then `README.md` (layout + how to run), then the newest `results/*/NOTES.md`.
