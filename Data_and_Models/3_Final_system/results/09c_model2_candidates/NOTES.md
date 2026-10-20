# Step 09c notes — Model 2 candidates measured for real (CV + held-out sessions + per class)

| Candidate | CV macro-F1 (25 folds) | vs the synthetic-data model (corrected p / Wilcoxon) | Test macro-F1 / accuracy | Test acc per session 1 / 13 / 17 |
|---|---|---|---|---|
| catboost_synthetic (the synthetic-data model recipe) | 0.374 ± 0.096 | — | 0.537 / 0.663 | 1.00 / 0.25 / 0.64 |
| **catboost_synthetic_G** (+ trend features) | 0.389 ± 0.108 | +0.015 (p = 0.36 / 0.02), 18/25 | **0.591 / 0.703** | 0.99 / 0.29 / 0.72 |
| quant_real (no synthetic) | 0.426 ± 0.051 | +0.052 (p = 0.26 / 0.007), 18/25 | 0.463 / 0.623 | 1.00 / 0.10 / 0.63 |
| catboost_c22 (real only) | 0.393 ± 0.098 | +0.019 (p = 0.57) | **0.608 / 0.730** | 1.00 / 0.29 / 0.77 |
| vote_2 (catboost_synthetic_G + quant) | **0.453 ± 0.140** | +0.079 (p = 0.08 / 0.0003), 19/25 | 0.516 / 0.660 | 1.00 / 0.06 / 0.73 |
| vote_3 (+ catboost_c22) | 0.446 ± 0.117 | +0.072 (**p = 0.051** / 0.0000), 21/25 | 0.574 / 0.697 | 1.00 / 0.16 / 0.76 |

Reference: Phase 1 accuracy 0.677. (The synthetic-data report's test numbers averaged 5 training seeds;
here one seed, hence catboost_synthetic 0.663 vs original 0.680.)

Findings
- **No candidate meets the pre-registered rule** (corrected p < 0.05 AND ≥ 0.03 AND holds on
  test). vote_3 is at the edge in CV (p = 0.051) but is not better on the held-out sessions.
- **CatBoost and QUANT fail on different faults** (pooled OOF per-class F1):
  Jamming 0.00 vs **0.99**, Doppler 0.77 vs **0.93**, RF Filters 0.34 vs **0.77** — but
  Buffer Overflow **0.61** vs 0.36, HNC-Sudden **0.37** vs 0.00, Resource Alloc **0.28** vs 0.08.
  CatBoost's Jamming F1 of 0 in CV: Jamming exists in only 2 training sessions (0, 2); when one is
  held out the level-based features do not transfer, while QUANT's quantile/shape features do.
  That is why the vote wins in CV; on test (where both Jamming sessions are in training) the
  advantage disappears and session 13 drags the vote down.
- **Test session 13 (HNC-Gradual, Buffer Overflow, HNC-Sudden) is hard for every model
  (0.06–0.29)**; HNC-Gradual F1 ≈ 0 everywhere. Unchanged from the synthetic-data report.
- Trend features (catboost_synthetic_G): small but consistent gain in CV (18/25, Wilcoxon 0.02) and
  on test (+0.054 macro-F1, +4 points accuracy, better on sessions 13 and 17).
- Recommendation: final Model 2 = **the synthetic-data model + trend features (catboost_synthetic_G)** — same
  pipeline, same algorithm, consistent in CV and test, accuracy 0.703 vs Phase 1 0.677.
  QUANT is reported as the complementary model (Jamming/Doppler/RF-Filters specialist).
- Files: `per_class_f1_cv.csv`, `per_class_f1_test.csv`, `confusion_test_<candidate>.csv`, `folds.csv`.
