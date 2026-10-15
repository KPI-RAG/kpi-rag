# Step 06 notes — Farida's final model reproduced + her Next Steps 1 & 2

**Reproduction: 25/25 folds match her `configBC_records.csv` (B_fft) to 4 decimals; mean 0.3743.**
`src/v3.py` is therefore a faithful port of the 26-Sep notebook (Part 7, cells 16b-29).

| Variant (25 session folds, nested synthetic generation) | macro-F1 | vs real-only 0.2908 | vs her final |
|---|---|---|---|
| real_only (Phase 1 baseline) | 0.291 ± 0.092 | — | — |
| **final** = v3 (500/class) + Config B + FFT | 0.374 ± 0.096 | +0.084, p = 0.0065, 25/25 | — |
| excl_pair (next step 1: no augmentation of HNC-G / BO) | 0.377 ± 0.126 | +0.086, p = 0.019, 21/25 | +0.003, p = 0.91, 12/25 |
| **trend** (next step 2: + G1 G2 G3 features, 496 dims) | **0.389 ± 0.108** | **+0.098, p = 0.0018, 24/25** | +0.015, p = 0.36, 18/25 |
| trend_fixed (trend + BLER parser fix) | 0.385 ± 0.097 | +0.094, p = 0.0005, 24/25 | +0.011, p = 0.57, 15/25 |

- **Her synthetic method is confirmed significant** against the real-only baseline (p < 0.01).
- **Next step 1 (exclude the hard pair) does not help**: same mean, much higher variance. Dropped.
- **Next step 2 (trend features) is the best Model 2 so far: 0.389.** The gain over her final
  model is small and not significant (p = 0.36), but it is the most significant variant vs the
  baseline. The parser fix changes little for Model 2 (as in 05b); keep it for correctness.
- Step 08 will take `trend` as the Model 2 setting (G features on, no exclusion) and test the
  best algorithms from step 07 on top of it.
