# Step 07 notes — algorithm comparison for Model 2 (real data only, same 25 session folds)

| Algorithm (input) | macro-F1 | vs Phase 1 baseline 0.2908 | folds better | time/fold |
|---|---|---|---|---|
| **QUANT** (raw 16×128 windows) | **0.428 ± 0.054** | **+0.137, p = 0.008** | 22/25 | 5 s |
| CatBoost + catch22 (A + 352 catch22 feats) | 0.402 ± 0.097 | +0.111, p = 0.021 | 23/25 | 148 s |
| LightGBM (A + FFT + G, 240 feats) | 0.391 ± 0.067 | +0.100, p = 0.020 | 23/25 | 3 s |
| XGBoost (same 240) | 0.358 ± 0.073 | +0.067, p = 0.22 | 19/25 | 3 s |
| CatBoost (same 240) | 0.346 ± 0.091 | +0.056, p = 0.11 | 21/25 | 105 s |
| MiniRocket + ridge (raw) | 0.328 ± 0.093 | +0.037, p = 0.48 | 13/25 | 2 s |
| MultiRocket-Hydra (raw) | 0.314 ± 0.081 | +0.023, p = 0.68 | 15/25 | 11 s |
| ExtraTrees (240) | 0.309 ± 0.065 | +0.018, p = 0.78 | 17/25 | 2 s |
| CatBoost, Config A (reference) | 0.291 ± 0.097 | +0.000 | 10/25 | 30 s |

- **The algorithm matters.** Three methods beat the Phase 1 baseline *significantly* (corrected
  p < 0.05) on real data alone: QUANT, CatBoost + catch22, LightGBM. QUANT is also the most
  stable (std 0.054) and the cheapest.
- QUANT (Dempster et al. 2024) works directly on the raw windows: interval quantile features
  + ExtraTrees. It needs no hand-crafted features at all.
- Convolution-kernel methods (MiniRocket, Hydra) do **not** help here: with 935 windows and
  heavy-tailed KPI scales, their ridge/kernel features generalise poorly across sessions.
- Reference line check: `catboost_A` reproduces the baseline again (0.2910 vs 0.2908).
- Step 08 takes the top 3 (quant, catboost_c22, lightgbm) + CatBoost, and trains them on
  real + the synthetic data with the trend features.
- Environment: aeon 1.6.0 installed with `--no-deps` in `~/.kpi_rag_vendor` (see requirements.txt).
