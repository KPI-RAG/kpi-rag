# Step 09a notes — does more synthetic data help? (extends the cell-23 sweep of the synthetic-data notebook)

Screening as in the original: seeds 42 & 7 × 5 session folds = 10 folds; synthetic + Config B + FFT + trend
features; synthetic windows generated once per fold at 2000/class and truncated (identical
to generating 500 / 1000 directly).

| Windows per class | CatBoost (the synthetic-data model) | LightGBM | QUANT |
|---|---|---|---|
| 500 (the synthetic-data model) | 0.379 | 0.403 | 0.389 |
| 1000 | 0.378 (p = 0.95) | 0.434 (+0.031, p = 0.20, 9/10) | 0.375 |
| 2000 | 0.378 (p = 0.96) | 0.409 (p = 0.79) | 0.368 |

- **CatBoost: no effect at all** (0.379 → 0.378 → 0.378). The choice of 500/class stands.
- LightGBM gains ~0.03 at 1000/class, but p = 0.20 → below the adopted rule (p < 0.05 and
  ≥ 0.03). Not acted on.
- QUANT gets worse with more synthetic data (consistent with step 08).
- **Decision: keep 500 synthetic windows per class.** "More data" is not the lever here; the
  generator reproduces the same 10 fault sessions' effects, so extra copies add no new
  information.
- Timing: ~15 min/fold, ~10 min of which is the single-threaded generator → motivated the
  synthetic-window cache (step 09z) for all later steps.
