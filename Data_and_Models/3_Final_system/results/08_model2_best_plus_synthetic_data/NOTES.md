# Step 08 notes — best algorithms + the synthetic data + trend features

Setting (from step 06): synthetic data 500/class rebuilt inside each training fold, Config B
+ FFT + G1 G2 G3 (496 features), Phase 1 parser. 25 session folds, 5 seeds.

| Algorithm (real + synthetic) | CV macro-F1 | vs real-only 0.291 | vs the synthetic-data model 0.374 | post-hoc test (300 fault windows) |
|---|---|---|---|---|
| **LightGBM** | **0.412 ± 0.061** | +0.121, p = 0.026, 22/25 | +0.038, p = 0.37 (Wilcoxon 0.02), 19/25 | **macro-F1 0.609 / accuracy 0.733** |
| CatBoost + catch22 | 0.410 ± 0.096 | +0.119, p = 0.001, 25/25 | **+0.035, p = 0.038, 23/25** | 0.572 / 0.693 |
| CatBoost (the synthetic-data model + trend) | 0.391 ± 0.106 | +0.100, p = 0.0006, 25/25 | +0.017, p = 0.27, 19/25 | 0.591 / 0.703 |
| QUANT | 0.381 ± 0.073 | +0.090, p = 0.031, 21/25 | +0.007, p = 0.85 | 0.492 / 0.637 |
| *reference: the synthetic-data report* | | | | real-only 0.586 / 0.705; synthetic+B+FFT 0.557 / 0.680; Phase 1 acc 0.677 |

Findings
- **LightGBM + synthetic + trend features is the first synthetic-trained model that also wins on the
  held-out sessions**: test macro-F1 0.609 / accuracy 0.733 vs real-only 0.586 / 0.705 and
  Phase 1 0.677. The synthetic-data model on its own had lost on test (0.557); the algorithm change fixes
  that transfer problem. Lowest fold variance of all (0.061).
- CatBoost + catch22 is the only variant significantly better than the synthetic-data model under the
  conservative corrected test (p = 0.038), but it does not transfer as well to test (0.572).
- **Synthetic data hurts QUANT** (real-only 0.428 in step 07 → 0.381 here; test 0.49). QUANT reads
  raw windows, and the transplanted fault segments apparently carry artefacts it latches onto.
  QUANT stays a candidate *without* synthetic data (its post-hoc test is still to be measured).
- Candidate shortlist for the final Model 2: (a) LightGBM + synthetic + trend, (b) QUANT real-only,
  (c) an ensemble of both (step 09c). Step 09a tests whether more synthetic data (1000, 2000/class)
  changes the picture.
- Reproducibility note: the server container restarted mid-run (16 GB memory limit, two jobs at
  once); the run resumed from `folds_partial.csv` at fold 13 with identical results (deterministic seeds).
