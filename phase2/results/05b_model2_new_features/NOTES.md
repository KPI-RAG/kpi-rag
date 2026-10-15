# Step 05b notes — new features for Model 2 (real data only)

| Variant (25 session folds) | macro-F1 | accuracy | vs Phase 1 baseline |
|---|---|---|---|
| A_phase1 (= 0.2908 baseline) | 0.291 ± 0.092 | — | — |
| A_fixed (parser fix only) | 0.291 ± 0.097 | 0.359 | +0.000 |
| A + G1 G3 | 0.327 ± 0.082 | 0.422 | +0.036, 21/25 folds better, Wilcoxon p = 0.001, corrected p = 0.14 |
| **A + G1 G2 G3** | **0.332 ± 0.066** | **0.446** | **+0.042, 20/25 better, Wilcoxon p = 0.001, corrected p = 0.19** |

- The three paper-based feature groups give a consistent gain (+0.04 macro-F1, +9 points
  accuracy) and lower fold variance (0.092 → 0.066). The Wilcoxon test is significant; the
  Nadeau-Bengio corrected test is not (it is very conservative with 28 sessions).
- Post-hoc test (300 test fault windows): macro-F1 0.583 → 0.636, accuracy 0.697 → 0.737
  (Phase 1: 0.677). The gain transfers to held-out sessions.
- The parser fix alone changes nothing for Model 2 (as expected: 5 of 935 fault windows).
- **Not solved:** HNC-Gradual is still predicted as Buffer Overflow in 86–89 % of cases, and
  CCI-Severe as CCI-Mild in 32–38 %. These two pairs need a different idea (step 06 trend
  features inside the synthetic pipeline, class grouping as a reporting complement).
- Comparison point for step 06: real-only + G features (0.332) vs Farida's v3 + B + FFT (0.374).
