# Step 10c notes — reject policy by probability threshold (follow-up to step 10)

Same models as step 10 (recipe + trend + 1,000 OOF hard negatives); decision "reject iff P(Normal) ≥ τ".

| τ | CV macro-F1 (seed 42, 5 folds) | real faults rejected (CV) | held-out: FP rejected of 116 | TP rejected of 266 |
|---|---|---|---|---|
| argmax (step 10) | 0.369 (25 folds) | 19.8 % | 16 | 0 |
| 0.5 | 0.348 | 13.1 % | 13 | 0 |
| 0.6 | 0.351 | 9.8 % | 5 | 0 |
| 0.7 | 0.348 | 7.6 % | 2 | 0 |
| 0.8 | 0.348 | 6.0 % | 0 | 0 |
| 0.9 | 0.362 | 4.8 % | 0 | 0 |
| no false-alarm filter | 0.389 (25 folds) | 0 % | 0 | 0 |

- There is **no threshold that both removes false alarms and keeps real faults**: by the time fault
  rejection falls below 10 % (τ ≥ 0.6), the class rejects ≤ 5 of the 116 residual false alarms; at
  τ = 0.9 it never fires. The "Normal" class learned from idle-link windows is simply not where the
  remaining false alarms live (§10 notes).
- The pre-registered CV rule picked τ = 0.9 (least harm), which is equivalent to no false-alarm filter.
- **Decision: the final pipeline has no false-alarm filter** (11 fault classes, Phase 1 behaviour). The reject
  option is reported as a negative result under the fixed Model 1, with the explanation above.
