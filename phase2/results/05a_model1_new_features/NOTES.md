# Step 05a notes — new features for Model 1

| Variant (OOF, 25 folds, t = 0.5) | PR-AUC | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| A (adopted, step 03) | 0.492 | 0.440 | 0.499 | 0.376 | 1021 |
| A+G1G3 | 0.521 | 0.468 | 0.531 | 0.413 | 952 |
| **A+G1G2G3** | **0.532** | **0.484** | **0.545** | **0.430** | 959 |
| levelfree (G1G2G3 only) | 0.433 | 0.444 | 0.433 | 0.361 | 724 |

- **A+G1G2G3** improves every metric (PR-AUC better in 20/25 folds) and recovers the recall
  lost in step 03 (0.499 → 0.545, above Phase 1's 0.530). Corrected p = 0.23 (PR-AUC) and
  0.12 (F1), so it **does not meet the pre-registered p < 0.05 rule**.
- Post-hoc test: PR-AUC 0.893 → 0.917, recall 0.887 → 0.900, F1 0.780 → 0.779.
- **Hypothesis rejected (do not retry):** dropping absolute levels ("levelfree") cuts
  session-5 false alarms (351 → 108) but loses recall and collapses on test (PR-AUC 0.456).
  Absolute levels are needed, in line with the paper's scale ablation (Table 7).
- **Unsolved by any feature set:** session 0. ~575 false alarms per fold on its Normal
  windows and ~0% recall on its real Jamming windows, for every variant. It is a domain
  shift: real Jamming is learned from a single other session (2).
