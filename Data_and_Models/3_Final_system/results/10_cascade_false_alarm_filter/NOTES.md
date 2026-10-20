# Step 10 notes — the complete cascade and Raneem's false-alarm filter

Model 1 = step-03 adopted detector (10a rule); Model 2 = the synthetic-data model + trend. Hard negatives chosen
**out-of-fold** (Model 1 OOF probabilities, 5 session folds; the 1,000 hardest Normal windows have
p ≥ 0.991 — they are the idle-link windows of sessions 0/5, see 10b).

**A. Model 2 quality (25 session folds, 11-class macro-F1 on real validation faults)**
| Model 2 | macro-F1 | faults wrongly rejected as "Normal" |
|---|---|---|
| no false-alarm filter (reference) | 0.389 ± 0.108 | 0 % |
| + hard-negative Normal class (1,000, OOF) | 0.369 ± 0.116 (within 2 SD) | **19.8 %** |
| + random Normal class (1,000) | 0.361 ± 0.090 (within 2 SD) | 17.8 % |

**B. Cascade on the 5 held-out sessions (6,312 windows; post-hoc)**
Model 1 reproduces step 03 exactly: TP 266 / FP 116 / FN 34 / TN 5,896 (Phase 1: 269 / 1,105).
| Cascade | FP entering Model 2 | FP rejected | residual FP | dominant class among residual FP | TP rejected | useful precision | correct fault label on true faults |
|---|---|---|---|---|---|---|---|
| no reject | 116 | 0 | 116 | CCI-Mild 64.7 % | 0 | 0.521 | 0.748 |
| hard-negative 1000 | 116 | **16** | 100 | CCI-Mild 68.0 % | 0 | 0.549 | 0.756 |
| random 1000 | 116 | 21 | 95 | CCI-Mild 66.3 % | 3 (1.1 %) | 0.545 | 0.733 |

Promotion (Raneem's rule, verified): residual catch-all < 41.1 % → **not met** (68 % / 66 %); F1 within 2 SD → met.
**Not promoted.**

Why it does not work here (diagnosis)
- The false-alarm filter was designed for Phase 1's 1,105 false alarms, which were dominated by idle-link
  Normal windows; Raneem's hard negatives (chosen in-sample by the Phase 1 detector) covered exactly those
  and removed 83 % of them. After the Model 1 fixes only **116** false alarms remain, and they are a
  different kind: Normal windows of test sessions 3/4 whose levels resemble faults. The idle-link Normal
  class does not cover them, so only 16 of 116 are rejected.
- In CV the same class rejects ~20 % of real faults (quiet fault windows look like idle windows) — the
  argmax decision is too aggressive. Step 10c tests a probability-threshold policy (reject iff
  P(Normal) ≥ τ) on the same models.
- The catch-all itself has shrunk from 82.2 % of 1,105 (= 908 windows) to 64.7 % of 116 (= 75 windows):
  the Model 1 fix solved most of the original problem; the residual is a Model 2 confusion
  (CCI-Mild is the class nearest to "normal-looking" windows, 10b T2).
Files: `folds.csv`, `results.json`, `oof_model1_prob.npy`, `confusion_cascade_test_*.csv`, `log.txt`.
