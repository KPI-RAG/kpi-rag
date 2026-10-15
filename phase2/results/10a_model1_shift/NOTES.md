# Step 10a notes — Model 1 alternatives targeted at the cross-session failure

Protocol: step 03 (5 seeds × 5 session folds, 80/20 rebalance in the training fold, t = 0.5).

| Variant (CV mean) | PR-AUC | precision | recall | F1 | FA/fold | FA rate s0 / s5 | recall s12 / s18 | vs adopted |
|---|---|---|---|---|---|---|---|---|
| adopted (step 03) | 0.492 | 0.440 | 0.499 | 0.376 | 1,021 | 0.96 / 0.58 | 0.41 / 0.20 | — |
| adopted + G | 0.532 | 0.484 | 0.545 | 0.430 | 959 | 0.96 / 0.54 | 0.53 / 0.23 | +0.040 (p = 0.23, 20/25) |
| **jitter** (level-jitter augmentation) | **0.550** | **0.520** | **0.568** | **0.453** | **879** | 0.94 / 0.47 | **0.68 / 0.51** | +0.058 (p = 0.17, 22/25) |
| QUANT (raw windows) | 0.484 | 0.431 | 0.554 | 0.384 | 1,261 | 0.98 / 0.63 | 0.54 / 0.06 | −0.008 |
| QUANT + CatBoost | 0.527 | 0.504 | 0.568 | 0.450 | 1,035 | 0.98 / 0.61 | 0.53 / 0.17 | +0.035 (p = 0.36) |

| Held-out sessions (post-hoc) | PR-AUC | TP | FP | FN | flagged in Normal-only s3 / s4 |
|---|---|---|---|---|---|
| adopted | 0.893 | 266 | **116** | 34 | 44 / 72 |
| adopted + G | **0.917** | 270 | 123 | 30 | 80 / 43 |
| jitter | 0.618 | 215 | **805** | 85 | 422 / 383 |
| QUANT | 0.538 | 257 | **2,512** | 43 | 395 / 2,117 |
| QUANT + CatBoost | 0.886 | 267 | 124 | 33 | 59 / 65 |

Findings
- **Level-jitter augmentation is the best in CV (22/25 folds, FA −14 %, recall +0.07) but collapses on the
  held-out sessions (FP 116 → 805).** The jitter scale is estimated from the training recordings; the two
  Normal-only test sessions (3, 4) sit at levels the jittered model now reads as faults. A CV-only gain
  that does not transfer → **rejected** by the rule (also p = 0.17).
- **QUANT as detector fails on test** (2,117 of 3,006 windows of session 4 flagged): shape features alone
  cannot tell a quiet Normal recording from a fault recording. Confirms step 05a (level-free ⇒ worse) and the
  paper's scale ablation: absolute levels are necessary for detection, even though they are what fails on
  session 0.
- **adopted + G** is the only variant that improves both CV (+0.04 PR-AUC, 20/25) and the held-out PR-AUC
  (0.893 → 0.917, +4 TP) at +7 FP; p = 0.23 → "promising, uncertain", not adopted by the rule.
- Session 0 stays at a 94–98 % false-alarm rate and ~0 % Jamming recall for every variant (see 10b: idle link
  vs blocked link — a data limit).
- Decision: **Model 1 = adopted (step 03)**; adopted+G reported as the recommended next candidate once more
  fault sessions exist. Files: `folds.csv`, `results.json`, `log.txt`.
