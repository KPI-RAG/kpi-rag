# Step 12b notes — specialist for HNC-Gradual vs Buffer Overflow (negative result)

Design: 2-class CatBoost trained only on the two classes (real + v3 synthetic windows of the training fold)
with growth-shape features of the congestion channels (slope, late/early ratio, time-to-peak, peak/mean of
Estimated_UL_Buffer, PRB utilisation, NPRB, packets, bytes, BLER) + differences between buffer, PRB and
packet growth; it re-decides every window the main Model 2 (final recipe) labels as either class.

| 25 session folds | main Model 2 | + specialist | better in |
|---|---|---|---|
| 11-class macro-F1 | 0.389 | 0.378 | 5/25 folds |
| pair F1 (HNC-G, BO) | 0.298 | 0.247 | 5/25 folds |
| HNC-Gradual predicted as Buffer Overflow (pooled) | 232 / 280 | 230 / 280 | — |

Held-out sessions (post-hoc): macro-F1 0.591 → 0.590, accuracy 0.703 → 0.700, HNC→BO 30/37 → 30/37.
**Not adopted** (rule: pair F1 better in ≥ 18/25 folds and macro-F1 not lower; neither met).

Why it fails (what the result teaches)
- The specialist sees exactly the windows the main model already confused and, trained on the same 56 + 117
  real windows (≤ 10 sessions), makes the same mistake: the "buffer grows while PRB stays flat" signature is
  not consistently present in the real HNC-Gradual windows of other sessions. The confusion is a property of
  the recordings, not of the classifier's capacity.
- Folds where HNC-Gradual is absent show the specialist can even hurt (fold 4: pair F1 0.28 → 0.00), because
  it cannot say "neither".
- Consistent with Farida's report (70 % confusion) and with the TelecomTS injection design (both faults are
  gradual build-ups on overlapping KPIs).

Decision: keep the single 11-class Model 2; report the family-level metric (congestion family accuracy 0.75)
as the honest complement; the pair needs more recordings (future work #3).
Files: `folds.csv`, `results.json`, `log.txt`.
