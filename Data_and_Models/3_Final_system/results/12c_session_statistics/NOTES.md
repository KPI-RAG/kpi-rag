# Step 12c-1 notes — session-level uncertainty (bootstrap CIs and power), no retraining

Unit = seed-fold (25 per comparison; each fold holds out ≈ 2 fault sessions). Bootstrap resamples folds
with replacement (10,000 draws). Caveat: seed-folds overlap in sessions, so these CIs are *optimistic*
compared with the Nadeau-Bengio corrected t-test used for decisions; both are reported.

| Comparison | mean Δ | 95 % bootstrap CI | Cohen's d | folds needed (α 0.05, power 0.8) |
|---|---|---|---|---|
| the synthetic-data model vs real-only baseline (macro-F1) | +0.084 | [+0.064, +0.103] | 1.62 | 4 |
| **final (+trend) vs real-only baseline** | **+0.098** | **[+0.078, +0.117]** | 1.90 | 3 |
| final (+trend) vs the synthetic-data model | +0.015 | [+0.004, +0.025] | 0.51 | **31** |
| QUANT real-only vs the synthetic-data model | +0.052 | [+0.020, +0.082] | 0.63 | 20 |
| vote(CB+trend, QUANT) vs the synthetic-data model | +0.079 | [+0.048, +0.109] | 0.99 | 8 |
| Model 1 fix_both vs Phase 1 (PR-AUC) | +0.052 | [+0.023, +0.085] | 0.64 | 19 |
| Model 1 fix_both vs Phase 1 (false alarms, k/fold) | −154 | [−231, −90] | 0.83 | 12 |
| Model 1 adopted+G vs adopted (PR-AUC) | +0.040 | [+0.020, +0.065] | 0.67 | 18 |

Reading
- The two large effects (synthetic data; synthetic + trend vs baseline) are unambiguous: CI far from 0,
  d > 1.5, detectable with 3–4 folds.
- The small effects (trend over the synthetic-data model +0.015; adopted+G +0.040 PR-AUC) have CIs excluding 0 under the
  bootstrap but d ≈ 0.5–0.7, so a corrected paired test needs ≈ 18–31 folds — i.e. roughly **2–3× the
  present 10 fault sessions** — to reach p < 0.05. This is the quantitative form of "consistent but not
  significant".
- Per-fold view (seed-averaged): the trend model is ahead or equal in folds 1, 2, 5 and behind only in
  fold 4 (sessions 0, 12) — the session-0 fold again.
- What additional data would resolve it: ≈ 10–20 more fault sessions, or recordings where Normal and fault
  windows co-occur (currently only session 0), which would also lift the fingerprint problem (10b T1).

Files: `bootstrap_ci.csv`, `results.json`, `log.txt`. Part 2 (leave-one-session-out) in `12c_loso/`.
