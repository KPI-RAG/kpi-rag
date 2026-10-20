# KPI-RAG Phase 2 — Data & Models: Methods, Experiments and Results

Farida El Ghabary · 2026-10-14
Code and all result files: `Data_and_Models/3_Final_system/` in the project repository on GitHub.

## 1. Summary

Phase 2 took the Phase 1 pipeline (CatBoost cascade on 64 Config A statistics, session-level split) and
the post-Phase-1 synthetic-data work, reproduced both exactly, found and fixed two defects in Model 1,
validated the synthetic method statistically, compared nine algorithms and five feature/augmentation
strategies under one frozen protocol, and delivered a final cascade with the Phase 1 Layer 2 JSON schema.

| | Phase 1 (12-Sep) | Phase 2 final | Evidence |
|---|---|---|---|
| Model 1 on the 5 held-out sessions | TP 269 / **FP 1,105**, precision 0.196, F1 0.32 | TP 266 / **FP 116**, precision 0.70, F1 0.78 | §4 |
| Model 2 test accuracy (300 fault windows) | 0.677 | **0.703** (macro-F1 0.537 → 0.591) | §6 |
| Model 2 CV macro-F1 (25 session folds) | 0.291 (baseline) | **0.389** (p = 0.002 vs baseline) | §5–6 |
| Synthetic data vs real-only | "consistent, not significant" | **significant, p = 0.006, 25/25 folds** | §5 |
| Catch-all (false alarms → CCI-Mild) | 82 % | [pending step 10] | §7 |

Everything was decided on the 28 training sessions (session-grouped cross-validation); the 5 test sessions
were used once per step as a post-hoc check, never for selection.

## 2. Data, split and protocol

- TelecomTS windows: 16 KPI channels × 128 steps (100 ms); 32,000 windows; 1,235 anomalous (3.9 %); 11 fault types.
- Sessions: 33 recordings reconstructed by Raneem (300 s gap). Test = sessions {1, 3, 4, 13, 17}
  (6,312 windows, 300 faults); training = 28 sessions (25,688 windows, 935 faults).
- **Session structure found in Phase 2 (step 04):** 18 training sessions are Normal-only, 9 are fault-only,
  and only session 0 mixes both (3,007 Normal + 93 real Jamming). Every injected-fault session is 100 % faults.
- Protocol (frozen in step 01): StratifiedGroupKFold by session, 5 folds × 5 seeds; Nadeau-Bengio corrected
  paired t-test (Wilcoxon as check); a change is adopted only if corrected p < 0.05, the gain is material
  (≥ 0.03 macro-F1 or a material cascade gain) and it holds on the held-out sessions.
- Baseline reproduced to 4 decimals: Model 2 real-only 0.2908 ± 0.0902; the synthetic-data model (synthetic data + Config B + FFT) 0.3743.
- The TelecomTS paper's own benchmark uses a random split of overlapping windows (`src/run.py`, line 26), so its
  numbers (e.g. Toto 0.848) are not comparable with session-level results and were not reproduced.

## 3. What was wrong in Phase 1 (verified against the original artifacts)

1. **Model 1 corrected the class imbalance twice**: 80/20 rebalancing *and* `auto_class_weights="Balanced"`
   (`make_base_clf()` in notebook [8]). Effect: the detector over-flags.
2. **Feature parser bug**: the regex `[-+]?\d*\.?\d+(...)` reads `1.e-05` as two numbers (`1`, `-5`). BLER lies
   in [0, 1] (TelecomTS paper, Appendix), so BLER = −5 is impossible; 492 training windows (1.9 %) affected.
3. Report corrections: Table 6 was built from notebooks 03/03b/03d that were never executed in the submitted
   state; the 4.05 % near-duplicate figure could not be reproduced; Config A was chosen on the test set;
   the statement in `02b_split_analysis` ("anomalies in 2 of 33 sessions") is superseded by Raneem's split (13 of 33).
4. Phase 1's Layer 2 export chose the SHAP source by ground truth (not available at inference); Phase 2 chooses
   it from the models' own output (§8).

## 4. Model 1 (binary detector)

### 4.1 Bug fixes (step 03, 2 × 2 design)
| Variant (CV, t = 0.5) | PR-AUC | precision | recall | false alarms / fold |
|---|---|---|---|---|
| Phase 1 (both defects) | 0.440 | 0.365 | 0.530 | 1,175 |
| fix parser | 0.442 | 0.375 | 0.532 | 1,171 |
| fix weighting | 0.498 | 0.445 | 0.494 | 1,020 |
| **fix both (adopted)** | **0.492** | **0.440** | **0.499** | **1,021** |

False alarms fall in 25/25 folds; corrected p = 0.24 (one fold, sessions 0/6/12, dominates the variance).
Held-out sessions: FP 1,105 → 116, F1 0.32 → 0.78, recall 0.897 → 0.887. Threshold stays 0.5: the
per-fold optimum ranges from 0.008 to 0.927 (step 02), so no single threshold transfers.

### 4.2 Why every detector fails on session 0 (steps 04, 10a, 10b)
- 90 % of the remaining false alarms come from two Normal-only sessions (0 and 5) whose KPI levels differ from
  the training sessions; session 0 is *inverted* (Normal p ≈ 0.99, its real Jamming p ≈ 0.01).
- Fold 4 (sessions 0, 6, 12) scores PR-AUC ≈ 0.02 for **every** detector tried, level-based (CatBoost) and
  shape-based (QUANT) alike (step 10a) → the cause is in the data.
- Root cause (step 10b): session 0's "Normal" windows are an **idle link** (median TX 44 B, PRB ≈ 0, UL_MCS 9),
  which is what session 2's Jamming looks like (TX 8 kB, PRB 1, UL_MCS 9); session 0's own Jamming is a
  *loaded* link (TX 438 kB). With two contradictory jamming recordings, "idle" and "blocked by jamming"
  cannot be separated by any classifier. Every Normal session is also perfectly fingerprintable (AUC 1.0),
  so a held-out session is always an extrapolation.

### 4.3 Targeted alternatives (step 10a)
Two hypotheses were tested with purpose-built variants: **H1** (the detector relies on absolute levels) →
level-jitter augmentation (each training window duplicated with its per-channel level shifted by a random offset
drawn from the between-session spread of Normal medians, estimated inside the training fold); **H2** (shape
features transfer across recordings) → QUANT as detector, and a QUANT + CatBoost average.

| Variant | CV PR-AUC | CV F1 | CV false alarms/fold | vs adopted (corrected p) | Held-out: PR-AUC · TP · FP |
|---|---|---|---|---|---|
| adopted (step 03) | 0.492 | 0.376 | 1,021 | — | 0.893 · 266 · **116** |
| adopted + G (trend features) | 0.532 | 0.430 | 959 | +0.040 (p = 0.23, 20/25) | **0.917** · 270 · 123 |
| level-jitter augmentation | **0.550** | **0.453** | **879** | +0.058 (p = 0.17, 22/25) | 0.618 · 215 · **805** |
| QUANT | 0.484 | 0.384 | 1,261 | −0.008 | 0.538 · 257 · **2,512** |
| QUANT + CatBoost | 0.527 | 0.450 | 1,035 | +0.035 (p = 0.36) | 0.886 · 267 · 124 |

Interpretation. The jitter idea works where it was aimed (recall on fault-only sessions 12/18 rises from
0.41/0.20 to 0.68/0.51 in CV) but it moves the decision boundary into the levels of the two Normal-only test
sessions (3 and 4: 422 and 383 windows flagged) — a cross-validation gain that does not transfer, hence rejected.
QUANT alone cannot separate a quiet Normal recording from a fault recording (session 4: 2,117 of 3,006 windows
flagged): absolute levels are necessary for detection, consistent with the level-free result of step 05a and the
paper's scale ablation. Adding the trend features (adopted + G) is the only variant better in both CV and on the
held-out sessions (PR-AUC 0.893 → 0.917, +4 TP, +7 FP) but p = 0.23 → *promising, uncertain*; by the rule
**Model 1 stays the step-03 detector**. Session 0 remains at a 94–98 % false-alarm rate for every variant (§4.2).

## 5. Synthetic data — audit and experiments

Audit of `src/synthetic_data.py` (verbatim port of `Synthetic_Data_Part1.ipynb` Part 7): donors and hosts come only from
the 28 training sessions; generation is nested inside CV (effects, references, bounds and hosts rebuilt from
training-fold sessions; validation = real windows only); seeds fixed; physical limits and real-data bounds
enforced; statistical gating (Wilcoxon, α = 0.001). The 25-fold cache (step 09z) reproduces inline generation
exactly (max difference 0.0). The synthetic-data model reproduced in 25/25 folds (0.3743).

| Experiment | Result | Decision |
|---|---|---|
| synthetic + Config B + FFT vs real-only (step 06) | 0.374 vs 0.291, **p = 0.006, 25/25 folds** | synthetic method validated |
| Next step 1: exclude HNC-G/BO from augmentation | 0.377, variance ↑ (0.126) | no gain |
| Next step 2: + trend/co-movement features | **0.389**, p = 0.002 vs baseline, 24/25 | adopted |
| Parser fix in the generator | 0.385 | negligible; kept for correctness in the final pipeline where possible |
| More synthetic windows: 500 → 1,000 → 2,000 /class (step 09a) | CatBoost 0.379 → 0.378 → 0.378 | keep 500 |
| Synthetic data with QUANT | 0.428 (real) → 0.381 | synthetic artefacts hurt raw-window models |

Three limits (not errors): (1) every synthetic fault copies effects from the same 10 fault sessions — no new
session context, which is why more windows do not help; (2) synthetic faults sit on Normal-session
backgrounds while real faults live in fault-only sessions (CV gain only partly transfers to held-out sessions);
(3) the signal cache inherits the Phase 1 parser (effect on Model 2 negligible, step 06). Host/donor
alternatives were not tested in Phase 2 (time); see §10.

## 6. Model 2 (11-class fault classifier)

### 6.1 Algorithms (step 07, real data only)
QUANT 0.428 (p = 0.008), CatBoost + catch22 0.402 (p = 0.021), LightGBM 0.391 (p = 0.020) beat the baseline
significantly; XGBoost 0.358, CatBoost on the same 240 features 0.346, MiniRocket 0.328, Hydra 0.314,
ExtraTrees 0.309 did not. With synthetic data (step 08) LightGBM reached 0.412 but not significantly over
the synthetic-data model (p = 0.37) → by the pre-registered rule no algorithm switch.

### 6.2 Final candidates measured for real (step 09c)
| Candidate | CV macro-F1 | test macro-F1 / accuracy | test acc. sessions 1 / 13 / 17 |
|---|---|---|---|
| the synthetic-data model (CatBoost + synthetic + B + FFT) | 0.374 | 0.537 / 0.663 | 1.00 / 0.25 / 0.64 |
| **+ trend features (final)** | **0.389** | **0.591 / 0.703** | 0.99 / 0.29 / 0.72 |
| QUANT, real only | 0.426 | 0.463 / 0.623 | 1.00 / 0.10 / 0.63 |
| CatBoost + catch22, real only | 0.393 | 0.608 / 0.730 | 1.00 / 0.29 / 0.77 |
| soft vote CatBoost+trend & QUANT | 0.453 (p = 0.08) | 0.516 / 0.660 | 1.00 / 0.06 / 0.73 |
| soft vote of all three | 0.446 (p = 0.051) | 0.574 / 0.697 | 1.00 / 0.16 / 0.76 |

Per-class (pooled OOF F1): CatBoost and QUANT fail on *different* faults — Jamming 0.00 vs 0.99, Doppler 0.77
vs 0.93, RF Filters 0.34 vs 0.77, but Buffer Overflow 0.61 vs 0.36, HNC-Sudden 0.37 vs 0.00. Jamming exists in
only two training sessions; QUANT's quantile/shape features transfer between them, CatBoost's level features
do not. On the held-out sessions (where both Jamming sessions are in training) the vote's advantage vanishes
and session 13 drags it down. HNC-Gradual → Buffer Overflow confusion remains (30 of 37 test windows).

## 7. The cascade and the false-alarm filter (step 10, 10c)

Cascade = step-03 Model 1 (t = 0.5) → Model 2 (the synthetic-data model + trend). Raneem's false-alarm filter adds a
"Normal" class to Model 2 from the 1,000 Normal training windows the detector finds hardest; in Phase 2 these
are chosen **out-of-fold** (session-grouped OOF probabilities) so Model 2 never sees its own validation
sessions. The 1,000 hardest windows have p ≥ 0.991 — the idle-link windows of sessions 0 and 5 (§4.2).

| Model 2 (25 session folds) | macro-F1 | real faults wrongly rejected |
|---|---|---|
| no false-alarm filter | 0.389 ± 0.108 | 0 % |
| + hard-negative Normal class | 0.369 ± 0.116 | 19.8 % |
| + random Normal class (control) | 0.361 ± 0.090 | 17.8 % |

| Cascade on the 5 held-out sessions | FP in | FP rejected | residual catch-all (CCI-Mild) | TP rejected | useful precision |
|---|---|---|---|---|---|
| Phase 1 | 1,105 | — | 82.2 % | — | — |
| Phase 2, no reject | **116** | 0 | 64.7 % (75 windows) | 0 | 0.521 |
| Phase 2, hard-negative false-alarm filter | 116 | 16 | 68.0 % | 0 | 0.549 |
| Phase 2, random reject | 116 | 21 | 66.3 % | 3 | 0.545 |

Raneem's promotion rule (residual catch-all < 41.1 % and macro-F1 within 2 SD) is **not met** by either variant.
Diagnosis: the false-alarm filter solves the Phase 1 problem — 1,105 false alarms dominated by idle-link windows,
of which Raneem's in-sample hard negatives removed 83 % — but after the Model 1 fixes only 116 false alarms
remain and they are a different population (Normal windows of test sessions 3/4 whose levels resemble faults).
In absolute terms the catch-all shrank from ≈ 908 windows (82 % of 1,105) to 75 (65 % of 116): Model 1
removed most of it, the residual is a Model 2 confusion (CCI-Mild is the fault class nearest to normal-looking
windows). Because the argmax decision also rejects ~20 % of real faults in cross-validation, step 10c tested a
probability policy (reject only if P(Normal) ≥ τ, τ = 0.5 … 0.9): fault rejection in CV falls from 13 % (τ = 0.5)
to 4.8 % (τ = 0.9), but the number of residual false alarms removed on the held-out sessions falls from 13 to 0
at the same time. No threshold removes false alarms without rejecting real faults, so **the final pipeline has
no false-alarm filter**; the false-alarm filter is a negative result under the fixed Model 1, with the Phase 1 numbers
explaining why it had looked so promising (its hard negatives matched the idle-link false alarms that the
Model 1 fix has since removed).

## 8. Final pipeline and Layer 2 JSON (step 11)

`scripts/11_final_pipeline.py` (+ `src/pipeline.py`) trains the final models on all 28 training sessions and
writes the artifacts in `results/final/`:

| Artifact | Content |
|---|---|
| `model1.cbm` | CatBoost detector, 64 Config A features (correct parser), 80/20 rebalance, no class weights, t = 0.5; trained on 4,675 rows |
| `model2.cbm` | CatBoost 11-class classifier, 496 features (Config B + FFT + G1 G2 G3), real 935 + 4,250 synthetic windows (5,185 rows) |
| `feature_spec.json`, `normal_means.json` | feature names/order, parser per model, hyper-parameters, library versions; reference means for `feature_vs_normal` |
| `layer2_output_sessionsplit_phase2.json` | 6,312 test windows, exact Phase 1 schema |
| `validation.json` | schema checks, cascade counts, side-by-side with Phase 1, reload test |

Validation: 6,312 records, schema identical to Phase 1 (`window_index, anomaly_type, confidence, shap_top3,
signal_statistics`), 0 invalid field values (every `channel` is one of the 16 KPIs; `shap_effect` and
`feature_vs_normal` take only the Phase 1 values), and reloading the saved models reproduces every prediction.

On 18 October 2026 an audit found that the KPI names of the patch features in `shap_top3` were in the wrong order: the code named them patch by patch, while the features are built KPI by KPI, so 121 of the 382 fault records showed a wrong KPI. The names were corrected and the JSON regenerated; labels, confidences, models and metrics did not change.

| Layer 2 output | Phase 1 JSON | Phase 2 JSON |
|---|---|---|
| windows labelled as a fault | 1,374 (269 true + 1,105 false alarms) | **382 (266 true + 116 false alarms)** |
| windows labelled Normal | 4,938 | 5,930 |
| exact fault type correct (of 300 true faults) | 0.677 × 300 ≈ 203 (accuracy on flagged faults) | 199 end-to-end (266 flagged × 0.748 label accuracy) |
| false alarms labelled CCI-Mild | 82.2 % of 1,105 (≈ 908 windows) | 64.7 % of 116 (75 windows) |

Two reporting complements were added on 2026-10-15 without changing the schema: (a) flagged windows with an **idle link** (no traffic,
no PRBs) carry `note: "idle_link_ambiguous"` in their `shap_top3` entries (50 windows: 45 false alarms, 5 true
faults — a review queue with 90 % precision on false alarms), (b) `validation.json` reports a family-level
accuracy (HNC-Gradual + Buffer Overflow + HNC-Sudden as one "congestion" family): 0.75 vs 0.663 exact
end-to-end. Undefined co-movement features (correlation of a flat KPI) are NaN: CatBoost treats them as missing,
`normal_means.json` is computed NaN-safe, and such features are never written to `shap_top3`.

**A calibration that was tested and rejected (step 12a).** An isotonic regression fitted on out-of-fold Model 1 probabilities lowered the calibration error (0.133 → 0.018 on the held-out sessions) but mapped every raw score between 0.30 and 0.99 to the same value (8.5 %): 2,908 of the 3,206 training windows scoring ≥ 0.9 are Normal windows of the two Normal-only sessions 0 and 5, so the map learned "a high score is usually a false alarm", which does not hold for other recordings. It erased the ranking information, was reported by a teammate and confirmed, and was removed. `confidence` in the Layer 2 JSON is therefore the raw Model 1 probability for Normal windows and Model 2's probability for fault windows, as in Phase 1 (which labelled it uncalibrated); no label or count changed. Model 1's reliability is reported empirically instead (precision 0.70 at t = 0.5 on the held-out sessions).

Two implementation differences from Phase 1, both internal to the schema: (1) the SHAP source is chosen from
the models' own output (Phase 1 used the ground-truth label, which is unavailable at inference); (2) the three
cross-KPI co-movement features (`n_shift_*`, `onset_*`, `diff_*`) are excluded from `shap_top3` so that the
`channel` field always names a KPI, as Rodina's alignment check expects. Model 2 reads the Phase 1-parsed
signals (as the synthetic cache does); Model 1 reads the correctly parsed signals — both parsers are recorded
in `feature_spec.json`.

### 8.1 Four real windows from the final JSON (held-out sessions)
| | Window | Truth | Phase 1 output | **Phase 2 output** | What the record says |
|---|---|---|---|---|---|
| A | 592 | CCI-Mild | Faulty RF Filters | **CCI-Mild** (conf 0.90) | top KPIs: RX_Bytes above normal, UL_SNR below normal; UL_BLER 0.115, RSRP −110.5 dBm |
| B | 19094 | Normal | CCI-Mild (false alarm) | **Normal** | idle window: TX 46 B, PRB 0 — no longer flagged |
| C | 595 | Faulty Handover | Faulty RF Filters | Normal (**missed**) | loaded link (TX 400 kB, UL_BLER 0.106) that Model 1 scores 0.27 |
| D | 19052 | Normal | Doppler Shift | Doppler Shift (**false alarm**) | DL_MCS below normal; quiet window of test session 3 |

Remaining errors on the 6,312 test windows: 116 false alarms (75 labelled CCI-Mild, 20 Faulty Handover, 8
Antenna Failure, 6 Doppler) and 34 missed faults (11 HNC-Sudden, 8 RF Filters, 6 Faulty Handover, 5 HNC-Gradual,
3 Buffer Overflow, 1 CCI-Mild). The misses are the quiet-onset classes, consistent with §4.2.

## 9. How to run
```
cd Data_and_Models/3_Final_system
python scripts/01_freeze_data_and_reproduce_baseline.py   # ... through 11_final_pipeline.py
```
All randomness is seeded; `requirements.txt` pins the versions; aeon/shap live in `~/.kpi_rag_vendor`.

## 10. Handling the three remaining limits (step 12)

**Session 0 — idle link vs blocked-by-jamming (12a).** The ambiguity is in the recordings (§4.2), so the system
now *says so* instead of guessing: an `idle_link` indicator (fixed physical thresholds: TX < 1 kB/sample,
PRB < 1 %, < 1 packet/sample) marks flagged windows whose only fault evidence is "no traffic", (an isotonic calibration of Model 1's
confidence was also tested and rejected, §8). On the held-out sessions the flag catches 45 of the 116 false alarms and touches
5 true faults; the 94 Jamming windows of session 1 are loaded-link and keep a clean label. No label is changed.

**HNC-Gradual vs Buffer Overflow (12b).** A 2-class specialist built from the physics of the pair (buffer growth
vs PRB-utilisation growth, late/early ratios, time-to-peak) re-decides windows the main model labels as either
class. Result (25 session folds): pair F1 0.298 → 0.247, macro-F1 0.389 → 0.378, HNC-Gradual→Buffer-Overflow 232/280 → 230/280 (pooled); held-out sessions 30/37 → 30/37. **Not adopted**: the specialist, trained on the same 56 + 117 real windows from ≤ 10 sessions, repeats the main model's confusion — the signature that separates the two faults is not consistently present across recordings. The family-level metric (§8, congestion family accuracy 0.75) is reported as the honest complement; resolving the pair needs more recordings (future work #3).

**Only 10 fault sessions (12c).** Session-level bootstrap confidence intervals and a power analysis quantify the
uncertainty: the gains of synthetic data and of the final model over the real-only baseline are large and
certain (+0.098, 95 % CI [+0.078, +0.117], Cohen's d = 1.9); the gain of the trend features over the synthetic-data model
is small (+0.015, CI [+0.004, +0.025], d = 0.5) and a paired test would need ≈ 31 folds, i.e. **2–3× the present
number of fault sessions**, to reach p < 0.05. The Model 1 fix (+0.052 PR-AUC, CI [+0.023, +0.085]) and the
trend features for Model 1 (+0.040, CI [+0.020, +0.065]) are in between. (Caveat: bootstrap over overlapping
seed-folds is optimistic; decisions used the corrected t-test.)

## 11. Limits, open problems and what would be needed
- Only 10 fault sessions (2 with Jamming): the dominant limit for both models; gains over the synthetic-data model are
  consistent but not significant under the conservative corrected test.
- Session 0 (idle link vs blocked link) and test session 13 (HNC-Gradual / Buffer Overflow) are unresolved by any
  model tried; they need more recordings (especially jamming with and without traffic) or richer context
  (traffic activity, mobility) in the labels.
- Deferred items, with evidence and requirements: `docs/future_work.md`.
