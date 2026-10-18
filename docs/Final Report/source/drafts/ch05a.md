## 5. Results and Evaluation

Where the first prototype (September 2026) produced a result, a section starts from it and ends with the final system. All final results use the session-level split of Section 3.3, and the five test sessions {1, 3, 4, 13, 17} were used only as a post-hoc check, never for selection.

### 5.1 Feature-engineering ablation and model selection

The first prototype tested the hypothesis that a 582-dimensional scale-preserving feature vector (global statistics, patchwise scale statistics, first-order differences and a categorical encoding) would outperform a statistics-only baseline. Its early ablation compared three configurations with a Random Forest [@breiman2001rf] under a window-level split. The first prototype itself replaced that split with the session-level split before reporting its final results. Adjacent windows overlap by 75 %, and under the window-level split 31.8 % of test windows had a training window with a cosine similarity of at least 0.999 (Section 3.3). The values in {tab:ablation} therefore show how the configurations ranked under a leaky split. They are not performance estimates, and none of them is a final result.

Table: In the first prototype's feature-configuration ablation, Config A (statistics only) scored highest on both tasks (window-level split, Random Forest; superseded protocol, reported as history only). {#tab:ablation}

| Config | Dims | Binary F1 | Fault macro-F1 | Samples per feature |
|---|---|---|---|---|
| A: Statistics only | 64 | 0.981 | 0.945 | 15.4 |
| B: Statistics + patchwise scale | 320 | 0.975 | 0.894 | 3.1 |
| C: Full vector | 582 | 0.964 | 0.883 | 1.7 |

The Phase 1 report traced the lead of Config A to the curse of dimensionality. Under the window-level split, 988 anomalous training windows for 582 features give 1.7 samples per feature, below the minimum of 10 that the report took as necessary for reliable tree-based classification; Config A had 15.4. The Phase 1 report also described five follow-up investigations, including SMOTE [@chawla2002smote] and hyper-parameter tuning, that left Config A ahead. When the ML team (Farida El Ghabary and Raneem Alnaghy) later re-examined the first prototype's artefacts, it found that the notebooks behind those investigations had not been executed in their submitted state and that Config A had been chosen on the test set. The scores of those investigations are therefore not repeated here. Within the first prototype, the categorical block had already been rebuilt from six zero-filled columns into eight encoded columns, taking the full vector from 582 to 584 dimensions; no ablation result is reported for the 584-dimensional vector.

On the session-level split, the first prototype compared four tree ensembles for binary detection on Config A: Random Forest, XGBoost [@chen2016xgboost], LightGBM [@ke2017lightgbm] and CatBoost [@prokhorenkova2018catboost]. CatBoost ranked first of the four and was selected. The scores of that comparison are not repeated here, because the same report gives a natural-rate test F1 of 0.321 for the selected model (Section 5.2) without reconciling the two. CatBoost remained the algorithm of both models in the final cascade.

The ML team later repeated the algorithm question for Model 2 under the frozen protocol of Section 3.4 (25 session-grouped folds, Nadeau-Bengio corrected paired t-test [@nadeau2003inference]). Nine algorithms were compared on real data against the real-only baseline of 0.291 macro-F1; {tab:algos} lists the eight for which scores are reported. QUANT [@dempster2024quant], CatBoost with catch22 features [@lubba2019catch22] and LightGBM beat the baseline significantly. XGBoost, CatBoost on a 240-feature set, MiniRocket [@dempster2021minirocket], Hydra [@dempster2023hydra] and ExtraTrees [@geurts2006extratrees] did not. With synthetic data, LightGBM reached 0.412, but its gain over Farida El Ghabary's v3 recipe (Section 5.3) was not significant (p = 0.37), so under the pre-registered adoption rule the classifier did not change.

Table: In the Model 2 algorithm comparison under the final protocol (macro-F1 over 25 session-grouped folds, real data only), only QUANT, CatBoost + catch22 and LightGBM score significantly above the real-only baseline of 0.291 (CV: cross-validated). {#tab:algos}

| Algorithm | CV macro-F1 | Corrected p vs baseline |
|---|---|---|
| QUANT | 0.428 | 0.008 |
| CatBoost + catch22 | 0.402 | 0.021 |
| LightGBM | 0.391 | 0.020 |
| XGBoost | 0.358 | not significant |
| CatBoost (240 features) | 0.346 | not significant |
| MiniRocket | 0.328 | not significant |
| Hydra | 0.314 | not significant |
| ExtraTrees | 0.309 | not significant |

For RQ1 and RQ3 this stage gives only the window-level ranking. In the final cascade Model 1 keeps the 64 Config A statistics and Model 2 uses 496 features (Config B + FFT + G1 G2 G3; Section 4.3).

### 5.2 Detection: Model 1 from the first prototype to the final detector

**First prototype.** The first detector was evaluated on the 6,312 held-out windows (300 faults) at the natural test rate. Its confusion matrix was 4,907 true negatives, 1,105 false positives, 31 false negatives and 269 true positives: recall 0.897, precision 0.196 and F1 0.321 (0.32). It missed 31 of the 300 faults, but 1,105 of the 1,374 windows it flagged were Normal. The Phase 1 report attributed the low precision to recall-first tuning (an F2-score objective) and to class imbalance. Its headline F1 of 0.698 was measured on a subset of 1,000 test windows rebalanced to 80 % Normal and 20 % anomalous, and was compared with an F1 of 0.227 for an Isolation Forest baseline [@liu2008iforest]. Because the subset changes the class ratio, the natural-rate F1 of 0.32 is the first prototype's detection result used below.

**Two defects.** The ML team's re-examination found the two Model 1 defects described in Sections 3.5 and 3.6: the class imbalance corrected twice (80/20 rebalancing and `auto_class_weights="Balanced"`), and a parser that read `1.e-05` as 1 and −5, an impossible BLER, in 492 training windows (1.9 %). The two fixes were tested in a 2 × 2 design in cross-validation at a threshold of 0.5 ({tab:m1fix}; PR-AUC is the area under the precision-recall curve).

Table: In a 2 × 2 design of the Model 1 defect fixes (session-grouped cross-validation, threshold 0.5), the weighting fix accounts for almost all of the change. {#tab:m1fix}

| Variant | PR-AUC | Precision | Recall | False alarms per fold |
|---|---|---|---|---|
| First detector (both defects) | 0.440 | 0.365 | 0.530 | 1,175 |
| Fix parser | 0.442 | 0.375 | 0.532 | 1,171 |
| Fix weighting | 0.498 | 0.445 | 0.494 | 1,020 |
| **Fix both (adopted)** | **0.492** | **0.440** | **0.499** | **1,021** |

The parser fix alone moves PR-AUC only from 0.440 to 0.442. The variant with both fixes was adopted. False alarms fell in 25 of 25 folds, yet the corrected p-value was 0.24, because one fold (sessions 0, 6 and 12) dominates the variance. On the held-out sessions the effect is large ({fig:m1}): false alarms fell from 1,105 to 116, precision rose from 0.196 to 0.70 (266 of 382 alarms), recall moved from 0.897 to 0.887, and F1 rose from 0.32 to 0.78. The threshold stayed at 0.5 (Section 3.6).

![On the five held-out sessions the fixed detector cuts false alarms from 1,105 to 116 and raises precision from 0.20 to 0.70 and F1 from 0.32 to 0.78 compared with the first detector.](figures/fig_model1_heldout.png){#fig:m1}

**Why every detector tried fails on session 0.** In cross-validation, 90 % of the remaining false alarms come from the Normal windows of two training sessions, 0 and 5, whose KPI levels differ from those of the other training sessions. Session 0 is inverted: its Normal windows score about 0.99 and its real Jamming windows about 0.01. The fold holding sessions 0, 6 and 12 scores a PR-AUC of about 0.02 for every detector tried, level-based (CatBoost) and shape-based (QUANT) alike, which places the cause in the data. Session 0 mixes 3,007 Normal windows with 93 real Jamming windows. Its "Normal" windows are an idle link (median TX 44 B, PRB ≈ 0, UL_MCS 9), which is what the Jamming of session 2 looks like (TX 8 kB, PRB 1, UL_MCS 9). The Jamming of session 0, in contrast, is a loaded link (TX 438 kB). With two contradictory jamming recordings, no classifier can separate an idle link from a link blocked by jamming. Every Normal session can also be told apart from the other sessions without error (an area under the curve, AUC, of 1.0), so a prediction on a held-out session is always an extrapolation.

**Targeted alternatives.** Two hypotheses were then tested ({tab:m1alt}). For H1, that the detector relies on absolute levels, level-jitter augmentation duplicated each training window with its per-channel level shifted by a random offset drawn from the between-session spread of Normal medians, estimated inside the training fold. For H2, that shape features transfer across recordings, QUANT was used as the detector alone and averaged with CatBoost.

Table: No targeted Model 1 alternative beat the adopted detector significantly in cross-validation, and none had fewer false alarms on the five held-out sessions. TP = true positives; FP = false positives. {#tab:m1alt}

| Variant | CV PR-AUC | CV F1 | CV false alarms per fold | Vs adopted (corrected p) | Held-out PR-AUC · TP · FP |
|---|---|---|---|---|---|
| Adopted (both fixes) | 0.492 | 0.376 | 1,021 | — | 0.893 · 266 · **116** |
| Adopted + G (trend features) | 0.532 | 0.430 | 959 | +0.040 (p = 0.23, 20/25) | **0.917** · 270 · 123 |
| Level-jitter augmentation | **0.550** | **0.453** | **879** | +0.058 (p = 0.17, 22/25) | 0.618 · 215 · 805 |
| QUANT | 0.484 | 0.384 | 1,261 | −0.008 | 0.538 · 257 · 2,512 |
| QUANT + CatBoost | 0.527 | 0.450 | 1,035 | +0.035 (p = 0.36) | 0.886 · 267 · 124 |

Level jitter worked where it was aimed: in cross-validation, recall on the fault-only sessions 12 and 18 rose from 0.41 and 0.20 to 0.68 and 0.51. It also moved the decision boundary into the levels of the Normal-only test sessions 3 and 4, where it flagged 422 and 383 windows, so its gain did not transfer and it was rejected. QUANT alone could not separate a quiet Normal recording from a fault recording and flagged 2,117 of the 3,006 windows of session 4. Absolute levels are therefore necessary for detection, which agrees with an earlier level-free experiment by the ML team and with the scale ablation of the TelecomTS paper [@feng2025telecomts]. The trend features were the only addition better both in cross-validation and on the held-out sessions (PR-AUC 0.893 to 0.917, four more true positives and seven more false positives), but at p = 0.23 the gain, though promising, is not significant. Under the adoption rule, Model 1 remained the detector with both fixes. Session 0 kept a false-alarm rate of 94 to 98 % for every variant.

### 5.3 Fault classification: Model 2 and synthetic data

**First prototype.** The first classifier reached an accuracy of 0.677 on the test partition and a macro-F1 of 0.553 on the single train/test split. The Phase 1 report treated the single-split value as optimistic and gave as its primary figure a five-fold session-aware cross-validated macro-F1 of 0.333 ± 0.076 (95 % CI [0.267, 0.400]). With CCI-Mild and CCI-Severe merged and the two congestion classes merged, it was 0.354 ± 0.072. Regularisation lowered the single-split score (0.553 to 0.508), and SMOTE raised it (0.553 to 0.593) while lowering the cross-validated score (0.333 to 0.319); both were rejected. The report named the small anomalous sample (1,235 windows across 11 fault types) as the constraint and proposed synthetic augmentation.

**Synthetic data.** After the first prototype, Farida El Ghabary developed the v3 synthetic-fault method (Section 3.7); the first evaluation of the method found its gain consistent but not significant. Under the frozen protocol the ML team reproduced both reference models to four decimals (Sections 3.4 and 3.7). All Model 2 gains below are measured against this 25-fold baseline of 0.291, a different protocol from the Phase 1 report's five-fold estimate of 0.333. {tab:synth} lists the experiments that followed.

Table: Of the synthetic-data experiments for Model 2 (macro-F1 over 25 session-grouped folds), the v3 method was validated and only the trend features were adopted on top of it. {#tab:synth}

| Experiment | Result | Decision |
|---|---|---|
| v3 + Config B + FFT vs real-only | 0.374 vs 0.291, **p = 0.006, 25/25 folds** | Synthetic method validated |
| Exclude Congestion-Gradual and Buffer Overflow from augmentation | 0.377, higher variance (0.126) | No gain |
| Add trend features | **0.389**, p = 0.002 vs baseline, 24/25 | Adopted |
| Parser fix in the generator | 0.385 | Negligible; kept for correctness where possible |
| More synthetic windows: 500 → 1,000 → 2,000 per class | CatBoost 0.379 → 0.378 → 0.378 | Keep 500 |
| Synthetic data with QUANT | 0.428 (real) → 0.381 | Synthetic artefacts hurt raw-window models |

The synthetic method's gain was significant (p = 0.006) and appeared in all 25 folds, the trend features raised the score to 0.389, and more synthetic windows did not help. The ML team names three limits of the method, none of them an error. Every synthetic fault copies effects from the same 10 fault sessions, so no new session context is added. Synthetic faults sit on Normal-session backgrounds while real faults occur in fault-only sessions, so the cross-validation gain transfers only partly to the held-out sessions. The signal cache inherits the first prototype's parser, with a negligible effect on Model 2.

**Final candidates.** Six candidates were then measured both in cross-validation and on the 300 held-out fault windows ({tab:m2cand}).

Table: For the Model 2 final candidates, the ranking by cross-validated macro-F1 (25 session folds) differs from the ranking by held-out macro-F1 and accuracy (300 fault windows); the last column gives held-out accuracy on the three fault sessions. {#tab:m2cand}

| Candidate | CV macro-F1 | Test macro-F1 / accuracy | Test accuracy, sessions 1 / 13 / 17 |
|---|---|---|---|
| v3 recipe (CatBoost + v3 + Config B + FFT) | 0.374 | 0.537 / 0.663 | 1.00 / 0.25 / 0.64 |
| **+ trend features (final)** | **0.389** | **0.591 / 0.703** | 0.99 / 0.29 / 0.72 |
| QUANT, real only | 0.426 | 0.463 / 0.623 | 1.00 / 0.10 / 0.63 |
| CatBoost + catch22, real only | 0.393 | 0.608 / 0.730 | 1.00 / 0.29 / 0.77 |
| Soft vote, CatBoost + trend and QUANT | 0.453 (p = 0.08) | 0.516 / 0.660 | 1.00 / 0.06 / 0.73 |
| Soft vote of all three | 0.446 (p = 0.051) | 0.574 / 0.697 | 1.00 / 0.16 / 0.76 |

The final Model 2 is the v3 recipe with the trend features: 0.389 in cross-validation, and on the held-out sessions a macro-F1 of 0.591 and an accuracy of 0.703 on the 300 fault windows, against 0.537 and 0.663 for the recipe alone. {fig:m2} shows that the two rankings disagree. The soft votes have the highest cross-validated scores, but neither gain is significant (p = 0.08 and p = 0.051) and both fall behind the final model on the held-out sessions. CatBoost with catch22 has the best held-out scores (0.608 and 0.730) but is only 0.004 above the final model in cross-validation, and the held-out sessions were never used for selection. The QUANT and CatBoost + catch22 scores in {tab:m2cand} come from this later step and differ slightly from those in {tab:algos} (0.426 against 0.428, and 0.393 against 0.402).

![The Model 2 candidates that score highest in 25 session-grouped folds (blue) are not those that score highest on the 300 held-out fault windows (orange).](figures/fig_model2_candidates.png){#fig:m2}

**Per-class complementarity.** In pooled out-of-fold F1, CatBoost and QUANT fail on different faults. QUANT is better on Jamming (0.99 against 0.00 for CatBoost), Doppler Shift (0.93 against 0.77) and RF Filters (0.77 against 0.34); CatBoost is better on Buffer Overflow (0.61 against 0.36) and Congestion-Sudden (0.37 against 0.00). Jamming exists in only two training sessions, and QUANT's quantile and shape features transfer between them while CatBoost's level features do not. On the held-out sessions, where both Jamming sessions are part of the training data, the advantage of the vote disappears and session 13 pulls its accuracy down.

**Congestion-Gradual against Buffer Overflow.** On the held-out sessions, 30 of the 37 Congestion-Gradual windows are labelled Buffer Overflow. A two-class specialist built from the physics of the pair (buffer growth against growth in PRB utilisation, late-to-early ratios, time to peak) re-decided the windows labelled as either class. In 25 session folds the pair F1 fell from 0.298 to 0.247 and the macro-F1 from 0.389 to 0.378. Pooled Congestion-Gradual windows labelled Buffer Overflow went from 232 of 280 to 230 of 280 and stayed at 30 of 37 on the held-out sessions. The specialist was not adopted. Trained on the same 56 and 117 real windows from at most 10 sessions, it repeats the main model's confusion, because the signature that separates the two faults is not consistently present across recordings.

### 5.4 The cascade on the held-out sessions

The final cascade passes each window to the fixed Model 1 at a threshold of 0.5 and each alarm to Model 2 (the v3 recipe with the trend features). The first prototype had shown how the cascade fails when the detector raises too many alarms. Model 2 labelled 82.2 % of its 1,105 false alarms as CCI-Mild, and CCI-Mild scored a precision and an F1 of 0.00 on its true test examples despite about 126 training windows. Model 2 has no way to say "none of the above", so it gives a Normal window the class whose boundary lies closest to normal-looking telemetry.

**Reject option: a negative result.** Raneem Alnaghy's reject option adds a Normal class to Model 2, built from the 1,000 Normal training windows that the detector finds hardest. Under the frozen protocol these windows were chosen out-of-fold, so Model 2 never sees its own validation sessions. All 1,000 have an anomaly probability of at least 0.991; they are the idle-link windows of sessions 0 and 5 (Section 5.2). {tab:reject} compares the reject class with a random Normal class as a control.

Table: Neither reject-class variant of Model 2 meets the promotion rule in cross-validation (25 session folds) or in the cascade on the five held-out sessions. {#tab:reject}

| Model 2 variant | CV macro-F1 | Real faults rejected (CV) | Held-out false alarms in | False alarms rejected | Residual share labelled CCI-Mild | True faults rejected | Useful precision |
|---|---|---|---|---|---|---|---|
| First prototype cascade | — | — | 1,105 | — | 82.2 % | — | — |
| No reject class (final) | 0.389 ± 0.108 | 0 % | **116** | 0 | 64.7 % (75 windows) | 0 | 0.521 |
| Hard-negative Normal class | 0.369 ± 0.116 | 19.8 % | 116 | 16 | 68.0 % | 0 | 0.549 |
| Random Normal class (control) | 0.361 ± 0.090 | 17.8 % | 116 | 21 | 66.3 % | 3 | 0.545 |

Neither variant meets Raneem Alnaghy's promotion rule (a residual catch-all share below 41.1 % and a macro-F1 within two standard deviations). The reject class addresses the first prototype's problem: its 1,105 false alarms were dominated by idle-link windows, and the original version, with the hard negatives chosen in-sample, removed 83 % of them. After the Model 1 fixes only 116 false alarms remain, and they are a different population: Normal windows of test sessions 3 and 4 whose levels resemble faults. The residual is a Model 2 confusion with CCI-Mild, the fault class nearest to normal-looking windows. Because the argmax decision also rejected 17.8 to 19.8 % of real faults in cross-validation, a probability policy was tested that rejects a window only if P(Normal) ≥ τ, for τ from 0.5 to 0.9. Fault rejection in cross-validation fell from 13 % at τ = 0.5 to 4.8 % at τ = 0.9, but the number of held-out false alarms removed fell from 13 to 0 over the same range. No threshold removes false alarms without rejecting real faults, so the final pipeline has no reject class.

**Calibration: tested and rejected.** An isotonic regression [@zadrozny2002isotonic] fitted on out-of-fold Model 1 probabilities lowered the calibration error on the held-out sessions from 0.133 to 0.018, but mapped every raw score between 0.30 and 0.99 to the same value, 8.5 %. Of the 3,206 training windows scoring at least 0.9, 2,908 are Normal windows of sessions 0 and 5, so the map learned that a high score is usually a false alarm, which does not hold for other recordings.

**Layer 2 output, first against final.** {tab:layer2} compares the classifier output that the first prototype passed to Layer 3 with the final output over the 6,312 test windows.

Table: On the 6,312 test windows the final cascade labels 382 windows as faults instead of the first prototype's 1,374, for almost the same number of detected faults. {#tab:layer2}

| Layer 2 output | First prototype | Final cascade |
|---|---|---|
| Windows labelled as a fault | 1,374 (269 true + 1,105 false alarms) | **382 (266 true + 116 false alarms)** |
| Windows labelled Normal | 4,938 | 5,930 |
| Fault-type accuracy | 0.677 Model 2 accuracy; 0.647 cascade correctness rate as defined in the Phase 1 report | 0.748 on the 266 flagged faults; 199 of 300 true faults (0.663) correct end to end |
| False alarms labelled CCI-Mild | 82.2 % of 1,105 (≈ 908 windows) | 64.7 % of 116 (75 windows) |

The final cascade detects 266 true faults against the first prototype's 269. {fig:falsealarms} shows the false alarms and the part of them labelled CCI-Mild: the share falls from 82.2 % to 64.7 %, and the count from about 908 windows to 75.

![On the five held-out sessions the cascade's false alarms fall from 1,105, about 908 of them labelled CCI-Mild, in the first prototype's cascade (12 September 2026) to 116, 75 of them labelled CCI-Mild, in the final cascade.](figures/fig_false_alarms.png){#fig:falsealarms}

**Remaining errors.** On the 6,312 test windows the final cascade makes 116 false alarms, including 75 labelled CCI-Mild, 20 Faulty Handover, 8 Antenna Failure and 6 Doppler Shift. It misses 34 faults: all 11 Congestion-Sudden windows, 8 RF Filters, 6 Faulty Handover, 5 Congestion-Gradual, 3 Buffer Overflow and 1 CCI-Mild. The misses belong to the classes with a quiet onset, consistent with the session analysis of Section 5.2. For example, window 595, a Faulty Handover on a loaded link (TX 400 kB, UL_BLER 0.106), receives a Model 1 score of 0.27 and is missed.

**Family-level accuracy.** Because Congestion-Gradual, Buffer Overflow and Congestion-Sudden are confused with one another (Section 5.3), the validation report also gives an accuracy for one "congestion" family containing all three. At that level the final cascade is correct on 0.75 of the true faults, against 0.663 for the exact fault type end to end.

**Uncertainty.** With only 10 training sessions that contain faults, the ML team quantified uncertainty with session-level bootstrap confidence intervals and a power analysis. The gain of the final Model 2 over the real-only baseline is large, and its interval excludes zero: +0.098 macro-F1, 95 % CI [+0.078, +0.117], Cohen's d = 1.9. The gain of the trend features over the v3 recipe is small, +0.015 (95 % CI [+0.004, +0.025], d = 0.5). A paired test would need about 31 folds, two to three times the present number of fault sessions, to reach p < 0.05. The Model 1 fix (+0.052 PR-AUC, 95 % CI [+0.023, +0.085]) and the trend features for Model 1 (+0.040, 95 % CI [+0.020, +0.065]) lie in between. A bootstrap over overlapping seed-folds is optimistic, so all adoption decisions used the corrected t-test; for the Model 1 fix, that test gave p = 0.24 (Section 5.2).

**Idle-link review flag.** Because the ambiguity of session 0 lies in the recordings, the final system flags it for review. The idle-link indicator of Section 4.3 marks 50 windows on the held-out sessions, 45 of the 116 false alarms and 5 true faults: a review queue with 90 % precision on false alarms. The 94 Jamming windows of session 1 are on a loaded link and carry no idle-link note, and no label is changed.

### 5.5 Comparison with published baselines and existing approaches

Published TelecomTS results and the project's results are not all measured under the same split, and the split decides whether a comparison is meaningful. {tab:baselines} states for each figure whether it can be compared with the final detector.

Table: Of the TelecomTS results reported by the project and in the literature, only the first prototype's natural-rate F1 can be compared directly with the final detector; the table gives the evaluation protocol behind each figure. {#tab:baselines}

| System | Reported figure | Evaluation protocol | Comparable with the final detector? |
|---|---|---|---|
| KPI-RAG final detector (Model 1) | F1 0.78 (precision 0.70, recall 0.887) | Five held-out sessions, natural test rate | — |
| KPI-RAG first prototype | F1 0.32 | Same five held-out sessions, natural test rate | Yes |
| KPI-RAG first prototype, headline of the Phase 1 report | F1 0.698 | 1,000 test windows rebalanced to 80/20 | No: different class ratio |
| Isolation Forest (baseline in the Phase 1 report) | F1 0.227 | Not stated; compared in the Phase 1 report with the 0.698 headline | No |
| Mantis [@feofanov2025mantis], in the TelecomTS paper's benchmark [@feng2025telecomts] | F1 0.800 | TelecomTS paper's benchmark: random split of overlapping windows | No |
| Toto, in the TelecomTS paper's benchmark [@feng2025telecomts] | Accuracy 0.848 on fault-type classification (not a detection result) | Random split of overlapping windows | No |

Only the internal comparison is like for like: the first prototype and the final detector were evaluated on the same five held-out sessions at the natural test rate, so the change from F1 0.32 to 0.78 is a measured improvement. The headline of 0.698 in the Phase 1 report was measured on a rebalanced subset. That report does not state the evaluation set of the Isolation Forest baseline (0.227) it compared with the headline, and neither can be set beside the natural-rate results.

In the TelecomTS paper's own benchmark, Mantis [@feofanov2025mantis] reaches F1 0.800 on binary detection and Toto an accuracy of 0.848 on the fault-type task [@feng2025telecomts]. That benchmark uses a random split of overlapping windows, so its numbers are not comparable with session-level results, and the project did not reproduce them; the final detector's 0.78 is set beside 0.800 without a like-for-like claim. For the same reason the final Model 2 (held-out accuracy 0.703, macro-F1 0.591, cross-validated macro-F1 0.389) is not set against Toto's 0.848 and is compared only with the project's own baselines (Section 5.3).

The explanation layer has no published figure to compare with: none of the systems reviewed in {tab:related} produces a 3GPP citation that is checked automatically. The closest existing approach is ungrounded LLM generation, which condition C1 of Track C reproduces. With only the fault label and the SHAP features in the prompt, the system cites the correct specification for 5/30 windows, against 21/30 (70.0 %) when condition C3 adds the fault's row of the alignment table (matrix v1.2) and the per-window RCA evidence (Section 5.7). Condition C2 adds retrieved past incidents, the kind of context that RCACopilot retrieves ({tab:related}), and also scores 5/30, which is why the retrieval design is examined next.
