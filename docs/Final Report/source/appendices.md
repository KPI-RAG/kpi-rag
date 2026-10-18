## Appendices

### Appendix A: Causal mechanisms of the alignment table by fault group

Rodina Khallaf's analysis below gives the causal mechanism behind each of the 11 fault classes of the alignment table (matrix v1.2, {tab:alignment}), arranged in five functional groups; the numbers #1 to #11 are the row IDs of the table. The v1.2 mapping couples physical transmission phenomena and MAC/RLC protocol dynamics to standard KPI telemetry signatures across O-RAN functional splits.

#### Group 1: physical radio front-end and antenna failures (#1, #7)

**Antenna Failure (#1).** Physical antenna degradation or connector disconnection reduces effective isotropic radiated power (EIRP) and receiver sensitivity at the O-RU antenna port A (governed by 3GPP TS 38.141-1 §4.2.1 and §7.2 `PREFSENS` limits) [@ts38141]. The physical signal attenuation manifests as a sharp drop in Reference Signal Received Power (`RSRP`). To maintain the target Block Error Rate (`DL_BLER`), the gNB's link adaptation (AMC) algorithm drops the Downlink Modulation and Coding Scheme (`DL_MCS`) to robust QPSK constellations, leading to a severe throughput collapse.

**RF Filters (#7).** Thermal or mechanical degradation of receiver bandpass filters at the O-RU allows out-of-band emissions to bleed into active carrier bandwidths. Under TS 38.141-1 §7.5 (out-of-band blocking) and §7.6 (receiver spurious emissions), filter failure raises the receiver thermal noise floor. This gradually degrades the Uplink Signal-to-Noise Ratio (`UL_SNR`) and forces the MAC scheduler to assign extra uplink PRBs (`PRBs_UL_Current`) to compensate for retransmissions.

#### Group 2: MAC scheduler, protocol and queueing faults (#2, #10)

**Buffer Overflow (#2).** This fault occurs when the uplink packet arrival rate at the RLC layer exceeds the gNB scheduler's grant allocation rate. Under TS 38.321 §6.1.3.1 (BSR MAC Control Elements) and §5.4.5 (Buffer Status Reporting procedures) [@ts38321], the UE repeatedly issues Regular BSRs indicating high queue occupancy. Insufficient O-DU scheduling grants cause the RLC buffer to saturate, raising `PRB_Utilization_UL` to 100 % while triggering packet drops and latency spikes.

**Resource Allocation Bugs (#10).** Software defects in the O-DU MAC scheduler corrupt the token-bucket state variable $B_j$ during Logical Channel Prioritisation (LCP), governed by TS 38.321 §5.4.3.1.3. Instead of allocating resources based on the Prioritised Bit Rate (PBR) and LCG priority, the scheduler assigns excessive downlink PRBs (`PRBs_DL_Current`) to specific bearers while starving others, creating extreme variance in `PRBs_DL_Current_max` without proportional throughput increases.

#### Group 3: interference and over-the-air RF anomalies (#3, #4, #11)

**CCI-Mild (#3) and CCI-Severe (#4).** Uncoordinated neighbour cell transmissions on overlapping resource blocks elevate the co-channel interference power density. Under TS 38.141-1 §7.8 (in-channel selectivity) and TS 38.104 §4.9 (base station co-location requirements) [@ts38104], mild interference (#3) causes a moderate `UL_SNR` drop while RSRP remains unchanged. Severe interference (#4) completely swamps the receiver, causing `UL_SNR` collapse, `DL_BLER` spikes above decoding thresholds and unrecoverable packet loss at the O-RU.

**Jamming (#11).** Active over-the-air intentional RF jamming emits high-power wideband noise across active carrier frequencies. Unlike multi-path fading, jamming simultaneously suppresses `RSRP` and causes 100 % `DL_BLER`, collapsing the transmitted payload (`TX_Bytes`). As an external real-world testbed anomaly, it is evaluated against operational ticket logs rather than standardised 3GPP test clauses or O-RAN logical component targets.

#### Group 4: environmental and mobility channels (#5, #6)

**Doppler Shift (#5).** High UE velocity relative to the gNB induces a time-varying Doppler frequency shift $f_d$:

$$f_d = \frac{v}{\lambda} \cos\theta$$

- $f_d$: time-varying Doppler frequency shift ($\text{Hz}$).
- $v$: relative velocity of the User Equipment (UE) with respect to the gNodeB (gNB) ($\text{m/s}$).
- $\lambda$: carrier wavelength ($\text{m}$), defined as $\lambda = \frac{c}{f_c}$, where $c$ is the speed of light and $f_c$ is the carrier frequency.
- $\theta$: angle between the UE's velocity vector and the line-of-sight signal propagation path to the gNB antenna.

Modelled under 3GPP TR 38.901 §7.6.6 and §7.6.10 [@tr38901], rapid phase variation violates the channel coherence time, corrupting demodulation reference signal (DMRS) channel estimation at the O-RU/O-DU. The AMC scheduler reacts by driving conservative `DL_MCS` fallbacks and expanding uplink grant allocations (`UL_NPRB`).

**Faulty Handover (#6).** Sub-optimal mobility parameter configuration (for example a small `a3-Offset` or an insufficient `timeToTrigger`) in the Near-RT RIC xApp triggers rapid, cyclic ping-pong handovers between adjacent cells per TS 38.331 §5.5.4.4 (Event A3) and §5.5.4.1 [@ts38331]. This causes periodic `RSRP` fluctuations, control-plane signalling storms and transient throughput drops during RRC reconfiguration.

#### Group 5: traffic congestion dynamics (#8, #9)

**Congestion-Gradual (#8) and Congestion-Sudden (#9).** Both are monitored via the TS 28.552 §5.1.1.2.1 (`RRU.PrbDl`) and §5.1.1.2.2 (`RRU.PrbUl`) performance measurements at the O-DU [@ts28552]. Gradual congestion (#8) exhibits a steady upward trend in PRB demand, driving `PRB_Utilization_UL` towards 100 % while `UL_SNR` slowly degrades due to increased intra-cell inter-user interference. Sudden spike congestion (#9) represents an instantaneous burst of user arrivals, causing immediate PRB exhaustion, transient `DL_BLER` spikes and `UL_SNR` degradation within a single evaluation window.

### Appendix B: Model development log

After the first prototype the ML team (Farida El Ghabary, Raneem Alnaghy) ran the numbered development steps 01 to 12c in {tab:app_mllog}, each with its result and the decision taken. The session reconstruction, the session split and the reject option are Raneem Alnaghy's. The v3 synthetic data and the Model 2 recipe (CatBoost + v3 + Config B + FFT) are Farida El Ghabary's. "CV" is session-grouped cross-validation on the 28 training sessions (5 folds × 5 seeds = 25 folds) and "held-out" means the five test sessions. Model 2 scores are CV macro-F1 unless stated otherwise. The last four rows are experiments that the ML team reports without a step number.

Table: The two Model 1 fixes (step 03), the v3 synthetic data (step 06) and the trend features are the development steps that changed the final models; each row gives a step's result and the decision taken. {#tab:app_mllog}

| Step | What was done | Result | Decision |
|---|---|---|---|
| 01 | Froze the data and the protocol: StratifiedGroupKFold by session, 5 folds × 5 seeds; Nadeau-Bengio corrected paired t-test [@nadeau2003inference], with Wilcoxon as a check; a change is adopted only if corrected p < 0.05, the gain is material (≥ 0.03 macro-F1 or a material cascade gain) and it holds on the held-out sessions. Reproduced the baselines. | Model 2 real-only 0.2908 ± 0.0902; final v3 recipe of Farida El Ghabary (v3 + Config B + FFT) 0.3743 | Protocol used for every later decision; the held-out sessions used once per step as a post-hoc check, never for selection |
| 02 | Searched the Model 1 decision threshold per fold | Per-fold optimum ranges from 0.008 to 0.927 | Threshold stays 0.5; no single threshold transfers |
| 03 | Fixed the two Model 1 defects in a 2 × 2 design: the double imbalance correction (80/20 rebalancing and `auto_class_weights="Balanced"`) and the parser that reads 1.e-05 as two numbers (492 training windows, 1.9 %) | CV PR-AUC 0.440 (both defects), 0.442 (parser fixed), 0.498 (weighting fixed), 0.492 (both fixed); false alarms fall in 25/25 folds, corrected p = 0.24; held-out FP 1,105 → 116, F1 0.32 → 0.78, recall 0.897 → 0.887 | Both fixes adopted |
| 04 | Analysed the session structure of the training data | 18 training sessions are Normal-only, 9 are fault-only, and only session 0 mixes both (3,007 Normal + 93 real Jamming); every injected-fault session is 100 % faults | — |
| 05a | Level-free experiment (cited in the ML team's report only as "the level-free result") | Consistent with the later finding that absolute levels are necessary for detection | — |
| 06 | Compared Model 2 trained with v3 synthetic data (Config B + FFT) against real-only training | 0.374 vs 0.291, p = 0.006, 25/25 folds; the first prototype's parser, inherited by the signal cache, has a negligible effect on Model 2 | Synthetic method validated |
| 07 | Compared algorithms for Model 2 on real data only | QUANT 0.428 (p = 0.008), CatBoost + catch22 0.402 (p = 0.021) and LightGBM 0.391 (p = 0.020) beat the baseline significantly; XGBoost 0.358, CatBoost on a 240-feature set 0.346, MiniRocket 0.328, Hydra 0.314 and ExtraTrees 0.309 did not | See step 08 |
| 08 | Repeated the algorithm comparison with synthetic data | LightGBM 0.412, not significantly above the v3 recipe (p = 0.37) | No algorithm switch, by the pre-registered rule |
| 09a | Increased the synthetic volume from 500 to 1,000 to 2,000 windows per class | CatBoost 0.379 → 0.378 → 0.378 | Keep 500 |
| 09c | Measured the final candidates: CV macro-F1, then test macro-F1 / accuracy on the 300 fault windows | v3 recipe 0.374 (0.537 / 0.663); + trend features 0.389 (0.591 / 0.703); QUANT, real only 0.426 (0.463 / 0.623); CatBoost + catch22, real only 0.393 (0.608 / 0.730); soft vote CatBoost+trend & QUANT 0.453, p = 0.08 (0.516 / 0.660); soft vote of all three 0.446, p = 0.051 (0.574 / 0.697) | The v3 recipe + trend features is the final Model 2; on the held-out sessions the vote's advantage vanishes |
| 09z | Built a 25-fold cache of the synthetic data | Reproduces inline generation exactly (max difference 0.0); the final v3 model reproduced in 25/25 folds (0.3743) | — |
| 10 | Built the cascade (step-03 Model 1 at t = 0.5, then Model 2) and tested the reject option (Raneem Alnaghy): a "Normal" class for Model 2 from the 1,000 Normal training windows the detector finds hardest, chosen out of fold | CV: 0.389 ± 0.108 without a reject class, 0.369 ± 0.116 with hard negatives (19.8 % of real faults wrongly rejected), 0.361 ± 0.090 with random Normal windows (17.8 %). Held-out: hard negatives reject 16 of 116 false alarms, residual catch-all 68.0 % (64.7 %, 75 windows, without a reject class) | The promotion rule (residual catch-all < 41.1 % and macro-F1 within 2 SD) not met |
| 10a | Tested targeted Model 1 alternatives: level-jitter augmentation (H1), QUANT and a QUANT + CatBoost average (H2), and the step-03 detector with trend features (adopted + G) | Fold 4 (sessions 0, 6, 12) scores PR-AUC ≈ 0.02 for every detector; level jitter has the best CV PR-AUC (0.550) but 805 held-out false alarms; QUANT 2,512; adopted + G is better in CV and held-out (PR-AUC 0.893 → 0.917, +4 TP, +7 FP) but p = 0.23 | Model 1 stays the step-03 detector |
| 10b | Traced the session 0 false alarms back to the recordings | Session 0's Normal windows are an idle link (median TX 44 B, PRB ≈ 0, UL_MCS 9), which is what session 2's Jamming looks like (TX 8 kB, PRB 1, UL_MCS 9); session 0's own Jamming is a loaded link (TX 438 kB) | The cause is in the data: "idle" and "blocked by jamming" cannot be separated by any classifier |
| 10c | Tested a probability policy for the reject class (reject only if P(Normal) ≥ τ, τ = 0.5 to 0.9) | Fault rejection in CV falls from 13 % (τ = 0.5) to 4.8 % (τ = 0.9), while the residual false alarms removed on the held-out sessions fall from 13 to 0 | No threshold removes false alarms without rejecting real faults; the final pipeline has no reject class |
| 11 | Trained the final models on all 28 training sessions (`scripts/11_final_pipeline.py` + `src/pipeline.py`) and wrote the Layer 2 JSON ({tab:artefacts}) | 6,312 records, schema identical to the first prototype's, 0 invalid field values, reloading the saved models reproduces every prediction; 382 windows labelled as a fault (266 true + 116 false alarms) | Final Layer 2 output |
| 12a | Added an idle-link indicator (fixed thresholds: TX < 1 kB/sample, PRB < 1 %, < 1 packet/sample); tested an isotonic calibration of Model 1 [@zadrozny2002isotonic] | The indicator catches 45 of the 116 false alarms and also marks 5 true faults; no label changes. The calibration lowered the calibration error (0.133 → 0.018 on the held-out sessions) but mapped every raw score between 0.30 and 0.99 to the same value (8.5 %) | Indicator written as the note `idle_link_ambiguous`; calibration removed, so confidence is the raw model probability |
| 12b | Built a two-class specialist for Congestion-Gradual vs Buffer Overflow | Pair F1 0.298 → 0.247, macro-F1 0.389 → 0.378, Congestion-Gradual → Buffer Overflow 232/280 → 230/280 (pooled); held-out 30/37 → 30/37 | Not adopted; family-level accuracy (congestion family 0.75) reported instead |
| 12c | Session-level bootstrap confidence intervals and a power analysis | Final model over the real-only baseline +0.098 (95 % CI [+0.078, +0.117], Cohen's d = 1.9); trend features over the v3 recipe +0.015 (CI [+0.004, +0.025], d = 0.5), ≈ 31 folds needed to reach p < 0.05; Model 1 fix +0.052 PR-AUC (CI [+0.023, +0.085]); trend features for Model 1 +0.040 (CI [+0.020, +0.065]) | Bootstrap over overlapping seed-folds is optimistic; decisions used the corrected t-test |
| — | Excluded Congestion-Gradual and Buffer Overflow from augmentation | 0.377, variance up (0.126) | No gain |
| — | Added trend features to the synthetic-data model | 0.389, p = 0.002 vs baseline, 24/25 folds | Adopted |
| — | Fixed the parser in the synthetic-data generator | 0.385 | Negligible; kept for correctness in the final pipeline where possible |
| — | Trained QUANT with synthetic data | 0.428 (real only) → 0.381 | Synthetic artefacts hurt raw-window models |

The steps are run in order from the code folder `kpi_rag/phase2`. All randomness is seeded, `requirements.txt` pins the versions, and aeon and shap are installed in `~/.kpi_rag_vendor`.

```
cd kpi_rag/phase2
python scripts/01_freeze_data_and_reproduce_baseline.py   # ... through 11_final_pipeline.py
```

### Appendix C: Per-window Track C results

{tab:app_trackc} gives the outcome of every window in Ahmed Al Shobaki's final Track C run (`data/processed/track_c_v3/`; 30 windows, 3 per true fault type, seed 42) under the three prompt conditions. The totals reproduce the counts of {tab:trackc}: 5/30 under C1, 5/30 under C2 and 21/30 under C3.

Table: Under C3 every correctly classified window cites a valid standard, and all nine C3 failures are misclassified or missed windows. ✓ or ✗ marks a citation as valid or not for the true fault; numbers without a prefix are TS documents, and a dash marks a window the detector missed, for which no explanation was produced. {#tab:app_trackc}

| Window | True fault | Predicted | C1 | C2 | C3 |
| --- | --- | --- | --- | --- | --- |
| 597 | CCI-Mild | CCI-Mild | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 592 | CCI-Mild | CCI-Mild | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 645 | CCI-Mild | RF Filters | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 835 | Buffer Overflow | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 893 | Buffer Overflow | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 717 | Buffer Overflow | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 703 | CCI-Severe | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 607 | CCI-Severe | CCI-Severe | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 700 | CCI-Severe | CCI-Severe | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 864 | Antenna Failure | Antenna Failure | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 622 | Antenna Failure | Antenna Failure | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 687 | Antenna Failure | Antenna Failure | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 697 | RF Filters | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 610 | RF Filters | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 605 | RF Filters | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 682 | Congestion-Gradual | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 845 | Congestion-Gradual | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✗ 38.321 |
| 846 | Congestion-Gradual | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✗ 38.321 |
| 713 | Doppler Shift | Doppler Shift | ✗ 38.104 | ✗ 38.104 | ✓ TR 38.901 |
| 611 | Doppler Shift | Doppler Shift | ✗ 38.104 | ✗ 38.104 | ✓ TR 38.901 |
| 658 | Doppler Shift | Doppler Shift | ✗ 38.104 | ✗ 38.104 | ✓ TR 38.901 |
| 648 | Faulty Handover | RF Filters | ✗ 38.104 | ✗ 38.104 | ✗ 38.141-1 |
| 879 | Faulty Handover | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 715 | Faulty Handover | Faulty Handover | ✓ 38.331 | ✓ 38.331 | ✓ 38.331 |
| 684 | Resource Allocation Bugs | Resource Allocation Bugs | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 710 | Resource Allocation Bugs | Resource Allocation Bugs | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 686 | Resource Allocation Bugs | Resource Allocation Bugs | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 866 | Congestion-Sudden | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 827 | Congestion-Sudden | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 843 | Congestion-Sudden | Normal (missed) | ✗ — | ✗ — | ✗ — |
|  | Total correct |  | 5 / 30 | 5 / 30 | 21 / 30 |

### Appendix D: One window explained under the three conditions

The explanation layer (Ahmed Al Shobaki) produced the explanations below for test window 611, whose true and predicted fault is Doppler Shift, under C1, C2 and C3 in the final Track C run (`track_c_v3/track_c_explanations.jsonl`).

**C1: label + SHAP; cites TS 38.104 (incorrect)**

> "Severe Doppler shift is caused by rapid user equipment movement resulting in abnormally high RSRP combined with degraded downlink modulation and coding scheme (DL_MCS: below normal)."

O-RAN component: Near-RT RIC

**C2: + past incidents; cites TS 38.104 (incorrect)**

> "Severe Doppler shift is induced by high-velocity user equipment mobility causing rapid radio channel fluctuations, evidenced by fluctuating RSRP levels above normal combined with suppressed downlink modulation and coding (DL_MCS below normal)."

O-RAN component: Near-RT RIC

**C3: + alignment row + evidence; cites TR 38.901 (correct)**

> "Severe Doppler shift caused by high user equipment velocity resulting in rapid channel phase variation, evidenced by high RSRP variance (std |SHAP|=0.654, fft energy |SHAP|=0.510) and degraded downlink modulation (DL_MCS mean=6, |SHAP|=0.484)."

Recommended action: "Enable high-speed mobility enhancements and adjust channel estimation filter lengths at the O-DU/O-RU to compensate for rapid phase rotation." O-RAN component: O-RU / O-DU

C1 and C2 describe the same pattern and cite the same specification. C2's reasoning trace adds RSRP values from the similar incidents ("-125 dB to -107.2 dB across similar historical traces") without changing its citation. C3 cites the channel-model report that governs Doppler [@tr38901] and quotes the feature-level evidence from the evidence block.

### Appendix E: Prompt template

The explanation layer (`src/llm_explainer.py`, Ahmed Al Shobaki) builds every prompt from the template below. All three conditions share the header and the instruction block; C2 and C3 add the retrieved incidents, and C3 also adds the standards reference and the per-window evidence block (`[EVIDENCE FROM RCA PIPELINE]`).

Header, used by all conditions:

```
You are a 5G network fault diagnosis expert.

Fault detected: {fault type}
Confidence: {confidence:.0%}

Top contributing KPIs (SHAP):
  {channel}: {above|below} normal (|SHAP|={value:.2f}, {toward|away from} predicted class)
  ... (three lines)
```

Evidence block, added by C3 directly after the header (`src/rca_loader.py`):

```
[EVIDENCE FROM RCA PIPELINE]
[RCA PIPELINE EVIDENCE — window {window index}]
Predicted fault: {fault type} (confidence: {confidence:.1%})
Model attribution (SHAP, top 3 features):
  - {feature}: |SHAP|={value:.3f} ({above|below} normal, {effect})
  ... (three lines)
Observed KPI means (window):
  - {channel}: mean={value}
Causal mechanism: {first 120 characters of the record's causal mechanism}...
```

Blocks added by C2 and C3:

```
Retrieved similar incidents:
  [1] (similarity 0.72) {first 300 characters of the incident's KPI narrative}...
  [2] ...
  [3] ...

[STANDARDS REFERENCE]                        (C3 only)
3GPP {standard} clause {clause}: {evidence text}
O-RAN component: {component}
```

Instruction block, identical in all conditions:

```
Ground your explanation in the specific KPI values and SHAP evidence provided above.

Return ONLY a JSON object with exactly these fields:
{
  "root_cause": "one sentence physical explanation referencing specific KPI values",
  "3gpp_reference": "TS XX.XXX or TR XX.XXX",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain citing specific KPI values from the evidence above"
}
```

### Appendix F: Evaluation rubrics and scoring rules

This appendix gives the Track B dimensions and scale as Rodina Khallaf defines them ({tab:app_geval}) and the Track C scoring rules of Ahmed Al Shobaki.

#### Track B: G-Eval dimensions and scale

Track B uses an LLM-as-judge protocol based on the G-Eval framework [@liu2023geval]. Each explanation in its 30-window sample (3 windows per fault type, 10 fault types) is scored against a structured ground-truth reference on four dimensions.

Table: Track B (G-Eval) scores each explanation on four dimensions, each defined by the question the judge answers (Rodina Khallaf). {#tab:app_geval}

| Dimension | Question the judge answers |
| --- | --- |
| Citation validity | Does the explanation cite a real, applicable 3GPP Technical Specification? Is the cited standard the correct one for the described fault mechanism? |
| Fault specificity | Does the explanation correctly identify the fault family and primary mechanism, linking root cause to the specific KPIs flagged by the SHAP analysis? |
| Actionability | Does the recommended remediation action name the correct O-RAN component and provide sufficient technical specificity for an engineer to act on it? |
| Causal soundness | Is the cause → symptom → KPI chain logically complete and technically accurate, consistent with 5G NR physical-layer and protocol-layer behaviour? |

- **Scale.** Each dimension is scored from 1 to 5 on an integer scale, using a five-point rubric calibrated against expert-written ground-truth explanations.
- **Composite.** The composite score is the unweighted mean of the four dimensions.
- **Judge-rated citation validity rate.** It is the proportion of windows in which the cited Technical Specification is correct, regardless of clause precision.

#### Track C: scoring rules

- **Sample.** Three test windows are drawn for each of the 10 fault types with a 3GPP reference (seed 42), sampled by true fault type, not by predicted label. Each window goes through the full system under C1, C2 and C3.
- **Correct (against the true row).** A citation counts as correct if it is valid for the window's true fault. A misclassification that brings in the wrong row therefore counts as wrong.
- **Missed windows.** A window that the detector missed has no explanation and counts as incorrect.
- **Valid (against the predicted row).** Validity is also recorded against the row of the fault the classifier named, over the windows that have an explanation. It measures how closely the LLM follows the row it was given.
- **Validation.** The parser normalises the first citation it finds ("TS38.331", "ts 38.331" and "3GPP TS 38.331 §5.5.4.4" all become `TS 38.331`). The reference must match the TS/TR format and appear in the list of valid references for the fault. Jamming, the only fault type with no 3GPP reference, can never have a valid citation, and a template answer produced after failed LLM calls is always invalid. Clause numbers are not checked.
- **Two rules that need no LLM.** Both are scored on the same windows: always citing the single standard that is correct most often (TS 38.141-1 in this sample), and citing the predicted fault's row.
- **Statistics.** Wilson 95 % intervals [@wilson1927] and exact McNemar tests [@mcnemar1947] describe effect sizes; on 30 windows per condition they are not significance tests.

### Appendix G: TeleQnA confidence-gate threshold sweep

{tab:app_sweep} gives the full confidence-gate sweep of Rodina Khallaf's Track D, followed by the worked example of context poisoning from the Track D report. The gate is applied post hoc to the stored answers of the 600 TeleQnA questions [@maatouk2025teleqna], with no additional LLM inference. The KPI-RAG answer is used only when the maximum retrieval score reaches the threshold τ, and the baseline answer is used otherwise. From τ = 0.45 upwards the gated accuracy is within 1.2 pp of the baseline, and at τ = 0.55 it matches the baseline exactly, as only one question clears that threshold.

Table: As the confidence-gate threshold τ rises, fewer of the 600 TeleQnA questions use retrieved context and the gated accuracy approaches the 76.3 % baseline (Track D). Coverage is rounded to one decimal, so 13 of 600 questions (2.17 %) appear as 2.2 %. {#tab:app_sweep}

| Threshold τ | Questions using RAG | Coverage | Gated accuracy | Δ vs baseline |
| --- | --- | --- | --- | --- |
| 0.20 | 496 | 82.7 % | 57.3 % | −19.0 pp |
| 0.25 | 399 | 66.5 % | 62.2 % | −14.2 pp |
| 0.30 | 294 | 49.0 % | 66.7 % | −9.7 pp |
| 0.35 | 193 | 32.2 % | 69.7 % | −6.7 pp |
| 0.40 | 101 | 16.8 % | 73.8 % | −2.5 pp |
| 0.45 | 33 | 5.5 % | 75.2 % | −1.2 pp |
| 0.50 | 13 | 2.2 % | 76.0 % | −0.3 pp |
| 0.55 | 1 | 0.2 % | 76.3 % | +0.0 pp |
| 0.60 | 0 | 0.0 % | 76.3 % | +0.0 pp |

In Rodina Khallaf's manual inspection, the retrieved chunks contain operationally plausible telemetry language that is semantically unrelated to the question. Question q00021 (TS 38.211 [@ts38211]) is one such case:

> **Q (q00021):** "What is used to initialize the scrambling sequence generator for UE-specific search space in PDCCH?"
>
> **Expected:** `pdcch-DMRS-ScramblingID` (3GPP TS 38.211)
>
> **Retrieved chunk (score = 0.217):** "Issue: Severe communication degradation due to external RF jamming detected on sector 4B. Interference power: −72 dBm..."
>
> **Baseline** → correct (option 4). **KPI-RAG** → wrong (option 1: C-RNTI).

### Appendix H: Configuration

The explanation layer reads its parameters from `configs/config.yaml` (Ahmed Al Shobaki); the excerpt below shows the retrieval, LLM, data and evaluation settings.

```
rag:
  embedding_model: "all-MiniLM-L6-v2"
  collection_name: "telecomts_descriptions"   # per-window KPI narratives, cosine
  top_k: 5
  cosine_threshold: 0.35
llm:
  backend: "gemini"                           # gemini | groq | ollama
  gemini_model: "gemini-3.5-flash-lite"
  temperature: 0.1
  max_retries: 3
  min_request_interval_s: 2.0
  backoff_base_s: 5
  backoff_max_s: 60
data:
  train_idx_file: "train_idx_sessionsplit.npy"
  test_idx_file: "test_idx_sessionsplit.npy"
  layer2_output_file: "layer2_output_sessionsplit_phase2.json"
  alignment_table_path: "data/alignment_table.json"
  rca_evidence_path: "data/processed/rca_evidence.json"
  random_state: 42
evaluation:
  citation_validity_threshold: 0.70
  samples_per_fault: 3
```

The final Layer 2 artefacts of the ML team are listed in {tab:artefacts} (Section 4.3).

### Appendix I: Reproduction and user instructions

The commands below, as documented by Ahmed Al Shobaki, install and test the system, reproduce the explanation-layer results and start the dashboard.

**Install and test.**

```
git clone -b master https://github.com/KPI-RAG/kpi-rag.git
cd kpi-rag
uv sync                      # CPU-only torch
cp .env.example .env         # add GEMINI_API_KEY
uv run pytest tests/ -q      # 207 passed, 3 skipped
```

**Reproduce the reported numbers.**

```
uv run python scripts/run_eval_track_c.py --output data/processed/my_run   # 72 LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/track_c_v3 --summarise-only
uv run python scripts/experiment_retrieval_design.py                       # needs data/raw (TelecomTS)
uv run python scripts/build_index.py --rebuild                             # rebuild the index from data/raw
```

The Track C script refuses to overwrite an earlier run without `--force`, and `--summarise-only` recomputes the summary from the saved explanations without new LLM calls.

**Use the dashboard.**

```
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

1. Choose an input in the sidebar: *Use example*, *Browse Dataset Windows* (the 382 test alarms; filter by fault type, then pick a window) or *Upload JSON*.
2. Click **Generate Explanation**.
3. Read the diagnosis and its citation badge, then check the retrieved incidents and the evidence panel before acting.

### Appendix J: Test inventory

{tab:app_tests} lists the modules of the project's automated test suite, built by Ahmed Al Shobaki, with the number of tests in each. The suite has 210 tests in 29 modules (21 unit and 8 integration modules); 207 pass and the 3 skipped tests need a Groq API key.

Table: Each automated test module is listed with its number of tests; 207 of the 210 tests pass and 3 are skipped. {#tab:app_tests}

| Module | Tests | Module | Tests |
| --- | --- | --- | --- |
| test_review3_fixes | 26 | test_schema | 10 |
| test_phase2_models | 23 | test_evaluator | 10 |
| test_llm_explainer | 23 | test_data_loader | 6 |
| test_phase2_pipeline | 18 | test_run_eval_track_c | 6 |
| test_phase2_data | 12 | test_run_pipeline | 5 |
| test_rag_query | 5 | test_run_eval_track_b | 4 |
| test_kg_indexer | 4 | test_config_loader | 3 |
| test_build_index | 3 | test_explanation_panel | 3 |
| test_run_eval_track_a | 2 | detection / KPI / sources panels; SHAP panel | 2 each; 3 |
| integration: schema, RAG query, ChromaDB, alignment | 3 · 4 · 6 · 6 | integration: explainer, evaluator, pipeline, live RAG | 6 · 4 · 4 · 5 |
| Total | 210 | passing / skipped (Groq key) | 207 / 3 |
