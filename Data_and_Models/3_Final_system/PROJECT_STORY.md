# The KPI-RAG project story (Data & Models), in simple words

## 1. What problem are we solving?
A 5G network reports numbers every 100 ms (signal strength, errors, traffic, …). When something breaks — an
antenna fault, interference, congestion, jamming — engineers must look at those numbers, find the fault, and
check the 3GPP standards to explain it. That takes time. KPI-RAG does the first part automatically: it reads a
12.8-second window of 16 numbers, decides if something is wrong, names the fault, and tells the explanation
layer (Rodina/Ahmed) which KPIs mattered.

## 2. What does Model 1 do?
Model 1 is the **alarm**. For every window it answers one question: normal or fault? It must catch faults
(recall) without crying wolf (false alarms = alarm raised although the network is normal).

## 3. What does Model 2 do?
Model 2 is the **diagnosis**. Only for windows Model 1 flagged, it names which of 11 faults it is
(Antenna Failure, Jamming, Buffer Overflow, …).

## 4. Why do we use synthetic data?
We only have 935 real fault windows from 10 recordings — too few to learn 11 faults well. The synthetic-data method
(v3) takes the measured effect of a real fault and transplants it onto a real normal window, giving ~4,400
extra, realistic fault windows per training round. Important: synthetic windows are used only for training,
never for scoring, and never from the held-out recordings.

## 5. What was wrong with Phase 1?
- Model 1 corrected the class imbalance **twice** (rebalancing + class weights), so it over-flagged:
  **1,105 false alarms** on the test recordings.
- A number-reading bug turned `1.e-05` into `1` and `-5`, giving impossible error rates (BLER = −5) in 2 % of windows.
- Some report numbers came from notebooks that had never been run; one feature choice was made on the test set.
- 82 % of the false alarms were given one fault name (CCI-Mild), which polluted the explanation layer.

## 6. What did we change in Phase 2?
1. Re-built everything as clean scripts that reproduce the old numbers exactly (so we can trust comparisons).
2. Fixed the two Model 1 bugs.
3. Added "trend" features (how KPIs grow over time, how many move together) from the TelecomTS paper.
4. Proved the v3 synthetic-data method statistically; kept the synthetic-data model as the final Model 2.
5. Compared 9 algorithms, more synthetic data, a false-alarm filter, several Model 1 fixes — kept only what
   held up on unseen recordings.
6. Added an "idle link" review flag and a family-level metric. (A probability calibration was tried and rejected: it flattened every score to 8.5 %.)
7. Produced the final models and the Layer 2 JSON in the same format as Phase 1.

## 7. Which results improved, and by how much? (5 held-out recordings, 6,312 windows)
| | Phase 1 | Phase 2 |
|---|---|---|
| Model 1 false alarms | 1,105 | **116** |
| Model 1 faults caught | 269 / 300 | 266 / 300 |
| Model 2 accuracy on the 300 true faults | 0.677 | **0.703** |
| Windows handed to the explanation layer as "fault" | 1,374 (1,105 wrong) | **382 (116 wrong)** |

## 8. What problems remain and why?
- **Session 0**: its "normal" windows are an idle link, which looks exactly like one jamming recording.
  No model can tell "idle" from "blocked by jamming" with two contradictory recordings. We flag such windows
  as ambiguous instead of guessing.
- **HNC-Gradual vs Buffer Overflow**: two congestion faults that look alike; 30 of 37 test windows of one are
  named as the other. Too few recordings to learn the difference.
- **Only 10 fault recordings**: improvements over the synthetic-data model are consistent but small; the statistics
  say we'd need 2–3× more recordings to prove them.

## 9. Why did we choose the final configuration?
Because it is the best that **held up on unseen recordings**: every alternative that scored higher in
cross-validation (LightGBM, QUANT, ensembles, jitter augmentation) did worse or no better on the held-out
sessions. Simpler and proven beats complex and unproven.

## 10. Which files should the team use?
`results/final/`: `model1.cbm`, `model2.cbm`, `feature_spec.json`,
`normal_means.json`, `layer2_output_sessionsplit_phase2.json` (for Rodina/Ahmed), `validation.json`.
Report: `docs/phase2_results.md` / `.docx`. Status: `PROJECT_STATUS.md`. All experiments: `EXPERIMENT_LOG.md`.

## 11. What next, with more time or data?
See `docs/future_work.md`: more jamming recordings (with and without traffic), recordings where normal and
fault windows co-occur, 10–20 more fault sessions; then the specialist classifier and trend features for
Model 1 can be proven.
