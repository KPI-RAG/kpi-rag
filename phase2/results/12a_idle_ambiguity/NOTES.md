# Step 12a notes — idle-link ambiguity: flag it, calibrate it, never suppress it

**Indicator** `idle_link` = TX_Bytes mean < 1,000 B/sample AND PRB_Util_DL mean < 1 % AND packets < 1/sample
(fixed physical thresholds). Coverage on the training partition: 48 % of session-0 "Normal" windows, 0.6 % of
other Normal windows; 13 % of fault windows (CCI-Severe 38 %, CCI-Mild 33 %, Jamming 9 %, congestion classes 0 %).

**Calibration** (isotonic regression fitted on the step-10 out-of-fold Model 1 probabilities, training only):
| | ECE raw | ECE calibrated |
|---|---|---|
| OOF (training partition) | 0.199 | 0.000 |
| held-out sessions (post-hoc) | 0.133 | 0.018 |
Idle Normal windows of session 0: mean probability 0.97 → **0.09** after calibration. The decision still uses
the raw probability at t = 0.5 (validated), only the reported `confidence` is calibrated.

**Held-out sessions — the flag as a review queue (no label changes):**
| flagged windows | idle | of which true faults | of which false alarms |
|---|---|---|---|
| 382 | **50** | 5 (all CCI-Mild) | **45** |
| non-idle | 332 | 261 | 71 |
- 45 of the 116 false alarms (39 %) are idle-link windows → the flag has 90 % precision as a "needs operator
  confirmation" queue; it touches 5 true faults.
- Test session 1's 94 Jamming windows are **not** idle (loaded link) → they keep their label and no flag.
- Per session: s3 21/44 flagged windows idle, s4 24/72, s13 5/54, s1 and s17 none.

**Implementation** (`src/pipeline.py`, `results/final/`): `note: "idle_link_ambiguous"` on the `shap_top3`
entries of flagged idle windows (Phase 1 schema already has the `note` field); `confidence` of Normal windows =
calibrated probability; `model1_calibrator.pkl` saved next to the models. Files: `results.json`, `log.txt`.
