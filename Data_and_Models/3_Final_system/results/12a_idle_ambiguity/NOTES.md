# Step 12a notes — idle-link flag (adopted) and Model 1 calibration (REJECTED)

**Indicator** `idle_link` = TX_Bytes mean < 1,000 B/sample AND PRB_Util_DL mean < 1 % AND packets < 1/sample
(fixed physical thresholds). Coverage on the training partition: 48 % of session-0 "Normal" windows, 0.6 % of
other Normal windows; 13 % of fault windows (CCI-Severe 38 %, CCI-Mild 33 %, Jamming 9 %, congestion classes 0 %).

**Held-out sessions — the flag as a review queue (no label changes):**
| flagged windows | idle | of which true faults | of which false alarms |
|---|---|---|---|
| 382 | **50** | 5 (all CCI-Mild) | **45** |
| non-idle | 332 | 261 | 71 |
- 45 of the 116 false alarms (39 %) are idle-link windows → 90 % precision as a "needs operator confirmation"
  queue; it touches 5 true faults. Test session 1's 94 Jamming windows are loaded-link → no flag, label kept.
- Implementation (`src/pipeline.py`): `note: "idle_link_ambiguous"` on the `shap_top3` entries of flagged idle windows
  (the Phase 1 schema already has `note`). **Adopted.**

**Calibration — REJECTED (negative result, found by a teammate, confirmed 2026-10-16).**
Isotonic regression on the step-10 out-of-fold Model 1 probabilities (training only):
| | ECE raw | ECE calibrated |
|---|---|---|
| OOF (training partition) | 0.199 | 0.000 |
| held-out sessions (post-hoc) | 0.133 | 0.018 |
but the fitted map is **flat**: 15 raw scores between 0.30 and 0.99 → **1** distinct value (0.085); 0.20 → 0.080,
0.10 → 0.060, 1.00 → 1.000. Cause: 3,206 OOF windows score ≥ 0.9 and 2,908 of them (91 %) are Normal windows of the
Normal-only sessions 0 and 5 (true fault rate 7.5 %), so the map learned "a high score is usually a false alarm".
That holds for those recordings, not for others (held-out precision at t = 0.5 is 70 %); the low ECE was partly
an artefact of flattening (a constant output also has low ECE). It destroys the ranking information.
**Decision:** removed from the pipeline. `confidence` of Normal windows is the raw Model 1 probability (as in
Phase 1, which labelled it uncalibrated); the old file is kept only as `rejected_isotonic_calibrator.pkl` here.
Consequence for the JSON: no label or count changed (6,312 records, 382 fault labels, 116 false alarms); Normal-window
confidence went from 19 distinct calibrated values (0.915–0.983) to 469 raw values (0.500–0.992).
Lesson: a probability fitted across recordings is as unstable as the threshold (step 02). A proper approach would
need more recordings (future work #14). Files: `results.json`, `log.txt` (includes the flatness check).
