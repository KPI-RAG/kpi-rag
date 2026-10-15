KPI-RAG PHASE 2 (DATA & MODELS) — FINAL HANDOFF                                   2026-10-15
=====================================================================================

Everything is finished and synced (143 files on Drive, same on the server).
Server: kpi_rag/phase2/   =   Drive: Farida/Phase 2 - Data & Models/

1. THE FINAL CONFIGURATION (WHAT WE SHIP)
   - Model 1 (alarm): CatBoost on 64 basic statistics, with the two bugs fixed (correct
     number parsing, imbalance corrected once), threshold 0.5. Its confidence is now
     *calibrated*: a confidence of 0.9 means "right about 9 times in 10".
   - Model 2 (diagnosis): your recipe (CatBoost + your v3 synthetic data + Config B + FFT)
     plus the trend features. 11 fault classes, no reject class.
   - Output for the team: the same Layer 2 JSON format as Phase 1, with one addition inside
     an existing field: an "idle link, ambiguous" note on windows whose only fault evidence
     is "no traffic".

2. BASELINE vs FINAL (5 held-out recordings, 6,312 windows)
                                                  Phase 1          Phase 2
   False alarms (alarm raised, network normal)    1,105            116
   Real faults caught                             269 / 300        266 / 300
   Model 2 accuracy on the 300 real faults        0.677            0.703
   Windows sent to the explanation layer          1,374            382
     as "fault"                                   (1,105 wrong)    (116 wrong)
   Fault-family accuracy (congestion grouped)     -                0.75

3. THE IMPROVEMENTS THAT MATTERED, AND WHY
   - Fixing Model 1's double imbalance correction: this alone removed ~90% of the false alarms.
   - Your synthetic data: now statistically proven (p = 0.006, better in 25/25 folds).
   - Trend features: small but consistent gain for Model 2 in CV and on the held-out sessions.
   - Honest reporting: calibrated confidence, idle-link flag (45 of the 116 false alarms,
     5 true faults), and a family-level metric.

4. REJECTED, WITH EVIDENCE (so nobody repeats them)
   LightGBM / XGBoost / QUANT / MiniRocket / Hydra as replacements; more synthetic data
   (no effect); excluding the hard pair from augmentation; the reject option (removes 16 of
   116 false alarms, rejects ~20% of real faults); level-jitter augmentation and QUANT for
   Model 1 (good in CV, bad on unseen recordings); the pair specialist (no change: 30/37
   still confused).

5. REMAINING LIMITS (data, not code)
   - Session 0's idle link looks like one jamming recording.
   - HNC-Gradual vs Buffer Overflow confusion.
   - Only 10 fault recordings, so small gains can't reach significance (would need 2-3x
     more sessions).

6. FUTURE WORK
   docs/future_work.md (10 items, each with the evidence and what it would need).

7. FILES  (server kpi_rag/phase2/  =  Drive Farida/Phase 2 - Data & Models/)
   - results/final/: model1.cbm, model2.cbm, model1_calibrator.pkl, feature_spec.json,
     normal_means.json (0 NaN now), layer2_output_sessionsplit_phase2.json,
     validation.json (ok: true).
   - docs/phase2_results.md + .docx (report), PROJECT_STORY.md (beginner story),
     PROJECT_STATUS.md (A/B/C table), EXPERIMENT_LOG.md (19 experiments), DECISIONS.md,
     docs/future_work.md, and a NOTES.md in every results/ folder.

8. HOW TO EXPLAIN IT IN ONE MINUTE
   "Phase 1 raised 1,105 false alarms because the detector corrected the class imbalance
   twice. We fixed that and a number-parsing bug: false alarms fell to 116. We proved the
   synthetic-data method works, added features from the TelecomTS paper, and raised the
   diagnosis accuracy from 0.677 to 0.703. We tested nine algorithms and several tricks;
   we kept only what held up on recordings the models had never seen. What's left is
   limited by data: ten fault recordings and two contradictory jamming recordings - the
   report says exactly what new data would fix it."

For the beginner-level walkthrough of any step, start with PROJECT_STORY.md, then the
NOTES.md of that step.
