# Step 03 notes — Model 1 bug fixes (ADOPTED)

- **Design:** 2 x 2 (parser fix x weighting fix), 5 seeds x 5 grouped folds, primary metric PR-AUC.
- **Result:** fixing the weighting gives almost all the gain: false alarms −13% (better in
  25/25 folds), PR-AUC +0.05 (better in 18–21/25), recall −0.03. The parser fix alone ≈ 0.
- **Statistics:** corrected p ≈ 0.24, so it does not meet the pre-registered p < 0.05 rule.
  Fold 4 (sessions 0, 6, 12) has PR-AUC ≈ 0.02 for every variant, which inflates variance.
- **Post-hoc test:** FP 1105 → 116, F1 0.321 → 0.780, recall 0.897 → 0.887.
- **Decision (2026-10-13, Farida):** adopt `fix_both`, because both changes are bug
  fixes rather than tuned choices. Encoded in `src/config.py`
  (`MODEL1_FIXED_FEATURES`, `MODEL1_CLASS_WEIGHTING`).
- **Next:** step 04 investigates why sessions 0/6/12 fail.
