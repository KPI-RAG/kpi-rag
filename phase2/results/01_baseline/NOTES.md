# Step 01 notes — baseline reproduction

- **Question:** does the Phase 2 code reproduce the recorded Model 2 baseline exactly?
- **Answer:** yes. All 25 fold scores match `final_confirm_records.csv` (Farida, 26-Sep)
  to 4 decimals: 0.2908 ± 0.0902. Integrity checks pass.
- **Files:** `manifest.json` (SHA-256 of all inputs + library versions),
  `baseline_folds.csv` (per fold), `log.txt`.
- **Reuse:** any later Model 2 experiment compares against `baseline_folds.csv` fold by fold.
- **Gotcha:** CatBoost `random_seed` must equal the CV seed, or the numbers drift.
