# Step 02 notes — Model 1 threshold

- **Question:** is 0.5 a good threshold for Model 1, chosen without the test set?
- **Answer:** keep 0.5. The F2-optimal threshold per fold ranges from 0.008 to 0.927, so
  no single threshold transfers across sessions. The pre-registered rule (t = 0.061)
  roughly doubles false alarms.
- **Validation:** the rebuilt Model 1 reproduces the Phase 1 test confusion matrix
  exactly (TP 269 / FP 1105 / FN 31 / TN 4907; 82.2% catch-all).
- **Lessons (do not repeat):**
  - Phase 1 CatBoost used `auto_class_weights="Balanced"`. Leaving it out gives
    FP 84 instead of 1105. Always copy `make_base_clf()` exactly when reproducing.
  - Test features must be rebuilt with the Phase 1 parser replica
    (`config_a_phase1_parser`), not from `raw_kpis.npy`, for a like-for-like check.
  - Relative-error checks blow up on near-zero features; use `np.allclose` with `atol`.
