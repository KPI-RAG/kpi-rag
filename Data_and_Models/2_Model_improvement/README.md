# Model improvement (between Phase 1 and the final system)

Three experiments run after the first prototype:

- `07_false_alarm_filter`: gives Model 2 a "Normal" class so it can send false alarms back.
  Tested again in the final system (step 10) and not kept.
- `08_fft_features`: frequency-domain features for Model 2. Kept in the final model.
- `09_synthetic_data`: synthetic fault windows built by transplanting real fault effects into
  normal windows. Versions 1 and 2 did not work; version 3 is used in the final model.
