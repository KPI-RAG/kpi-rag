# Step 04 notes — why Model 1 fails on some sessions

- **Data structure (key finding):** of the 28 training sessions, **18 are 100% Normal,
  9 are 100% injected faults** (16, 18, 14, 32, 2, 15, 12, 11, 19), and only **session 0
  is mixed** (3007 Normal + 93 real Jamming). So Model 1 mostly has to tell
  *Normal sessions* from *fault sessions*; within-session contrast exists only in session 0.
- **Where the false alarms come from (OOF, t = 0.5, adopted Model 1):** 5455 in total,
  of which **2959 come from session 0** and **1986 from session 5**, i.e. ~90% from two
  sessions. These are whole Normal sessions whose operating levels differ from the
  training sessions (session 0: UL_BLER and UL_MCS levels shifted, robust z ≈ −7 / −5),
  **not** isolated traffic spikes.
- **Session 0 is inverted:** Normal windows get median p = 0.986, its Jamming windows
  p = 0.010. The model has learned Jamming from session 2 (Jamming-only) by absolute
  levels, which do not transfer to session 0.
- **Fault segment coverage (train faults):** 47% of fault windows are anomalous over the
  whole window (Jamming 100%, CCI-Mild 52%); the rest are partial (median 41–104 steps).
- **Implication for step 05:** prefer *level-free, within-window* features (segment shift,
  cross-KPI co-movement, shape), and test a Model 1 variant that does not use absolute levels.
  Per-session normalization is not possible (fault sessions have no Normal reference).
- **Files:** `per_session.csv`, `normal_shift_z.csv`, `fault_strength.csv`, `summary.json`, `log.txt`.
