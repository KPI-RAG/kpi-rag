# Step 10b notes — why every detector fails on session 0 (model-agnostic cause found)

Step 10a: fold 4 (sessions 0, 6, 12) PR-AUC ≈ 0.02 for all five Model 1 variants, level-based
(CatBoost) and shape-based (QUANT) alike → the cause is in the data, not the model.

**T1 — recording fingerprint.** A 100-tree classifier separates any Normal session from the other
Normal sessions with AUC = 1.000 (sessions 0, 5, 6, 10 tested). Every recording has its own
signature; a held-out session is always an extrapolation.

**T2 — what session-0 "Normal" looks like.** Nearest class centroid (standardised Config A):
session 0 Normal → CCI-Mild 62 %, Jamming 25 %; session 5 Normal → CCI-Mild 60 %, Resource-Alloc
32 %; session 6 → Resource-Alloc 75 %. Sessions 10 and 20 → Normal 91–95 %. This is exactly the
false-alarm ranking of step 04 (0 > 5 > 6 > 10).

**T3/T4 — the Jamming contradiction (root cause).** Channel medians:

| | s0 "Normal" | s0 Jamming | other Normal | s2 Jamming |
|---|---|---|---|---|
| TX_Bytes | **44** | 438,154 | 163,696 | **8,092** |
| PRB_Util_DL | **0.006** | 60.4 | 32.8 | **1.1** |
| UL_MCS | **9.0** | 15.7 | 15.3 | **9.2** |
| UL_BLER | 0.000 | 0.092 | 0.091 | 0.001 |

- Session 0's "Normal" windows are an **idle link** (no traffic, no PRBs, UL_MCS 9).
- Session 2's Jamming windows (the only other Jamming source) look **the same**: blocked link,
  almost no traffic, UL_MCS 9. Centroid distance s0-Normal ↔ s2-Jamming = 3.1, vs 8.3 to the
  other Normal windows.
- Session 0's Jamming windows are the opposite: heavy traffic (TX 438 k, PRB 60 %), i.e. a
  loaded link under jamming. Jamming s0 vs s2 centroid distance 12.2 vs within-session spread 4.1.

So Model 1 learns "Jamming = idle/blocked link" from session 2, flags every idle Normal window of
session 0 (2,959 false alarms), and misses session 0's loaded-link Jamming. **No detector trained
on these sessions can resolve "idle" from "blocked by jamming" without context or more jamming
recordings.** Consequence for the pipeline: an "idle-link" guard (zero traffic + zero PRB) is a
policy question for the operator, not a classifier fix; recommended as future work with more
jamming sessions.

Files: `session0_channel_medians.csv`, `log.txt`.
