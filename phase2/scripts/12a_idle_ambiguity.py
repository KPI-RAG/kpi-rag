"""Step 12a - Idle-link ambiguity: say "uncertain" instead of guessing confidently.

Facts (10b): idle windows (no traffic, no PRBs) are labelled Normal in session 0 and Jamming in session 2
with healthy radio KPIs in both -> no classifier or rule can separate "idle" from "blocked by jamming".
A rule "idle => Normal" would delete the 94 Jamming windows of test session 1, so instead:
  1. idle_link indicator per window: TX_Bytes mean < TX_T, PRB_Utilization_DL mean < PRB_T,
     (UL+DL packets) mean < PKT_T. Thresholds are fixed physical constants (not fitted on labels);
     their coverage is reported per session on the training partition.
  2. Model 1 probability calibration (isotonic) fitted on the step-10 OUT-OF-FOLD probabilities of the
     training partition; ECE before/after plus a FLATNESS CHECK. RESULT (2026-10-16): REJECTED - the isotonic
     map is flat (8.5 %) for every raw score between 0.30 and 0.99, so it is not used (kept only as a negative result).
  3. Measurement on the 5 test sessions: how many flagged windows are idle, how many of them are true
     faults vs false alarms (= the value of the flag as an operator-review queue), and whether the
     flag changes any label (it must not).
The pipeline (src/pipeline.py) then writes note "idle_link_ambiguous" into shap_top3 of flagged idle windows
and keeps the RAW Model 1 probability as the Normal-window confidence. Schema unchanged.

Run from the phase2 folder:   python scripts/12a_idle_ambiguity.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src import config as C
from src import pipeline as P
from src.data import csv_labels, features, load_splits
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog

log = StepLog("12a_idle_ambiguity")
d = load_splits(); labels = csv_labels()
y = (d["train_y"] != C.NORMAL_LABEL).astype(int); sess = d["train_sessions"]
A_tr, A_te = features(d, "train", True), features(d, "test", True)
y_te = (labels[d["test_idx"]] != C.NORMAL_LABEL).astype(int); te_sess = d["test_sessions"]
y_te_str = labels[d["test_idx"]]

# 1. idle indicator ------------------------------------------------------------------------------------
idle_tr, idle_te = P.idle_link(A_tr), P.idle_link(A_te)
log(f"== 1. idle_link indicator (TX_Bytes mean < {P.IDLE_TX} B, PRB_Util_DL mean < {P.IDLE_PRB} %, packets < {P.IDLE_PKT}/sample) ==")
tab = pd.DataFrame({"session": sess, "idle": idle_tr, "fault": y, "label": d["train_y"]})
per = tab.groupby("session").agg(windows=("idle", "size"), idle_share=("idle", "mean"), faults=("fault", "sum"))
log("  training sessions with idle share > 5 %:")
log(per[per.idle_share > 0.05].round(3).to_string())
log(f"  idle share: Normal windows {tab[tab.fault == 0].idle.mean():.1%}  (session 0: {tab[(tab.session == 0) & (tab.fault == 0)].idle.mean():.1%}, "
    f"other Normal: {tab[(tab.session != 0) & (tab.fault == 0)].idle.mean():.1%}) | fault windows {tab[tab.fault == 1].idle.mean():.1%}")
log("  idle share by fault type:\n" + tab[tab.fault == 1].groupby("label").idle.mean().round(3).sort_values(ascending=False).to_string())

# 2. calibration -----------------------------------------------------------------------------------------
log("\n== 2. Model 1 calibration (isotonic on step-10 OOF probabilities) ==")
oof = np.load(os.path.join(C.RESULTS, "10_cascade_reject", "oof_model1_prob.npy"))
cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)


def ece(p, yy, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - yy[m].mean())
    return e


keep = rebalance_80_20(np.arange(len(y)), y)
m1 = anomaly_detector(C.SEED, class_weighting=C.MODEL1_CLASS_WEIGHTING).fit(A_tr[keep], y[keep])
p_te = m1.predict_proba(A_te)[:, 1]
log(f"  ECE on OOF (train):  raw {ece(oof, y):.3f} -> calibrated {ece(cal.predict(oof), y):.3f}")
log(f"  ECE on test (post-hoc): raw {ece(p_te, y_te):.3f} -> calibrated {ece(cal.predict(p_te), y_te):.3f}")
log(f"  idle Normal windows of session 0: mean raw p = {oof[(sess == 0) & (y == 0) & idle_tr].mean():.3f} -> calibrated "
    f"{cal.predict(oof[(sess == 0) & (y == 0) & idle_tr]).mean():.3f}")
grid = np.linspace(0.30, 0.99, 15)
distinct = sorted(set(np.round(cal.predict(grid), 3)))
hi = oof >= 0.9
log(f"  FLATNESS CHECK: 15 raw scores between 0.30 and 0.99 map to {len(distinct)} distinct calibrated value(s): {distinct}")
log(f"  Why: OOF windows with raw >= 0.9: {int(hi.sum())}, true fault rate {y[hi].mean():.3f}; "
    f"{int((hi & np.isin(sess, [0, 5]) & (y == 0)).sum())} of them are Normal windows of the Normal-only sessions 0 and 5")
log("  VERDICT: isotonic calibration REJECTED - it erases the ranking information (a raw 0.99 and a raw 0.30 get the same value) "
    "and a 'probability' fitted across recordings is unstable (same reason as the unstable threshold, step 02). "
    "The pipeline keeps the raw Model 1 probability; reliability is reported empirically (precision at t = 0.5).")
with open(os.path.join(log.dir, "rejected_isotonic_calibrator.pkl"), "wb") as f:
    pickle.dump(cal, f)

# 3. test measurement --------------------------------------------------------------------------------------
log("\n== 3. Held-out sessions: the flag as a review queue (labels unchanged) ==")
flag = p_te >= C.MODEL1_THRESHOLD
amb = flag & idle_te
res = {"flagged": int(flag.sum()), "flagged_idle": int(amb.sum()),
       "flagged_idle_true_faults": int((amb & (y_te == 1)).sum()), "flagged_idle_false_alarms": int((amb & (y_te == 0)).sum()),
       "flagged_nonidle_true_faults": int((flag & ~idle_te & (y_te == 1)).sum()), "flagged_nonidle_false_alarms": int((flag & ~idle_te & (y_te == 0)).sum()),
       "idle_true_faults_by_type": {k: int(v) for k, v in pd.Series(y_te_str[amb & (y_te == 1)]).value_counts().items()},
       "idle_by_test_session": {int(s): {"flagged": int((flag & (te_sess == s)).sum()), "flagged_idle": int((amb & (te_sess == s)).sum())} for s in np.unique(te_sess)},
       "ece": {"oof_raw": round(ece(oof, y), 4), "oof_cal": round(ece(cal.predict(oof), y), 4), "test_raw": round(ece(p_te, y_te), 4), "test_cal": round(ece(cal.predict(p_te), y_te), 4)}}
log(f"  flagged {res['flagged']}: idle {res['flagged_idle']} (true faults {res['flagged_idle_true_faults']}, false alarms {res['flagged_idle_false_alarms']}); "
    f"non-idle {res['flagged'] - res['flagged_idle']} (true faults {res['flagged_nonidle_true_faults']}, false alarms {res['flagged_nonidle_false_alarms']})")
log(f"  idle true faults by type: {res['idle_true_faults_by_type']}")
log("  per test session (flagged / of which idle): " + ", ".join(f"s{s}: {v['flagged']}/{v['flagged_idle']}" for s, v in res["idle_by_test_session"].items()))
log("  Reading: the flag marks windows whose fault label rests on 'no traffic' evidence only; it changes no label.")
log.save_json("results.json", res)
log.close()
