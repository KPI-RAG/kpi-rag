"""Step 10b - Why does EVERY detector (level-based CatBoost and shape-based QUANT) fail on session 0?

Session 0 is the only mixed recording (3,007 "Normal" + 93 real Jamming windows). Step 10a shows
fold 4 (sessions 0, 6, 12) has PR-AUC ~0.02 for all five Model 1 variants, so the cause is not
the model. Hypotheses:
  D1  session 0's "Normal" windows are not normal (recorded during the live jamming campaign, labelled
      Normal outside the annotated segments) -> they look like faults to every model.
  D2  session 0's Jamming differs from session 2's Jamming (the only other Jamming source in training).
Tests (training partition only, cheap):
  T1  separability: can a small classifier tell session-0 Normal from other Normal windows? (AUC)
  T2  nearest-class: for session-0 Normal windows, is the closest class centroid (standardised
      Config A) Normal-other, Jamming, or another fault? Compare with Normal windows of session 5/6.
  T3  Jamming s0 vs s2: centroid distance vs within-class spread; channel-wise effect sizes.
  T4  label sanity: KPI levels of session-0 Normal vs session-0 Jamming vs other Normal (medians).

Run from the phase2 folder:   python scripts/10b_session0_diagnosis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config as C
from src.data import features, load_splits
from src.utils import StepLog

CH = ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR", "TX_Bytes", "RX_Bytes",
      "Est_UL_Buffer", "PRBs_DL", "PRBs_UL", "PRB_Util_DL", "PRB_Util_UL", "UL_Packets", "DL_Packets"]
log = StepLog("10b_session0_diagnosis")
d = load_splits()
A = features(d, "train", True)
y = d["train_y"]; sess = d["train_sessions"]
normal = y == C.NORMAL_LABEL
mu, sd = A[normal].mean(0), A[normal].std(0) + 1e-9
Z = (A - mu) / sd

# T1 separability
log("== T1. Session-0 Normal vs other Normal: 5-fold AUC of a 100-tree ExtraTrees ==")
for s in (0, 5, 6, 10):
    m = normal & np.isin(sess, [s] + [t for t in np.unique(sess[normal]) if t != s])
    lab = (sess[normal] == s).astype(int)
    p = cross_val_predict(ExtraTreesClassifier(100, random_state=0, n_jobs=8), Z[normal], lab,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    log(f"  session {s:>2} Normal ({int(lab.sum())} windows) vs other Normal: AUC = {roc_auc_score(lab, p):.3f}")
log("  (AUC ~1.0 = that session's Normal windows are trivially distinguishable, i.e. a recording fingerprint)")

# T2 nearest centroid
log("\n== T2. Nearest class centroid (standardised Config A) for Normal windows of selected sessions ==")
cent = {c: Z[y == c].mean(0) for c in np.unique(y)}
cent["Normal (other sessions)"] = Z[normal & (sess != 0)].mean(0)
names = [c for c in cent if c != C.NORMAL_LABEL]
for s in (0, 5, 6, 10, 20):
    m = normal & (sess == s)
    dist = np.stack([np.linalg.norm(Z[m] - cent[c], axis=1) for c in names], 1)
    near = pd.Series(np.array(names)[dist.argmin(1)]).value_counts(normalize=True)
    log(f"  session {s:>2}: " + ", ".join(f"{k[:28]} {v:.0%}" for k, v in near.head(3).items()))

# T3 Jamming s0 vs s2
log("\n== T3. Jamming in session 0 vs session 2 ==")
j0, j2 = Z[(y == "Jamming") & (sess == 0)], Z[(y == "Jamming") & (sess == 2)]
within = (np.linalg.norm(j0 - j0.mean(0), axis=1).mean() + np.linalg.norm(j2 - j2.mean(0), axis=1).mean()) / 2
log(f"  centroid distance s0-s2 = {np.linalg.norm(j0.mean(0) - j2.mean(0)):.2f}  vs mean within-session spread = {within:.2f}")
eff = (j0.mean(0) - j2.mean(0)) / (np.sqrt((j0.var(0) + j2.var(0)) / 2) + 1e-9)
top = np.argsort(np.abs(eff))[::-1][:6]
log("  largest standardised differences: " + ", ".join(f"{CH[i // 4]}_{['mean', 'std', 'min', 'max'][i % 4]} {eff[i]:+.1f}" for i in top))
n0, oth = Z[normal & (sess == 0)], Z[normal & (sess != 0)]
log(f"  session-0 Normal centroid: distance to Jamming-s0 {np.linalg.norm(n0.mean(0) - j0.mean(0)):.2f}, "
    f"to Jamming-s2 {np.linalg.norm(n0.mean(0) - j2.mean(0)):.2f}, to other Normal {np.linalg.norm(n0.mean(0) - oth.mean(0)):.2f}")

# T4 medians
log("\n== T4. Channel medians (mean feature): session-0 Normal vs session-0 Jamming vs other Normal ==")
rows = []
for i, ch in enumerate(CH):
    rows.append({"channel": ch, "s0_normal": np.median(A[normal & (sess == 0), i * 4]),
                 "s0_jamming": np.median(A[(y == "Jamming") & (sess == 0), i * 4]),
                 "other_normal": np.median(A[normal & (sess != 0), i * 4]),
                 "s2_jamming": np.median(A[(y == "Jamming") & (sess == 2), i * 4])})
t4 = pd.DataFrame(rows).set_index("channel")
t4.to_csv(log.path("session0_channel_medians.csv"))
log(t4.round(3).to_string())
log.close()
