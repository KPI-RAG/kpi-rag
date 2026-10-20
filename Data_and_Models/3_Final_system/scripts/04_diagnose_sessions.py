"""Step 04 - Why is Model 1 near random on some sessions (fold 4 = sessions 0, 6, 12)?

Uses the ADOPTED Model 1 (src/config.py). Diagnostics only - nothing is tuned here.
1. Per-session table: windows, anomalies, fault mix, and out-of-fold Model 1 metrics
   (PR-AUC, recall, false-alarm rate at t = 0.5), seed 42, 5 session-grouped folds.
2. Covariate shift: for Normal windows, how far each session's typical feature values sit
   from the other sessions (robust z = median difference / pooled IQR).
3. Fault strength: anomaly segment length and the anomaly-vs-own-session-Normal contrast
   per session (are the faults in the failing sessions shorter or weaker?).

Run from the phase2 folder:   python scripts/04_diagnose_sessions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.data import features, load_splits
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog

CHANNELS = ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
            "TX_Bytes", "RX_Bytes", "Est_UL_Buffer", "PRBs_DL", "PRBs_UL",
            "PRB_Util_DL", "PRB_Util_UL", "UL_Packets", "DL_Packets"]
FEAT_NAMES = [f"{c}_{s}" for c in CHANNELS for s in ("mean", "std", "min", "max")]

log = StepLog("04_session_diagnosis")
d = load_splits()
X = features(d, "train", C.MODEL1_FIXED_FEATURES)
y_str = d["train_y"]
y = (y_str != C.NORMAL_LABEL).astype(int)
sess = d["train_sessions"]
meta = pd.read_csv(C.RAW_META).iloc[d["train_idx"]].reset_index(drop=True)

# 1. OOF probabilities with the adopted Model 1 -------------------------------------------
oof = np.zeros(len(y))
fold_of = np.zeros(len(y), dtype=int)
cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
for fold, (tr, va) in enumerate(cv.split(X, y, groups=sess), start=1):
    keep = rebalance_80_20(tr, y)
    m1 = anomaly_detector(C.SEED, class_weighting=C.MODEL1_CLASS_WEIGHTING)
    m1.fit(X[keep], y[keep])
    oof[va] = m1.predict_proba(X[va])[:, 1]
    fold_of[va] = fold

rows = []
for s in np.unique(sess):
    m = sess == s
    ys, ps = y[m], oof[m]
    faults = pd.Series(y_str[m][ys == 1]).value_counts()
    rows.append({
        "session": int(s), "fold": int(fold_of[m][0]), "windows": int(m.sum()),
        "anomalies": int(ys.sum()),
        "pr_auc": round(average_precision_score(ys, ps), 3) if ys.sum() else np.nan,
        "recall@0.5": round(float((ps[ys == 1] >= 0.5).mean()), 3) if ys.sum() else np.nan,
        "false_alarm_rate@0.5": round(float((ps[ys == 0] >= 0.5).mean()), 4),
        "median_p_normal": round(float(np.median(ps[ys == 0])), 4),
        "median_p_anomaly": round(float(np.median(ps[ys == 1])), 4) if ys.sum() else np.nan,
        "fault_mix": "; ".join(f"{k} ({v})" for k, v in faults.items()),
    })
tab = pd.DataFrame(rows).sort_values(["fold", "session"])
tab.to_csv(log.path("per_session.csv"), index=False)
log("\n== 1. Per-session out-of-fold Model 1 (adopted) ==")
log(tab.drop(columns="fault_mix").to_string(index=False))
log("\nFault mix of sessions with anomalies:")
for _, r in tab[tab.anomalies > 0].iterrows():
    log(f"  session {r.session:>2} (fold {r.fold}, PR-AUC {r.pr_auc}): {r.fault_mix}")

# 2. Covariate shift of Normal windows -----------------------------------------------------
log("\n== 2. Normal-window shift vs all other sessions (robust z, top 6 features) ==")
normal = y == 0
shift_rows = []
for s in np.unique(sess):
    a = X[normal & (sess == s)]
    b = X[normal & (sess != s)]
    if len(a) == 0:
        continue
    iqr = np.subtract(*np.percentile(b, [75, 25], axis=0))
    z = (np.median(a, 0) - np.median(b, 0)) / np.where(iqr > 0, iqr, 1.0)
    shift_rows.append({"session": int(s), "max_abs_z": float(np.max(np.abs(z))),
                       "n_features_|z|>3": int((np.abs(z) > 3).sum()),
                       **{FEAT_NAMES[i]: round(float(z[i]), 2) for i in range(len(z))}})
shift = pd.DataFrame(shift_rows).set_index("session")
shift.to_csv(log.path("normal_shift_z.csv"))
fail_sessions = tab[(tab.anomalies > 0) & (tab.pr_auc < 0.1)].session.tolist()
log(f"sessions with faults and PR-AUC < 0.1: {fail_sessions}")
summary_cols = ["max_abs_z", "n_features_|z|>3"]
log(shift[summary_cols].round(2).sort_values("n_features_|z|>3", ascending=False).to_string())
for s in fail_sessions:
    z = shift.loc[s, FEAT_NAMES].astype(float)
    top = z.abs().sort_values(ascending=False).head(6)
    log(f"  session {s}: " + ", ".join(f"{k}={z[k]:+.1f}" for k in top.index))

# 3. Fault strength per session ------------------------------------------------------------
log("\n== 3. Anomaly segment length and contrast vs own-session Normal ==")
seg_len = (meta["anom_end"] - meta["anom_start"] + 1).where(y == 1)
strength = []
for s in np.unique(sess):
    m_a = (sess == s) & (y == 1)
    m_n = (sess == s) & (y == 0)
    if m_a.sum() == 0 or m_n.sum() == 0:
        continue
    iqr = np.subtract(*np.percentile(X[m_n], [75, 25], axis=0))
    contrast = np.abs(np.median(X[m_a], 0) - np.median(X[m_n], 0)) / np.where(iqr > 0, iqr, 1.0)
    strength.append({"session": int(s), "anomalies": int(m_a.sum()),
                     "median_segment_len": float(np.median(seg_len[m_a])),
                     "median_contrast": round(float(np.median(contrast)), 2),
                     "n_features_contrast>3": int((contrast > 3).sum()),
                     "pr_auc": tab.set_index("session").loc[s, "pr_auc"]})
st = pd.DataFrame(strength).sort_values("pr_auc")
st.to_csv(log.path("fault_strength.csv"), index=False)
log(st.to_string(index=False))

log.save_json("summary.json", {"failing_sessions": fail_sessions,
                               "per_session": tab.drop(columns="fault_mix").to_dict(orient="records")})
log.close()
