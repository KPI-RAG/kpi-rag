"""Step 02 - Choose Model 1's decision threshold out-of-fold.

Phase 1 used the default 0.5 threshold, chosen without validation. Here:
1. Out-of-fold (OOF) Model 1 probabilities: StratifiedGroupKFold(5) by session on the
   training partition; Normal downsampled to 80/20 inside each training fold only;
   validation folds scored at their natural ~96/4 rate.
2. Threshold rule (fixed before looking at results): highest OOF precision with
   OOF recall >= 0.85. F2-optimal threshold and per-fold spread are reported too.
3. Cascade effect: share of false alarms that Model 2 labels as the catch-all class.
4. One POST-HOC test check (test set already used in Phase 1; never used to choose).

Run from the phase2 folder:   python scripts/02_model1_threshold.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.data import check_phase1_parser, config_a_phase1_parser, csv_labels, load_splits
from src.evaluation import binary_report, threshold_f2, threshold_max_precision
from src.models import anomaly_detector, fault_classifier, rebalance_80_20
from src.utils import StepLog

log = StepLog("02_model1_threshold")
d = load_splits()
X = d["train_X"][:, :C.N_FEATURES_A]
y_str = d["train_y"]
y = (y_str != C.NORMAL_LABEL).astype(int)
groups = d["train_sessions"]
log(f"training partition: {len(y)} windows, {int(y.sum())} anomalies ({y.mean():.1%}), "
    f"{len(np.unique(groups))} sessions")


def catchall_share(m2_pred, false_alarm):
    return round(100 * float(np.mean(m2_pred[false_alarm] == C.CATCHALL_LABEL)), 1) \
        if false_alarm.any() else 0.0


# 1. OOF probabilities + fold-wise Model 2 --------------------------------------------
log("\n== 1. Out-of-fold Model 1 probabilities ==")
oof_prob = np.zeros(len(y))
oof_m2 = np.empty(len(y), dtype=object)
fold_thresholds = []
cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), start=1):
    keep = rebalance_80_20(tr, y)
    m1 = anomaly_detector()
    m1.fit(X[keep], y[keep])
    oof_prob[va] = m1.predict_proba(X[va])[:, 1]

    faults_tr = tr[y[tr] == 1]
    m2 = fault_classifier(C.SEED, params=C.MODEL2_PARAMS)
    m2.fit(X[faults_tr], y_str[faults_tr])
    oof_m2[va] = m2.predict(X[va]).ravel()

    fold_thresholds.append(threshold_f2(oof_prob[va], y[va]))
    log(f"  fold {fold}: {len(va)} windows ({int(y[va].sum())} anomalies), "
        f"PR-AUC={average_precision_score(y[va], oof_prob[va]):.3f}, "
        f"F2-optimal t={fold_thresholds[-1]:.3f}")

# 2. Threshold selection ---------------------------------------------------------------
log("\n== 2. Threshold selection on pooled OOF probabilities ==")
pr_auc = average_precision_score(y, oof_prob)
log(f"OOF PR-AUC = {pr_auc:.4f}   (a random detector scores {y.mean():.4f})")
log(f"F2-optimal threshold per fold: {np.round(fold_thresholds, 3).tolist()}")

candidates = {"default_0.5": 0.5,
              "max_precision_at_recall_0.85": threshold_max_precision(oof_prob, y, C.MIN_RECALL),
              "f2_optimal": threshold_f2(oof_prob, y)}
oof = {}
for name, t in candidates.items():
    r = binary_report(oof_prob, y, t)
    flagged = oof_prob >= t
    r["catchall_pct_of_false_alarms"] = catchall_share(oof_m2, flagged & (y == 0))
    oof[name] = r
    log(f"  {name:30s} t={t:.3f}  P={r['precision']:.3f} R={r['recall']:.3f} "
        f"F2={r['f2']:.3f}  TP={r['TP']} FP={r['FP']} FN={r['FN']}  "
        f"catch-all={r['catchall_pct_of_false_alarms']}%")
chosen = candidates["max_precision_at_recall_0.85"]
log(f"Chosen by the pre-registered rule: t = {chosen:.4f}")

# 3. Post-hoc test check -----------------------------------------------------------------
log("\n== 3. POST-HOC test check (reported only, not used for selection) ==")
check_phase1_parser(d, log=log)                     # test features built exactly as Phase 1
X_test = config_a_phase1_parser(d["test_idx"])
y_test = (csv_labels()[d["test_idx"]] != C.NORMAL_LABEL).astype(int)

keep = rebalance_80_20(np.arange(len(y)), y)       # Phase 1 recipe on the full partition
m1 = anomaly_detector()
m1.fit(X[keep], y[keep])
p_test = m1.predict_proba(X_test)[:, 1]
m2 = fault_classifier(C.SEED, params=C.MODEL2_PARAMS)
m2.fit(X[y == 1], y_str[y == 1])
m2_test = m2.predict(X_test).ravel()

test = {}
for name, t in [("default_0.5", 0.5), ("chosen", chosen)]:
    r = binary_report(p_test, y_test, t)
    r["catchall_pct_of_false_alarms"] = catchall_share(m2_test, (p_test >= t) & (y_test == 0))
    test[name] = r
    log(f"  {name:12s} t={t:.3f}  P={r['precision']:.3f} R={r['recall']:.3f}  "
        f"TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}  "
        f"catch-all={r['catchall_pct_of_false_alarms']}%")
ref = C.PHASE1_TEST_CM
log(f"  Phase 1 reference at 0.5: TP={ref['TP']} FP={ref['FP']} FN={ref['FN']} TN={ref['TN']}")

log.save_json("results.json", {"oof_pr_auc": round(pr_auc, 4),
                               "fold_f2_thresholds": np.round(fold_thresholds, 4).tolist(),
                               "oof": oof, "chosen_threshold": round(chosen, 4),
                               "posthoc_test": test, "phase1_test_reference": ref})
log.close()
