"""Step 03 - Fix the two Model 1 defects found in step 02 and measure the effect.

Defect A (features): the Phase 1 regex corrupts the BLER features of ~2% of windows.
    Fix: Config A computed from the correctly parsed signals (raw_kpis.npy).
Defect B (weighting): imbalance corrected twice (80/20 rebalance + "Balanced" weights).
    Fix: keep the 80/20 rebalance only.

Four variants (2 x 2), everything else identical to Phase 1:
    phase1          buggy features  + double correction   (= Phase 1 Model 1)
    fix_features    fixed features  + double correction
    fix_weighting   buggy features  + single correction
    fix_both        fixed features  + single correction

Protocol: training partition only, StratifiedGroupKFold(5) by session x 5 seeds.
Primary metric: PR-AUC (average precision) on each validation fold at its natural rate.
Also at t = 0.5: precision, recall, F1, and the share of false alarms that Model 2
labels as the catch-all class (cascade noise).
Decision rule (fixed in advance): a variant replaces Phase 1 only if its PR-AUC is higher
with Nadeau-Bengio corrected p < 0.05 AND its mean F1 at t = 0.5 is not lower.
The test set is reported once, post-hoc, and is not used to decide.

Run from the phase2 folder:   python scripts/03_model1_fixes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.data import csv_labels, features, load_splits
from src.evaluation import binary_report, nadeau_bengio_ttest
from src.models import anomaly_detector, fault_classifier, rebalance_80_20
from src.utils import StepLog

VARIANTS = {                      # name: (fixed features?, Balanced class weights?)
    "phase1":        (False, True),
    "fix_features":  (True,  True),
    "fix_weighting": (False, False),
    "fix_both":      (True,  False),
}

log = StepLog("03_model1_fixes")
d = load_splits()
y_str = d["train_y"]
y = (y_str != C.NORMAL_LABEL).astype(int)
groups = d["train_sessions"]
X = {fixed: features(d, "train", fixed) for fixed in (False, True)}
n_diff = int((~np.isclose(X[False], X[True], rtol=1e-3, atol=1e-4)).any(1).sum())
log(f"training windows whose features change with the parser fix: {n_diff} / {len(y)}")


def catchall_pct(m2_pred, false_alarm):
    return 100 * float(np.mean(m2_pred[false_alarm] == C.CATCHALL_LABEL)) if false_alarm.any() else 0.0


# 1. Repeated grouped CV -----------------------------------------------------------------
log("\n== 1. Out-of-fold comparison (5 seeds x 5 folds) ==")
rows = []
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(X[False], y, groups=groups), start=1):
        keep = rebalance_80_20(tr, y, seed=seed)
        faults_tr = tr[y[tr] == 1]
        m2_pred = {}
        for fixed in (False, True):           # Model 2 (Phase 1 recipe) for the cascade metric
            m2 = fault_classifier(seed, params=C.MODEL2_PARAMS)
            m2.fit(X[fixed][faults_tr], y_str[faults_tr])
            m2_pred[fixed] = m2.predict(X[fixed][va]).ravel()
        for name, (fixed, weighted) in VARIANTS.items():
            m1 = anomaly_detector(seed, class_weighting=weighted)
            m1.fit(X[fixed][keep], y[keep])
            p = m1.predict_proba(X[fixed][va])[:, 1]
            r = binary_report(p, y[va], 0.5)
            rows.append({"variant": name, "seed": seed, "fold": fold,
                         "pr_auc": average_precision_score(y[va], p),
                         "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                         "false_alarms": r["FP"],
                         "catchall_pct": catchall_pct(m2_pred[fixed], (p >= 0.5) & (y[va] == 0)),
                         "n_train": len(tr), "n_val": len(va)})
        last = {r["variant"]: r["pr_auc"] for r in rows[-len(VARIANTS):]}
        log(f"  seed={seed:<5} fold={fold}  PR-AUC  " +
            "  ".join(f"{k}={v:.3f}" for k, v in last.items()))

res = pd.DataFrame(rows)
res.to_csv(log.path("folds.csv"), index=False)

# 2. Summary + paired statistics vs Phase 1 -----------------------------------------------
log("\n== 2. Mean over 25 folds (t = 0.5 for precision/recall/F1) ==")
summary = res.groupby("variant")[["pr_auc", "precision", "recall", "f1", "false_alarms",
                                  "catchall_pct"]].mean().loc[list(VARIANTS)]
log(summary.round(4).to_string())

base = res[res.variant == "phase1"].sort_values(["seed", "fold"])
stats = {}
log("\n== 3. Paired comparison with phase1 (Nadeau-Bengio corrected t-test) ==")
for name in list(VARIANTS)[1:]:
    v = res[res.variant == name].sort_values(["seed", "fold"])
    stats[name] = {}
    for metric in ("pr_auc", "f1", "precision"):
        diff, t_stat, p, w = nadeau_bengio_ttest(v[metric].values - base[metric].values,
                                                  base["n_train"].values, base["n_val"].values)
        stats[name][metric] = {"mean_diff": round(diff, 4), "t": round(t_stat, 3),
                               "p_corrected": round(p, 4), "p_wilcoxon": round(w, 4)}
    s = stats[name]
    adopt = s["pr_auc"]["mean_diff"] > 0 and s["pr_auc"]["p_corrected"] < 0.05 \
        and s["f1"]["mean_diff"] >= 0
    stats[name]["meets_decision_rule"] = bool(adopt)
    log(f"  {name:14s} dPR-AUC={s['pr_auc']['mean_diff']:+.4f} (p={s['pr_auc']['p_corrected']:.3f})"
        f"  dF1={s['f1']['mean_diff']:+.4f} (p={s['f1']['p_corrected']:.3f})"
        f"  dPrecision={s['precision']['mean_diff']:+.4f} (p={s['precision']['p_corrected']:.3f})"
        f"  -> {'ADOPT' if adopt else 'not adopted'}")

# 3. Post-hoc test check -------------------------------------------------------------------
log("\n== 4. POST-HOC test check (3 held-out sessions; not used to decide) ==")
y_test = (csv_labels()[d["test_idx"]] != C.NORMAL_LABEL).astype(int)
keep = rebalance_80_20(np.arange(len(y)), y)
test = {}
for fixed in (False, True):
    X_te = features(d, "test", fixed)
    m2 = fault_classifier(C.SEED, params=C.MODEL2_PARAMS)
    m2.fit(X[fixed][y == 1], y_str[y == 1])
    m2_te = m2.predict(X_te).ravel()
    for name, (f, weighted) in VARIANTS.items():
        if f != fixed:
            continue
        m1 = anomaly_detector(C.SEED, class_weighting=weighted)
        m1.fit(X[fixed][keep], y[keep])
        p = m1.predict_proba(X_te)[:, 1]
        r = binary_report(p, y_test, 0.5)
        r["pr_auc"] = round(average_precision_score(y_test, p), 4)
        r["catchall_pct_of_false_alarms"] = round(catchall_pct(m2_te, (p >= 0.5) & (y_test == 0)), 1)
        test[name] = r
for name in VARIANTS:
    r = test[name]
    log(f"  {name:14s} PR-AUC={r['pr_auc']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f} "
        f"F1={r['f1']:.3f}  TP={r['TP']} FP={r['FP']} FN={r['FN']}  "
        f"catch-all={r['catchall_pct_of_false_alarms']}%")

log.save_json("results.json", {"oof_mean": summary.round(4).to_dict(orient="index"),
                               "paired_vs_phase1": stats, "posthoc_test": test})
log.close()
