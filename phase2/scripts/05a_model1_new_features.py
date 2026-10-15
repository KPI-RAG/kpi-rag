"""Step 05a - Do the new feature groups (src/features.py) improve Model 1?

Variants (adopted Model 1 recipe: fixed features, 80/20 rebalance, no Balanced weights):
    A            Config A (64)                         = current adopted Model 1
    A+G1G3       Config A + segment shift + co-movement (112)
    A+G1G2G3     Config A + all three new groups       (160)
    levelfree    G1 + G2 + G3 only, no absolute levels  (96)   <- step 04 hypothesis
Protocol and decision rule as step 03: 5 seeds x 5 session folds, primary PR-AUC,
adopt only if corrected p < 0.05 and F1 at t = 0.5 is not lower. Extra diagnostics:
false alarms in sessions 0 and 5 (90% of all false alarms in step 04) and recall on
session 0's real Jamming windows. One post-hoc test check.

Run from the phase2 folder:   python scripts/05a_model1_new_features.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, features, load_splits
from src.evaluation import binary_report, nadeau_bengio_ttest
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog

log = StepLog("05a_model1_new_features")
d = load_splits()
y = (d["train_y"] != C.NORMAL_LABEL).astype(int)
sess = d["train_sessions"]

A = {"train": features(d, "train", True), "test": features(d, "test", True)}
NEW = {"train": F.compute(d["train_idx"]), "test": F.compute(d["test_idx"])}
g = {"G1": slice(0, 32), "G2": slice(32, 80), "G3": slice(80, 96)}


def build(part, variant):
    new = NEW[part]
    return {"A": A[part],
            "A+G1G3": np.hstack([A[part], new[:, g["G1"]], new[:, g["G3"]]]),
            "A+G1G2G3": np.hstack([A[part], new]),
            "levelfree": new}[variant]


VARIANTS = ["A", "A+G1G3", "A+G1G2G3", "levelfree"]
X = {v: build("train", v) for v in VARIANTS}
log(f"feature dims: " + ", ".join(f"{v}={X[v].shape[1]}" for v in VARIANTS))

rows = []
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(A["train"], y, groups=sess), start=1):
        keep = rebalance_80_20(tr, y, seed=seed)
        line = []
        for v in VARIANTS:
            m1 = anomaly_detector(seed, class_weighting=C.MODEL1_CLASS_WEIGHTING)
            m1.fit(X[v][keep], y[keep])
            p = m1.predict_proba(X[v][va])[:, 1]
            r = binary_report(p, y[va], C.MODEL1_THRESHOLD)
            flag, sv = p >= C.MODEL1_THRESHOLD, sess[va]
            jam0 = (sv == 0) & (y[va] == 1)
            rows.append({"variant": v, "seed": seed, "fold": fold,
                         "pr_auc": average_precision_score(y[va], p),
                         "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                         "false_alarms": r["FP"],
                         "fa_session0": int((flag & (sv == 0) & (y[va] == 0)).sum()),
                         "fa_session5": int((flag & (sv == 5) & (y[va] == 0)).sum()),
                         "jamming_s0_recall": float(flag[jam0].mean()) if jam0.any() else np.nan,
                         "n_train": len(tr), "n_val": len(va)})
            line.append(f"{v}={rows[-1]['pr_auc']:.3f}")
        log(f"  seed={seed:<5} fold={fold}  PR-AUC  " + "  ".join(line))

res = pd.DataFrame(rows)
res.to_csv(log.path("folds.csv"), index=False)
cols = ["pr_auc", "precision", "recall", "f1", "false_alarms", "fa_session0", "fa_session5",
        "jamming_s0_recall"]
summary = res.groupby("variant")[cols].mean().loc[VARIANTS]
log("\n== Mean over 25 folds (t = 0.5) ==")
log(summary.round(4).to_string())

base = res[res.variant == "A"].sort_values(["seed", "fold"])
stats = {}
log("\n== Paired comparison with A (Nadeau-Bengio corrected t-test) ==")
for v in VARIANTS[1:]:
    cur = res[res.variant == v].sort_values(["seed", "fold"])
    stats[v] = {}
    for m in ("pr_auc", "f1", "precision", "recall"):
        diff, t_stat, p, w = nadeau_bengio_ttest(cur[m].values - base[m].values,
                                                  base["n_train"].values, base["n_val"].values)
        wins = int(((cur[m].values - base[m].values) > 0).sum())
        stats[v][m] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4),
                       "p_wilcoxon": round(w, 4), "folds_better": wins}
    s = stats[v]
    adopt = s["pr_auc"]["mean_diff"] > 0 and s["pr_auc"]["p_corrected"] < 0.05 \
        and s["f1"]["mean_diff"] >= 0
    s["meets_decision_rule"] = bool(adopt)
    log(f"  {v:10s} dPR-AUC={s['pr_auc']['mean_diff']:+.4f} (p={s['pr_auc']['p_corrected']:.3f}, "
        f"{s['pr_auc']['folds_better']}/25 better)  dF1={s['f1']['mean_diff']:+.4f} "
        f"(p={s['f1']['p_corrected']:.3f})  dRecall={s['recall']['mean_diff']:+.4f}  "
        f"-> {'ADOPT' if adopt else 'not adopted'}")

log("\n== POST-HOC test check (3 held-out sessions; not used to decide) ==")
y_te = (csv_labels()[d["test_idx"]] != C.NORMAL_LABEL).astype(int)
keep = rebalance_80_20(np.arange(len(y)), y)
test = {}
for v in VARIANTS:
    m1 = anomaly_detector(C.SEED, class_weighting=C.MODEL1_CLASS_WEIGHTING)
    m1.fit(X[v][keep], y[keep])
    p = m1.predict_proba(build("test", v))[:, 1]
    r = binary_report(p, y_te, C.MODEL1_THRESHOLD)
    r["pr_auc"] = round(average_precision_score(y_te, p), 4)
    test[v] = r
    log(f"  {v:10s} PR-AUC={r['pr_auc']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f} "
        f"F1={r['f1']:.3f}  TP={r['TP']} FP={r['FP']} FN={r['FN']}")

log.save_json("results.json", {"oof_mean": summary.round(4).to_dict(orient="index"),
                               "paired_vs_A": stats, "posthoc_test": test})
log.close()
