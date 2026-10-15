"""Step 05b - Do the new feature groups (src/features.py) improve Model 2?

Same harness as the reproduced baseline (step 01): 935 training fault windows,
CatBoost defaults + balanced class weights, 5 seeds x 5 session folds, macro-F1.
    A_phase1     Phase 1 Config A (BLER bug)     -> reused from results/01_baseline
    A_fixed      Config A, correctly parsed
    A+G1G3       A_fixed + segment shift + co-movement   (112)
    A+G1G2G3     A_fixed + all three new groups          (160)
Decision rule: adopt only if macro-F1 is higher than A_phase1 with corrected p < 0.05.
Also: pooled seed-42 confusion for the two known hard pairs (HNC Gradual vs Buffer
Overflow, CCI Mild vs Severe), and one post-hoc test check on the 3 test sessions.

Run from the phase2 folder:   python scripts/05b_model2_new_features.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, features, load_splits
from src.evaluation import nadeau_bengio_ttest
from src.models import balanced_class_weights, fault_classifier
from src.utils import StepLog

HARD_PAIRS = [("High Network Congestion (Gradual Buildup)", "Buffer Overflow (Gradual Buildup)"),
              ("Co-Channel Interference (Severe)", "Co-Channel Interference (Mild)")]

log = StepLog("05b_model2_new_features")
d = load_splits()
fault = d["train_y"] != C.NORMAL_LABEL
y = d["train_y"][fault]
groups = d["train_sessions"][fault]
rows_tr = d["train_idx"][fault]

A_fixed = features(d, "train", True)[fault]
NEW = F.compute(rows_tr)
X = {"A_fixed": A_fixed,
     "A+G1G3": np.hstack([A_fixed, NEW[:, 0:32], NEW[:, 80:96]]),
     "A+G1G2G3": np.hstack([A_fixed, NEW])}
VARIANTS = list(X)
log(f"fault windows: {len(y)} | dims: " + ", ".join(f"{v}={X[v].shape[1]}" for v in VARIANTS))

base = pd.read_csv(os.path.join(C.RESULTS, "01_baseline", "baseline_folds.csv"))
base = base.rename(columns={"f1_macro": "f1"}).assign(variant="A_phase1")

rows, preds42 = [], {v: np.empty(len(y), dtype=object) for v in VARIANTS}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(A_fixed, y, groups=groups), start=1):
        w = balanced_class_weights(y[tr])
        line = []
        for v in VARIANTS:
            clf = fault_classifier(seed, class_weights=w)
            clf.fit(X[v][tr], y[tr])
            pred = clf.predict(X[v][va]).ravel()
            if seed == C.SEED:
                preds42[v][va] = pred
            f1 = f1_score(y[va], pred, average="macro", zero_division=0)
            rows.append({"variant": v, "seed": seed, "fold": fold, "f1": f1,
                         "accuracy": accuracy_score(y[va], pred),
                         "n_train": len(tr), "n_val": len(va)})
            line.append(f"{v}={f1:.4f}")
        b = base[(base.seed == seed) & (base.fold == fold)].f1.iloc[0]
        log(f"  seed={seed:<5} fold={fold}  macro-F1  A_phase1={b:.4f}  " + "  ".join(line))

res = pd.concat([base[["variant", "seed", "fold", "f1", "n_train", "n_val"]], pd.DataFrame(rows)])
res.to_csv(log.path("folds.csv"), index=False)
order = ["A_phase1"] + VARIANTS
summary = res.groupby("variant")[["f1", "accuracy"]].agg(["mean", "std"]).loc[order]
log("\n== Macro-F1 / accuracy over 25 folds ==")
log(summary.round(4).to_string())

ref = res[res.variant == "A_phase1"].sort_values(["seed", "fold"])
stats = {}
log("\n== Paired comparison with A_phase1 (= 0.2908 baseline) ==")
for v in VARIANTS:
    cur = res[res.variant == v].sort_values(["seed", "fold"])
    diffs = cur.f1.values - ref.f1.values
    diff, t_stat, p, wp = nadeau_bengio_ttest(diffs, ref.n_train.values, ref.n_val.values)
    adopt = diff > 0 and p < 0.05
    stats[v] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4), "p_wilcoxon": round(wp, 4),
                "folds_better": int((diffs > 0).sum()), "meets_decision_rule": bool(adopt)}
    log(f"  {v:10s} dF1={diff:+.4f}  corrected p={p:.3f}  Wilcoxon p={wp:.3f}  "
        f"{stats[v]['folds_better']}/25 better  -> {'ADOPT' if adopt else 'not adopted'}")

log("\n== Hard pairs (seed 42, pooled OOF): share of class A predicted as class B ==")
pairs = {}
for v in VARIANTS:
    for a, b in HARD_PAIRS:
        m = y == a
        share = float(np.mean(preds42[v][m] == b))
        pairs[f"{v}: {a} -> {b}"] = round(share, 3)
        log(f"  {v:10s} {a[:32]:32s} -> {b[:32]:32s} {share:.1%}")

log("\n== POST-HOC test check (test fault windows; not used to decide) ==")
te_fault = csv_labels()[d["test_idx"]] != C.NORMAL_LABEL
y_te = csv_labels()[d["test_idx"]][te_fault]
A_te = features(d, "test", True)[te_fault]
NEW_te = F.compute(d["test_idx"][te_fault])
X_te = {"A_fixed": A_te, "A+G1G3": np.hstack([A_te, NEW_te[:, 0:32], NEW_te[:, 80:96]]),
        "A+G1G2G3": np.hstack([A_te, NEW_te])}
test = {}
for v in VARIANTS:
    clf = fault_classifier(C.SEED, class_weights=balanced_class_weights(y))
    clf.fit(X[v], y)
    pred = clf.predict(X_te[v]).ravel()
    test[v] = {"macro_f1": round(f1_score(y_te, pred, average="macro", zero_division=0), 4),
               "accuracy": round(accuracy_score(y_te, pred), 4)}
    log(f"  {v:10s} macro-F1={test[v]['macro_f1']:.4f}  accuracy={test[v]['accuracy']:.4f}  "
        f"(n={len(y_te)})")

log.save_json("results.json", {"cv": {k: {kk: round(float(vv), 4) for kk, vv in s.items()}
                                      for k, s in res.groupby("variant").f1.agg(["mean", "std"]).iterrows()},
                               "paired_vs_phase1": stats, "hard_pairs": pairs, "posthoc_test": test})
log.close()
