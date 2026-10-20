"""Step 10a - Model 1: diagnosis-driven alternatives for the cross-session failure.

Diagnosis (step 04): the detector fails on whole recordings whose KPI LEVELS differ from the
training recordings (session 0 inverted: Normal p~0.99 / real Jamming p~0.01; session 5: 66% false
alarms). Hypotheses: H1 reliance on absolute levels; H2 shape features transfer across
recordings while levels do not (Model 2 evidence: QUANT Jamming F1 0.99 vs CatBoost 0.00).

Candidates (step-03 protocol: 5 seeds x 5 session folds, 80/20 rebalance inside the training
fold, natural-rate validation, threshold 0.5):
    adopted        CatBoost, Config A (correct parser), no double weighting   [= step-03 fix_both]
    adopted+G      + G1 G2 G3 trend/co-movement features                       [= step-05a best]
    jitter         adopted+G + level-jitter augmentation (H1): one extra copy of every training
                   window whose per-channel level (mean/min/max) is shifted by delta_c ~ N(0, s_c),
                   s_c = std of per-session channel medians over the training-fold Normal
                   sessions. Teaches invariance to recording-level offsets. Nested (s_c from
                   training-fold sessions only).
    quant          QUANT binary detector on the raw windows (H2)
    quant+cb       average of QUANT and adopted+G probabilities (both views)
Metrics: PR-AUC, P/R/F1 at 0.5, false alarms, FA in sessions 0 and 5, recall of session 0's real
Jamming, recall in sessions 2/12/18 (fault-only). Paired Nadeau-Bengio tests vs `adopted`.
One post-hoc test check (5 held-out sessions). Rule: adopt only if p < 0.05, material, holds on test.

Run from the phase2 folder:   python scripts/10a_model1_shift_experiments.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/.kpi_rag_vendor"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, features, load_splits
from src.evaluation import binary_report, nadeau_bengio_ttest
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog

log = StepLog("10a_model1_shift")
d = load_splits()
y = (d["train_y"] != C.NORMAL_LABEL).astype(int)
sess = d["train_sessions"]
raw = np.load(C.RAW_KPIS, mmap_mode="r")
A = features(d, "train", True)                              # (25688, 64) correct parser
G = F.compute(d["train_idx"])                               # (25688, 96) cached
AG = np.hstack([A, G])
LEVEL_COLS = np.array([c * 4 + k for c in range(16) for k in (0, 2, 3)])   # mean, min, max per channel
VARIANTS = ["adopted", "adopted+G", "jitter", "quant", "quant+cb"]
TAG_SESS = {"fa_s0": (0, 0), "fa_s5": (5, 0), "rec_jam_s0": (0, 1), "rec_s2": (2, 1), "rec_s12": (12, 1), "rec_s18": (18, 1)}


def jitter_copy(X, tr_rows, keep, rng):
    """Level-jitter augmentation: shift mean/min/max of each channel by a per-window offset."""
    normal_tr = tr_rows[y[tr_rows] == 0]
    sess_tr = sess[normal_tr]
    med = np.stack([np.median(A[normal_tr[sess_tr == s]][:, 0::4], 0) for s in np.unique(sess_tr)])  # (n_sess, 16) channel medians
    s_c = med.std(0)                                                                         # (16,) between-session spread
    Xj = X[keep].copy()
    delta = rng.normal(0, 1, (len(keep), 16)) * s_c
    for k in (0, 2, 3):
        Xj[:, [c * 4 + k for c in range(16)]] += delta
    return Xj


def quant_proba(Str, ytr, Sva, seed):
    from aeon.classification.interval_based import QUANTClassifier
    est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy",
                               class_weight="balanced", random_state=seed, n_jobs=8)
    m = QUANTClassifier(estimator=est, random_state=seed).fit(Str, ytr)
    return m.predict_proba(Sva)[:, list(m.classes_).index(1)]


def cb_proba(Xtr, ytr, Xva, seed):
    m = anomaly_detector(seed, class_weighting=C.MODEL1_CLASS_WEIGHTING)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def probas(tr, va, seed, S_va, A_va, AG_va):
    keep = rebalance_80_20(tr, y, seed=seed)
    P = {"adopted": cb_proba(A[keep], y[keep], A_va, seed),
         "adopted+G": cb_proba(AG[keep], y[keep], AG_va, seed)}
    Xj = jitter_copy(AG, tr, keep, np.random.default_rng(seed * 7 + 1))
    P["jitter"] = cb_proba(np.vstack([AG[keep], Xj]), np.concatenate([y[keep], y[keep]]), AG_va, seed)
    P["quant"] = quant_proba(np.asarray(raw[d["train_idx"][keep]], dtype=np.float64), y[keep], S_va, seed)
    P["quant+cb"] = (P["quant"] + P["adopted+G"]) / 2
    return P


ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(A, y, groups=sess), start=1):
        if (seed, fold) in done:
            continue
        t0 = time.time()
        S_va = np.asarray(raw[d["train_idx"][va]], dtype=np.float64)
        P = probas(tr, va, seed, S_va, A[va], AG[va])
        sv, yv = sess[va], y[va]
        line = []
        for v in VARIANTS:
            p = P[v]
            r = binary_report(p, yv, C.MODEL1_THRESHOLD)
            flag = p >= C.MODEL1_THRESHOLD
            rec = {"variant": v, "seed": seed, "fold": fold, "pr_auc": average_precision_score(yv, p),
                   "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "false_alarms": r["FP"],
                   "n_train": len(tr), "n_val": len(va)}
            for k, (s, cls) in TAG_SESS.items():
                m = (sv == s) & (yv == cls)
                rec[k] = float(flag[m].mean()) if m.any() else np.nan
            rows.append(rec)
            line.append(f"{v}={rec['pr_auc']:.3f}")
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        log(f"  seed={seed:<5} fold={fold}  PR-AUC  " + "  ".join(line) + f"  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(log.path("folds.csv"), index=False)
cols = ["pr_auc", "precision", "recall", "f1", "false_alarms"] + list(TAG_SESS)
summary = res.groupby("variant")[cols].mean().loc[VARIANTS]
log("\n== Mean over 25 folds (t = 0.5); fa_* = false-alarm rate in that Normal session, rec_* = recall in that fault session ==")
log(summary.round(3).to_string())

base = res[res.variant == "adopted"].sort_values(["seed", "fold"])
stats = {}
log("\n== Paired vs adopted (Nadeau-Bengio corrected t-test) ==")
for v in VARIANTS[1:]:
    cur = res[res.variant == v].sort_values(["seed", "fold"])
    stats[v] = {}
    for m in ("pr_auc", "f1", "recall", "precision"):
        diff, t_stat, p, w = nadeau_bengio_ttest(cur[m].values - base[m].values, base.n_train.values, base.n_val.values)
        stats[v][m] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4), "p_wilcoxon": round(w, 4),
                       "folds_better": int(((cur[m].values - base[m].values) > 0).sum())}
    s = stats[v]
    log(f"  {v:10s} dPR-AUC={s['pr_auc']['mean_diff']:+.4f} (p={s['pr_auc']['p_corrected']:.3f}, {s['pr_auc']['folds_better']}/25)"
        f"  dF1={s['f1']['mean_diff']:+.4f} (p={s['f1']['p_corrected']:.3f})  dRecall={s['recall']['mean_diff']:+.4f}  dPrec={s['precision']['mean_diff']:+.4f}")

log("\n== POST-HOC test (5 held-out sessions, 6,312 windows; not used to decide) ==")
labels = csv_labels()
y_te = (labels[d["test_idx"]] != C.NORMAL_LABEL).astype(int)
te_sess = d["test_sessions"]
A_te, G_te = features(d, "test", True), F.compute(d["test_idx"])
S_te = np.asarray(raw[d["test_idx"]], dtype=np.float64)
P = probas(np.arange(len(y)), None, C.SEED, S_te, A_te, np.hstack([A_te, G_te]))
test = {}
for v in VARIANTS:
    r = binary_report(P[v], y_te, C.MODEL1_THRESHOLD)
    r["pr_auc"] = round(average_precision_score(y_te, P[v]), 4)
    flag = P[v] >= C.MODEL1_THRESHOLD
    r["per_session"] = {int(s): {"n": int((te_sess == s).sum()), "flagged": int(flag[te_sess == s].sum()),
                                 "faults": int(y_te[te_sess == s].sum()),
                                 "caught": int((flag & (y_te == 1))[te_sess == s].sum())} for s in np.unique(te_sess)}
    test[v] = r
    log(f"  {v:10s} PR-AUC={r['pr_auc']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  "
        f"TP={r['TP']} FP={r['FP']} FN={r['FN']}  per-session flagged/faults: "
        + ", ".join(f"s{s}: {x['flagged']}/{x['faults']}" for s, x in r["per_session"].items()))
log("  reference: Phase 1 TP 269 / FP 1105; step 03 adopted TP 266 / FP 116")
log.save_json("results.json", {"cv_means": summary.round(4).to_dict(orient="index"), "paired_vs_adopted": stats, "posthoc_test": test})
log.close()
