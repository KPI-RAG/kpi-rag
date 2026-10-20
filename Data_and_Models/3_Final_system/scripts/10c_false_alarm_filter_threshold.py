"""Step 10c - Reject-option follow-up: decision policy instead of argmax.

Step 10 shows the hard-negative reject class (argmax decision) rejects 25-43 % of REAL fault windows
in cross-validation: the 1,000 hardest Normal windows are idle-link windows (p >= 0.99, sessions 0/5)
whose feature region overlaps quiet fault windows. Hypothesis: the Normal class is useful but the
argmax decision is too aggressive; rejecting only when P(Normal) is high should keep most of the
false-alarm removal while returning the lost faults.
Design: seed 42 x 5 session folds (screening), Model 2 = recipe + trend + hard1000 (as step 10);
decision: reject iff P(Normal) >= tau, tau in {0.5, 0.6, 0.7, 0.8, 0.9}, else best fault class.
Reported per tau: 11-class macro-F1 on real validation faults, share of faults rejected, and on the
held-out cascade (same Model 1 as step 10): FP rejected, residual catch-all, TP lost, useful precision.
Policy is chosen on the CV columns (fault rejection <= 10 % with the best macro-F1); test is post-hoc.

Run from the phase2 folder:   python scripts/10c_false_alarm_filter_threshold.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, features, load_splits
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog
from src.v3 import IDX_B_FFT, V3, fit_predict_w, full_feats, macro, synthetic_windows

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
log = StepLog("10c_false_alarm_filter_threshold")
d = load_splits(); labels = csv_labels()
y_all = d["train_y"]; y_bin = (y_all != C.NORMAL_LABEL).astype(int); sess = d["train_sessions"]
oof = np.load(os.path.join(C.RESULTS, "10_cascade_false_alarm_filter", "oof_model1_prob.npy"))
normal_idx = np.where(y_bin == 0)[0]
fault = y_bin == 1
real_rows, real_y, real_sess = d["train_idx"][fault], y_all[fault], sess[fault]
eng = V3(d); eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])
full = np.stack([full_feats(x) for x in sig]); Gf = F.from_signals(sig); A_ref = full[:, :64]
host_sess = d["session_ids"][eng.host_rows]
tbl = lambda fullX, GX: np.hstack([fullX[:, IDX_B_FFT], GX])
X2_real = tbl(full, Gf)
CLASSES = sorted(np.unique(real_y))


def feats_of(rows):
    eng._load_text([r for r in rows if int(r) not in eng.ROW2IDX])
    S = np.stack([eng.real_sig(r) for r in rows])
    return tbl(np.stack([full_feats(x) for x in S]), F.from_signals(S))


def train(tr_idx, va_sess, seed, fold, for_test=False):
    if for_test:
        z = np.load(os.path.join(C.RESULTS, "cache", "synth", f"all_s{seed}_n500_bug.npz"), allow_pickle=True); S_syn, Y_syn = z["S"], z["Y"]
    else:
        S_syn, Y_syn = synthetic_windows(eng, real_rows, real_sess, A_ref, host_sess, tr_idx, None, seed, fold)
    cand = normal_idx[~np.isin(sess[normal_idx], list(va_sess))]
    neg = cand[np.argsort(oof[cand])[::-1][:1000]]
    X = np.vstack([X2_real[tr_idx], tbl(np.stack([full_feats(x) for x in S_syn]), F.from_signals(S_syn)), feats_of(d["train_idx"][neg])])
    Y = np.concatenate([real_y[tr_idx], Y_syn, [C.NORMAL_LABEL] * len(neg)])
    clf, _ = fit_predict_w(X, Y, np.ones(len(Y)), X[:1], seed)
    return clf


def decide(clf, X, tau):
    P = clf.predict_proba(X); cls = [str(c) for c in clf.classes_]; ni = cls.index(C.NORMAL_LABEL)
    best_fault = np.array(cls)[np.argmax(np.where(np.arange(len(cls)) == ni, -1, P), axis=1)]
    return np.where(P[:, ni] >= tau, C.NORMAL_LABEL, best_fault)


log("== A. CV screening (seed 42, 5 folds): macro-F1 on real validation faults / share rejected, per tau ==")
cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
rows = []
for f_i, (tr, va) in enumerate(cv.split(A_ref, real_y, groups=real_sess)):
    t0 = time.time()
    clf = train(tr, set(real_sess[va].tolist()), C.SEED, f_i + 1)
    rec = {"fold": f_i + 1}
    for tau in TAUS:
        pred = decide(clf, X2_real[va], tau)
        rec[f"f1@{tau}"] = macro(real_y[va], pred); rec[f"rej@{tau}"] = float((pred == C.NORMAL_LABEL).mean())
    rows.append(rec)
    log(f"  fold {f_i+1}: " + "  ".join(f"tau={tau}: F1 {rec[f'f1@{tau}']:.3f} rej {rec[f'rej@{tau}']:.1%}" for tau in TAUS) + f"  ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows); res.to_csv(log.path("folds.csv"), index=False)
cvtab = pd.DataFrame({tau: {"macro_f1": res[f"f1@{tau}"].mean(), "faults_rejected": res[f"rej@{tau}"].mean()} for tau in TAUS}).T
log(cvtab.round(3).to_string())
ref = json.load(open(os.path.join(C.RESULTS, "10_cascade_false_alarm_filter", "results.json")))
ref_f1, ref_sd = ref["model2_cv"]["none"]["mean"], ref["model2_cv"]["none"]["std"]
ok = [tau for tau in TAUS if cvtab.loc[tau, "faults_rejected"] <= 0.10 and abs(cvtab.loc[tau, "macro_f1"] - ref_f1) <= 2 * ref_sd]
chosen = max(ok, key=lambda t: cvtab.loc[t, "macro_f1"]) if ok else None
log(f"  no-reject reference (step 10): {ref_f1:.4f} ± {ref_sd:.4f}; taus with <= 10 % faults rejected and F1 within 2 SD: {ok}; chosen tau = {chosen}")

log("\n== B. Cascade on the held-out sessions per tau (post-hoc) ==")
A_tr, A_te = features(d, "train", True), features(d, "test", True)
keep = rebalance_80_20(np.arange(len(y_bin)), y_bin)
m1 = anomaly_detector(C.SEED, class_weighting=C.MODEL1_CLASS_WEIGHTING).fit(A_tr[keep], y_bin[keep])
flag = m1.predict_proba(A_te)[:, 1] >= C.MODEL1_THRESHOLD
yt = labels[d["test_idx"]][flag]; is_fp = yt == C.NORMAL_LABEL
X_flag = feats_of(d["test_idx"][flag])
clf = train(np.arange(len(real_y)), set(), C.SEED, 0, for_test=True)
test = {}
for tau in TAUS + ["argmax"]:
    pred = decide(clf, X_flag, tau) if tau != "argmax" else clf.predict(X_flag).ravel()
    resid = is_fp & (pred != C.NORMAL_LABEL); dom = pd.Series(pred[resid]).value_counts()
    tp_rows = ~is_fp
    test[str(tau)] = {"fp_in": int(is_fp.sum()), "fp_rejected": int((is_fp & (pred == C.NORMAL_LABEL)).sum()), "fp_residual": int(resid.sum()),
                      "residual_catchall_share": round(float(dom.iloc[0] / resid.sum()), 3) if resid.sum() else 0.0,
                      "residual_catchall_class": dom.index[0] if len(dom) else None,
                      "tp_in": int(tp_rows.sum()), "tp_rejected": int((tp_rows & (pred == C.NORMAL_LABEL)).sum()),
                      "fault_label_correct": int((tp_rows & (pred == yt)).sum()),
                      "useful_precision": round(float((tp_rows & (pred == yt)).sum() / max((pred != C.NORMAL_LABEL).sum(), 1)), 3)}
    c = test[str(tau)]
    log(f"  tau={str(tau):6s} FP {c['fp_in']} -> rejected {c['fp_rejected']}, residual {c['fp_residual']} (dominant {c['residual_catchall_class']} {c['residual_catchall_share']:.0%})  "
        f"TP {c['tp_in']} -> rejected {c['tp_rejected']}, correct label {c['fault_label_correct']}  useful precision {c['useful_precision']:.3f}")
policy = {"reject_tau": chosen, "rule": "reject iff P(Normal) >= tau; chosen on CV (<= 10 % faults rejected, F1 within 2 SD, best F1)"}
log.save_json("results.json", {"cv": cvtab.round(4).to_dict(orient="index"), "chosen_tau": chosen, "test": test})
json.dump(policy, open(log.path("policy.json"), "w"), indent=2)
log.close()
