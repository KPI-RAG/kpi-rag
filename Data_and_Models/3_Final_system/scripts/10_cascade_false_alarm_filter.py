"""Step 10 - The complete cascade (Model 1 -> Model 2) and Raneem's reject option.

Model 1: the variant selected by step 10a's rule (else the adopted step-03 detector), t = 0.5.
Model 2: the synthetic-data model + trend (CatBoost, real + v3 500/class from the cache, Config B + FFT + G).
Reject option (Raneem, Phase 2 part 1): add a "Normal" class to Model 2 so false alarms can be
sent back instead of being forced into a fault label (Phase 1: 82% of FPs -> CCI-Mild).
    none          no reject class (reference)
    hard1000      hard-negative mining: the 1,000 training Normal windows with the highest
                  OUT-OF-FOLD Model 1 probability (session-grouped OOF => no leakage)
    random1000    Raneem's v1: 1,000 random training Normal windows (control)
Evaluation A (Model 2 quality, 25 session folds): 11-class macro-F1 on real validation fault
    windows (a fault predicted "Normal" counts as an error) + fault-rejection rate.
Evaluation B (cascade on the 5 held-out sessions, 6,312 windows, post-hoc): false alarms,
    FPs correctly rejected, residual catch-all share (dominant class among non-rejected FPs),
    true positives lost to rejection, useful precision, per-session and per-class confusion.
Promotion rule (Raneem's, verified): residual catch-all < 41.1% (half of 82.2%) AND 11-class
    macro-F1 within 2 SD of the no-reject reference.

Run from the phase2 folder:   python scripts/10_cascade_false_alarm_filter.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, features, load_splits
from src.evaluation import binary_report
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog
from src.v3 import IDX_B_FFT, V3, fit_predict_w, full_feats, macro, synthetic_windows

log = StepLog("10_cascade_false_alarm_filter")
d = load_splits()
labels = csv_labels()
y_all = d["train_y"]
y_bin = (y_all != C.NORMAL_LABEL).astype(int)
sess = d["train_sessions"]
raw = np.load(C.RAW_KPIS, mmap_mode="r")
CATCHALL = C.CATCHALL_LABEL

# ---- Model 1 variant from step 10a ---------------------------------------------------------------
M1 = "adopted"
r10a = os.path.join(C.RESULTS, "10a_model1_shift", "results.json")
if os.path.exists(r10a):
    r = json.load(open(r10a))
    for v, s in r["paired_vs_adopted"].items():
        if v in ("adopted+G", "jitter") and s["pr_auc"]["p_corrected"] < 0.05 and s["pr_auc"]["mean_diff"] >= 0.03 \
                and s["recall"]["mean_diff"] >= 0 and r["posthoc_test"][v]["FP"] <= r["posthoc_test"]["adopted"]["FP"]:
            M1 = v
log(f"Model 1 variant: {M1}")
A_tr, G_tr = features(d, "train", True), F.compute(d["train_idx"])
A_te, G_te = features(d, "test", True), F.compute(d["test_idx"])
X1_tr = np.hstack([A_tr, G_tr]) if M1 != "adopted" else A_tr
X1_te = np.hstack([A_te, G_te]) if M1 != "adopted" else A_te


def fit_m1(idx, seed):
    keep = rebalance_80_20(idx, y_bin, seed=seed)
    Xk, yk = X1_tr[keep], y_bin[keep]
    if M1 == "jitter":                       # level-jitter augmentation exactly as in step 10a
        normal_tr = idx[y_bin[idx] == 0]
        st = sess[normal_tr]
        med = np.stack([np.median(A_tr[normal_tr[st == s]][:, 0::4], 0) for s in np.unique(st)])
        delta = np.random.default_rng(seed * 7 + 1).normal(0, 1, (len(keep), 16)) * med.std(0)
        Xj = Xk.copy()
        for k in (0, 2, 3):
            Xj[:, [c * 4 + k for c in range(16)]] += delta
        Xk, yk = np.vstack([Xk, Xj]), np.concatenate([yk, yk])
    m1 = anomaly_detector(seed, class_weighting=C.MODEL1_CLASS_WEIGHTING)
    m1.fit(Xk, yk)
    return m1


# ---- Out-of-fold Model 1 probabilities on the whole training partition (for hard negatives) ----
log("\n== OOF Model 1 probabilities (5 session folds, seed 42) ==")
oof = np.zeros(len(y_bin))
cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
for tr, va in cv.split(X1_tr, y_bin, groups=sess):
    oof[va] = fit_m1(tr, C.SEED).predict_proba(X1_tr[va])[:, 1]
normal_idx = np.where(y_bin == 0)[0]
log(f"  Normal windows: {len(normal_idx)}; OOF p >= 0.5 on {int((oof[normal_idx] >= 0.5).sum())} of them "
    f"(top-1000 p range: {np.sort(oof[normal_idx])[-1000]:.3f} - {oof[normal_idx].max():.3f})")
np.save(log.path("oof_model1_prob.npy"), oof)

# ---- Model 2 data -------------------------------------------------------------------------------
fault = y_bin == 1
real_rows, real_y = d["train_idx"][fault], y_all[fault]
real_sess = sess[fault]
eng = V3(d)
eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])
full = np.stack([full_feats(x) for x in sig])
Gf = F.from_signals(sig)
A_ref = full[:, :64]
host_sess = d["session_ids"][eng.host_rows]
tbl = lambda fullX, GX: np.hstack([fullX[:, IDX_B_FFT], GX])
X2_real = tbl(full, Gf)
# features of Normal training windows in the Model 2 space (B+FFT+G) - from the Phase 1 parser
# signals so they match the synthetic/host cache; computed lazily for the needed windows only
_norm_feat_cache = {}
def normal_feats(idx):
    need = [i for i in idx if i not in _norm_feat_cache]
    if need:
        rows = d["train_idx"][need]
        eng._load_text([r for r in rows if int(r) not in eng.ROW2IDX])
        S = np.stack([eng.real_sig(r) for r in rows])
        Fx = tbl(np.stack([full_feats(x) for x in S]), F.from_signals(S))
        for i, f in zip(need, Fx): _norm_feat_cache[i] = f
    return np.stack([_norm_feat_cache[i] for i in idx])


def pick_negatives(kind, n, exclude_sess, seed):
    cand = normal_idx[~np.isin(sess[normal_idx], list(exclude_sess))]
    if kind == "hard":
        return cand[np.argsort(oof[cand])[::-1][:n]]
    return np.random.default_rng(seed).choice(cand, n, replace=False)


VARIANTS = {"none": None, "hard1000": ("hard", 1000), "random1000": ("random", 1000)}
CLASSES = sorted(np.unique(real_y))


def train_m2(tr_idx, va_sess, seed, fold, variant, for_test=False):
    if for_test:
        z = np.load(os.path.join(C.RESULTS, "cache", "synth", f"all_s{seed}_n500_bug.npz"), allow_pickle=True)
        S_syn, Y_syn = z["S"], z["Y"]
    else:
        S_syn, Y_syn = synthetic_windows(eng, real_rows, real_sess, A_ref, host_sess, tr_idx, None, seed, fold)
    X = [X2_real[tr_idx], tbl(np.stack([full_feats(x) for x in S_syn]), F.from_signals(S_syn))]
    Y = [real_y[tr_idx], Y_syn]
    if VARIANTS[variant]:
        kind, n = VARIANTS[variant]
        neg = pick_negatives(kind, n, va_sess, seed)
        X.append(normal_feats(neg)); Y.append(np.array([C.NORMAL_LABEL] * n))
    Xtr, ytr = np.vstack(X), np.concatenate(Y)
    clf, _ = fit_predict_w(Xtr, ytr, np.ones(len(ytr)), Xtr[:1], seed)
    return clf


# ---- Evaluation A: Model 2 quality with / without reject class ----------------------------------
log("\n== A. Model 2 11-class macro-F1 on real validation faults (25 session folds) ==")
ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
for seed in C.CV_SEEDS:
    cv2 = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for f_i, (tr, va) in enumerate(cv2.split(A_ref, real_y, groups=real_sess)):
        if (seed, f_i + 1) in done:
            continue
        t0 = time.time()
        rec = {"seed": seed, "fold": f_i + 1, "n_train": len(tr), "n_val": len(va)}
        va_sess = set(real_sess[va].tolist())
        for v in VARIANTS:
            clf = train_m2(tr, va_sess, seed, f_i + 1, v)
            pred = clf.predict(X2_real[va]).ravel()
            rec[v] = macro(real_y[va], pred)
            rec[f"{v}_rejected"] = float((pred == C.NORMAL_LABEL).mean())
        rows.append(rec); pd.DataFrame(rows).to_csv(ckpt, index=False)
        log(f"  seed={seed:<5} fold={f_i+1}  " + "  ".join(f"{v}={rec[v]:.3f} (rej {rec[v+'_rejected']:.1%})" for v in VARIANTS) + f"  ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows)
res.to_csv(log.path("folds.csv"), index=False)
ref_mean, ref_sd = res["none"].mean(), res["none"].std()
log(f"  reference (no reject): {ref_mean:.4f} ± {ref_sd:.4f}")
for v in ("hard1000", "random1000"):
    log(f"  {v}: {res[v].mean():.4f} ± {res[v].std():.4f}  (within 2 SD: {abs(res[v].mean() - ref_mean) <= 2 * ref_sd})  "
        f"faults rejected: {res[v + '_rejected'].mean():.1%}")

# ---- Evaluation B: the full cascade on the 5 held-out sessions ----------------------------------
log("\n== B. Cascade on the held-out sessions (6,312 windows; post-hoc) ==")
y_te_str = labels[d["test_idx"]]
y_te = (y_te_str != C.NORMAL_LABEL).astype(int)
te_sess = d["test_sessions"]
m1 = fit_m1(np.arange(len(y_bin)), C.SEED)
p1 = m1.predict_proba(X1_te)[:, 1]
flag = p1 >= C.MODEL1_THRESHOLD
r1 = binary_report(p1, y_te, C.MODEL1_THRESHOLD)
log(f"  Model 1 ({M1}): TP={r1['TP']} FP={r1['FP']} FN={r1['FN']} TN={r1['TN']}  (step 03 adopted: TP 266 / FP 116; Phase 1: 269 / 1105)")
S_te = np.asarray(raw[d["test_idx"]], dtype=np.float64)
te_text_rows = d["test_idx"]
eng._load_text([r for r in te_text_rows[flag] if int(r) not in eng.ROW2IDX])
S_te_flag = np.stack([eng.real_sig(r) for r in te_text_rows[flag]])
X2_te_flag = tbl(np.stack([full_feats(x) for x in S_te_flag]), F.from_signals(S_te_flag))
cascade = {}
for v in VARIANTS:
    clf = train_m2(np.arange(len(real_y)), set(), C.SEED, 0, v, for_test=True)
    pred2 = clf.predict(X2_te_flag).ravel()
    yt = y_te_str[flag]
    is_fp = yt == C.NORMAL_LABEL
    rejected_fp = is_fp & (pred2 == C.NORMAL_LABEL)
    residual_fp = is_fp & (pred2 != C.NORMAL_LABEL)
    tp_rows = ~is_fp
    tp_rejected = tp_rows & (pred2 == C.NORMAL_LABEL)
    retained = pred2 != C.NORMAL_LABEL
    correct_fault = tp_rows & (pred2 == yt)
    dom = pd.Series(pred2[residual_fp]).value_counts()
    catchall_share = float(dom.iloc[0] / residual_fp.sum()) if residual_fp.sum() else 0.0
    cm = confusion_matrix(yt, pred2, labels=CLASSES + [C.NORMAL_LABEL])
    pd.DataFrame(cm, index=CLASSES + [C.NORMAL_LABEL], columns=CLASSES + [C.NORMAL_LABEL]).to_csv(log.path(f"confusion_cascade_test_{v}.csv"))
    cascade[v] = {
        "flagged": int(flag.sum()), "false_alarms_in": int(is_fp.sum()),
        "fp_rejected": int(rejected_fp.sum()), "fp_residual": int(residual_fp.sum()),
        "residual_catchall_class": (dom.index[0] if len(dom) else None), "residual_catchall_share": round(catchall_share, 3),
        "cci_mild_share_of_all_fp": round(float((is_fp & (pred2 == CATCHALL)).sum() / max(is_fp.sum(), 1)), 3),
        "true_faults_in": int(tp_rows.sum()), "tp_rejected": int(tp_rejected.sum()),
        "tp_loss_rate": round(float(tp_rejected.sum() / max(tp_rows.sum(), 1)), 3),
        "useful_precision": round(float(correct_fault.sum() / max(retained.sum(), 1)), 3),
        "fault_label_accuracy_on_true_faults": round(float(correct_fault.sum() / max(tp_rows.sum(), 1)), 3),
        "per_session_fp_residual": {int(s): int((residual_fp & (te_sess[flag] == s)).sum()) for s in np.unique(te_sess)},
    }
    c = cascade[v]
    log(f"  {v:11s} flagged={c['flagged']}  FP in={c['false_alarms_in']} -> rejected {c['fp_rejected']}, residual {c['fp_residual']} "
        f"(dominant {c['residual_catchall_class']} {c['residual_catchall_share']:.1%}; CCI-Mild share of all FP {c['cci_mild_share_of_all_fp']:.1%})  "
        f"TP in={c['true_faults_in']} -> rejected {c['tp_rejected']} ({c['tp_loss_rate']:.1%})  useful precision={c['useful_precision']:.3f}  "
        f"fault-label acc={c['fault_label_accuracy_on_true_faults']:.3f}")
log("  Phase 1 reference: 1,105 FP, 82.2% CCI-Mild; Raneem's report (hard-negative v2): CCI-Mild 16.3%, 83.3% of FP rejected, 4.8% TP lost")

promotion = {}
for v in ("hard1000", "random1000"):
    ok_f1 = abs(res[v].mean() - ref_mean) <= 2 * ref_sd
    ok_ca = cascade[v]["residual_catchall_share"] < 0.411
    promotion[v] = {"f1_within_2sd": bool(ok_f1), "residual_catchall_below_41pct": bool(ok_ca), "promoted": bool(ok_f1 and ok_ca)}
    log(f"  promotion {v}: F1 within 2 SD = {ok_f1}, residual catch-all < 41.1% = {ok_ca} -> {'PROMOTED' if ok_f1 and ok_ca else 'not promoted'}")
log.save_json("results.json", {"model1_variant": M1, "model1_test": {k: r1[k] for k in ("TP", "FP", "FN", "TN", "precision", "recall", "f1")},
                               "model2_cv": {v: {"mean": round(res[v].mean(), 4), "std": round(res[v].std(), 4),
                                                 "faults_rejected": round(res[v + "_rejected"].mean(), 4)} for v in VARIANTS},
                               "cascade_test": cascade, "promotion": promotion})
log.close()
