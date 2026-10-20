"""Step 12b - Specialist classifier for the most confused pair: HNC-Gradual vs Buffer Overflow.

Problem: the final Model 2 predicts "Buffer Overflow" for 30 of the 37 HNC-Gradual test windows
(pooled OOF: HNC-Gradual F1 = 0.00). Both faults are gradual congestion build-ups.
Hypothesis (physics, TelecomTS paper Table 10 / the synthetic-data fault recipes): in Buffer Overflow the UL buffer
(Estimated_UL_Buffer) grows while PRB utilisation stays flat (the link cannot drain); in HNC-Gradual PRB
utilisation and packet counts grow with the buffer (more users). A 2-class model trained ONLY on these two
classes, with growth-shape features of those channels, should separate them better than the 11-class model.
Design (nested, 25 session folds, synthetic cache): main Model 2 = final recipe + trend; for validation
windows the main model labels as either class, the specialist re-decides. Metrics: pair F1 (both classes),
overall 11-class macro-F1, HNC-G -> BO confusion. Rule: adopt only if pair F1 improves consistently
(>= 18/25 folds) AND overall macro-F1 is not lower AND it holds on the held-out sessions (post-hoc).

Run from the phase2 folder:   python scripts/12b_pair_specialist.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, load_splits
from src.utils import StepLog
from src.v3 import IDX_B_FFT, V3, fit_predict_w, full_feats, macro, synthetic_windows

HNC, BO = "High Network Congestion (Gradual Buildup)", "Buffer Overflow (Gradual Buildup)"
PAIR = [HNC, BO]
CH = {"Estimated_UL_Buffer": 9, "PRB_Utilization_DL": 12, "PRB_Utilization_UL": 13, "UL_NPRB": 5,
      "UL_NumberOfPackets": 14, "DL_NumberOfPackets": 15, "TX_Bytes": 7, "DL_BLER": 1, "UL_BLER": 3}


def pair_feats(S):
    """Growth-shape features of the congestion channels: slope, late/early ratio, time of max, max/mean,
    buffer-growth vs PRB-growth ratio. S: (n,16,128)."""
    t = np.linspace(-1, 1, S.shape[2])
    out = []
    for name, c in CH.items():
        x = S[:, c, :]
        m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True) + 1e-8
        z = (x - m) / s
        slope = (z * t).sum(1) / (t * t).sum()
        early, late = x[:, :32].mean(1), x[:, -32:].mean(1)
        ratio = np.log((late + 1e-6) / (early + 1e-6))
        tmax = x.argmax(1) / S.shape[2]
        peak = x.max(1) / (m[:, 0] + 1e-6)
        out += [slope, ratio, tmax, peak]
    out = np.column_stack(out)
    buf = out[:, 1]; prb = out[:, 5]                      # late/early log-ratios: buffer vs PRB_Util_DL
    pk = out[:, 17]                                        # UL packets ratio
    return np.column_stack([out, buf - prb, buf - pk, prb - pk])


log = StepLog("12b_pair_specialist")
d = load_splits(); labels = csv_labels()
fault = d["train_y"] != C.NORMAL_LABEL
real_rows, real_y, real_sess = d["train_idx"][fault], d["train_y"][fault], d["train_sessions"][fault]
eng = V3(d); eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])
full = np.stack([full_feats(x) for x in sig]); Gf = F.from_signals(sig); A_ref = full[:, :64]
host_sess = d["session_ids"][eng.host_rows]
tbl = lambda fullX, GX: np.hstack([fullX[:, IDX_B_FFT], GX])
X2 = tbl(full, Gf); PF = np.hstack([pair_feats(sig), full[:, :64]])
CLASSES = sorted(np.unique(real_y))


def models(tr, va_sess, seed, fold, for_test=False):
    if for_test:
        z = np.load(os.path.join(C.RESULTS, "cache", "synth", f"all_s{seed}_n500_bug.npz"), allow_pickle=True); S_syn, Y_syn = z["S"], z["Y"]
    else:
        S_syn, Y_syn = synthetic_windows(eng, real_rows, real_sess, A_ref, host_sess, tr, None, seed, fold)
    Xs = tbl(np.stack([full_feats(x) for x in S_syn]), F.from_signals(S_syn))
    Xm, ym = np.vstack([X2[tr], Xs]), np.concatenate([real_y[tr], Y_syn])
    main, _ = fit_predict_w(Xm, ym, np.ones(len(ym)), Xm[:1], seed)
    pm = np.isin(real_y[tr], PAIR); ps = np.isin(Y_syn, PAIR)
    Xp = np.vstack([PF[tr][pm], np.hstack([pair_feats(S_syn[ps]), np.stack([full_feats(x) for x in S_syn[ps]])[:, :64]])])
    yp = np.concatenate([real_y[tr][pm], Y_syn[ps]])
    spec, _ = fit_predict_w(Xp, yp, np.ones(len(yp)), Xp[:1], seed)
    return main, spec


def cascade(main, spec, Xva, PFva):
    pred = main.predict(Xva).ravel().astype(str)
    m = np.isin(pred, PAIR)
    pred2 = pred.copy()
    if m.any():
        pred2[m] = spec.predict(PFva[m]).ravel().astype(str)
    return pred, pred2


def scores(y, p):
    return {"macro_f1": macro(y, p), "pair_f1": f1_score(y, p, labels=PAIR, average="macro", zero_division=0),
            "hnc_to_bo": int(((y == HNC) & (p == BO)).sum()), "hnc_n": int((y == HNC).sum())}


ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for f_i, (tr, va) in enumerate(cv.split(A_ref, real_y, groups=real_sess)):
        if (seed, f_i + 1) in done:
            continue
        t0 = time.time()
        main, spec = models(tr, set(real_sess[va].tolist()), seed, f_i + 1)
        p0, p1 = cascade(main, spec, X2[va], PF[va])
        s0, s1 = scores(real_y[va], p0), scores(real_y[va], p1)
        rec = {"seed": seed, "fold": f_i + 1, "n_train": len(tr), "n_val": len(va), **{f"main_{k}": v for k, v in s0.items()}, **{f"spec_{k}": v for k, v in s1.items()}}
        rows.append(rec); pd.DataFrame(rows).to_csv(ckpt, index=False)
        log(f"  seed={seed:<5} fold={f_i+1}  macro-F1 {s0['macro_f1']:.3f} -> {s1['macro_f1']:.3f}  pair-F1 {s0['pair_f1']:.3f} -> {s1['pair_f1']:.3f}  "
            f"HNC->BO {s0['hnc_to_bo']}/{s0['hnc_n']} -> {s1['hnc_to_bo']}/{s1['hnc_n']}  ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows); res.to_csv(log.path("folds.csv"), index=False)
log("\n== Mean over 25 folds ==")
for k in ("macro_f1", "pair_f1"):
    better = int((res[f"spec_{k}"] > res[f"main_{k}"]).sum())
    log(f"  {k:9s} main {res[f'main_{k}'].mean():.4f} -> specialist {res[f'spec_{k}'].mean():.4f}  (better in {better}/25 folds)")
log(f"  HNC-G predicted as BO: {res.main_hnc_to_bo.sum()}/{res.main_hnc_n.sum()} -> {res.spec_hnc_to_bo.sum()}/{res.spec_hnc_n.sum()} (pooled)")
adopt = int((res.spec_pair_f1 > res.main_pair_f1).sum()) >= 18 and res.spec_macro_f1.mean() >= res.main_macro_f1.mean()

log("\n== POST-HOC held-out sessions ==")
te_mask = labels[d["test_idx"]] != C.NORMAL_LABEL
te_rows, y_te = d["test_idx"][te_mask], labels[d["test_idx"]][te_mask]
eng._load_text([r for r in te_rows if int(r) not in eng.ROW2IDX])
S_te = np.stack([eng.real_sig(r) for r in te_rows])
full_te = np.stack([full_feats(x) for x in S_te])
main, spec = models(np.arange(len(real_y)), set(), C.SEED, 0, for_test=True)
p0, p1 = cascade(main, spec, tbl(full_te, F.from_signals(S_te)), np.hstack([pair_feats(S_te), full_te[:, :64]]))
t0s, t1s = scores(y_te, p0), scores(y_te, p1)
log(f"  main:       macro-F1 {t0s['macro_f1']:.4f}  acc {(p0 == y_te).mean():.4f}  pair-F1 {t0s['pair_f1']:.3f}  HNC->BO {t0s['hnc_to_bo']}/{t0s['hnc_n']}")
log(f"  specialist: macro-F1 {t1s['macro_f1']:.4f}  acc {(p1 == y_te).mean():.4f}  pair-F1 {t1s['pair_f1']:.3f}  HNC->BO {t1s['hnc_to_bo']}/{t1s['hnc_n']}")
holds = t1s["pair_f1"] >= t0s["pair_f1"] and t1s["macro_f1"] >= t0s["macro_f1"] - 0.005
log(f"\nDecision rule: CV consistent = {adopt}, holds on test = {holds} -> {'ADOPT' if adopt and holds else 'not adopted'}")
log.save_json("results.json", {"cv": {k: round(res[k].mean(), 4) for k in res.columns if k.startswith(("main_", "spec_"))},
                               "test": {"main": t0s, "specialist": t1s, "acc_main": round(float((p0 == y_te).mean()), 4), "acc_spec": round(float((p1 == y_te).mean()), 4)},
                               "adopt": bool(adopt and holds)})
log.close()
