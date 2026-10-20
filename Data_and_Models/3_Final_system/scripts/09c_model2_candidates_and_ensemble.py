"""Step 09c - Final Model 2 candidates, measured for real: CV + held-out sessions + per class.

Candidates (all on the same 25 session folds; synthetic windows from the step-09z cache):
    catboost_v3      the synthetic-data model: real + v3 (500/class) + Config B + FFT, CatBoost
    catboost_v3_G    the same + G1 G2 G3 trend features (step 06 "trend")
    quant_real       QUANT on the raw real windows only (no synthetic data)
    catboost_c22     CatBoost on Config A + catch22, real only (step 07 runner-up)
    vote_2           equal-weight soft vote of catboost_v3_G + quant_real
    vote_3           equal-weight soft vote of catboost_v3_G + quant_real + catboost_c22
Soft votes average class probabilities (no weights learned from the outer folds => no leak).
Decision rule (STATUS.md): a candidate replaces the synthetic-data model only if corrected p < 0.05
AND >= 0.03 macro-F1 in CV AND it holds on the held-out sessions.
Real results written: per-class precision/recall/F1 (CV pooled and test), confusion matrices
(CSV), per-session test accuracy, and results.json.

Run from the phase2 folder:   python scripts/09c_model2_candidates_and_ensemble.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/.kpi_rag_vendor"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import csv_labels, load_splits
from src.evaluation import nadeau_bengio_ttest
from src.models import balanced_class_weights, fault_classifier
from src.utils import StepLog
from src.synthetic_data import IDX_B_FFT, V3, fit_predict_w, full_feats, macro, synthetic_windows

log = StepLog("09c_model2_candidates")
d = load_splits()
fault = d["train_y"] != C.NORMAL_LABEL
real_rows, real_y = d["train_idx"][fault], d["train_y"][fault]
real_sess = d["session_ids"][real_rows]
eng = V3(d)
eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])
full = np.stack([full_feats(x) for x in sig])
G = F.from_signals(sig)
A_ref = full[:, :64]
host_sess = d["session_ids"][eng.host_rows]
CLASSES = sorted(np.unique(real_y))

import pycatch22
c22 = lambda S: np.nan_to_num(np.stack([np.concatenate([pycatch22.catch22_all(list(ch))["values"] for ch in x]) for x in S]))
C22 = c22(sig)
log(f"real fault windows {len(real_y)} | B+FFT {len(IDX_B_FFT)} | +G {len(IDX_B_FFT)+96} | A+catch22 {64+C22.shape[1]}")


def proba_catboost(Xtr, ytr, Xva, seed):
    clf, _ = fit_predict_w(Xtr, ytr, np.ones(len(ytr)), Xva, seed)
    p = clf.predict_proba(Xva)
    return pd.DataFrame(p, columns=[str(c) for c in clf.classes_]).reindex(columns=CLASSES, fill_value=0).values


def proba_quant(Str, ytr, Sva, seed):
    from aeon.classification.interval_based import QUANTClassifier
    est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy",
                               class_weight="balanced", random_state=seed, n_jobs=8)
    m = QUANTClassifier(estimator=est, random_state=seed).fit(Str, ytr)
    return pd.DataFrame(m.predict_proba(Sva), columns=[str(c) for c in m.classes_]).reindex(columns=CLASSES, fill_value=0).values


def proba_cb_c22(Xtr, ytr, Xva, seed):
    m = fault_classifier(seed, class_weights=balanced_class_weights(ytr)).fit(Xtr, ytr)
    return pd.DataFrame(m.predict_proba(Xva), columns=[str(c) for c in m.classes_]).reindex(columns=CLASSES, fill_value=0).values


def all_probas(tr_idx, va_idx, seed, fold, S_va, X_va_full, G_va, C22_va, ytr_real, is_test=False):
    """Class-probability matrices of the 4 base models for the validation/test rows."""
    S_syn, Y_syn = synthetic_windows(eng, real_rows, real_sess, A_ref, host_sess, tr_idx,
                                     va_idx if not is_test else np.array([], int), seed, fold) \
        if not is_test else synthetic_windows_all(seed)
    X_syn = np.stack([full_feats(x) for x in S_syn])
    G_syn = F.from_signals(S_syn)
    ytr = np.concatenate([ytr_real, Y_syn])
    Xb = np.vstack([full[tr_idx][:, IDX_B_FFT], X_syn[:, IDX_B_FFT]])
    Xg = np.hstack([Xb, np.vstack([G[tr_idx], G_syn])])
    P = {}
    P["catboost_v3"] = proba_catboost(Xb, ytr, X_va_full[:, IDX_B_FFT], seed)
    P["catboost_v3_G"] = proba_catboost(Xg, ytr, np.hstack([X_va_full[:, IDX_B_FFT], G_va]), seed)
    P["quant_real"] = proba_quant(sig[tr_idx], ytr_real, S_va, seed)
    P["catboost_c22"] = proba_cb_c22(np.hstack([A_ref[tr_idx], C22[tr_idx]]), ytr_real, C22_va, seed)
    P["vote_2"] = (P["catboost_v3_G"] + P["quant_real"]) / 2
    P["vote_3"] = (P["catboost_v3_G"] + P["quant_real"] + P["catboost_c22"]) / 3
    return P


def synthetic_windows_all(seed):
    """Synthetic windows for the FULL training partition (post-hoc test model), cached."""
    path = os.path.join(C.RESULTS, "cache", "synth", f"all_s{seed}_n500_bug.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True); return z["S"], z["Y"]
    ch_min = np.minimum(eng.NORMAL_MIN, A_ref[:, 2::4].min(0)); ch_max = np.maximum(eng.NORMAL_MAX, A_ref[:, 3::4].max(0))
    eff = eng.estimate_effects(real_rows, eng.host_rows)
    raw = []
    _, Y = eng.generate(eff, real_rows, eng.host_rows, np.random.default_rng(seed), ch_min, ch_max,
                        lambda x: (raw.append(x.copy()), np.zeros(1))[1])
    S = np.stack(raw); np.savez_compressed(path, S=S, Y=Y); return S, Y


CANDS = ["catboost_v3", "catboost_v3_G", "quant_real", "catboost_c22", "vote_2", "vote_3"]
ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
oof = {c: {s: np.empty(len(real_y), dtype=object) for s in C.CV_SEEDS} for c in CANDS}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for f_i, (tr, va) in enumerate(cv.split(A_ref, real_y, groups=real_sess)):
        t0 = time.time()
        P = all_probas(tr, va, seed, f_i + 1, sig[va], full[va], G[va], np.hstack([A_ref[va], C22[va]]), real_y[tr])
        rec = {"seed": seed, "fold": f_i + 1, "n_train": len(tr), "n_val": len(va)}
        for c in CANDS:
            pred = np.array(CLASSES)[P[c].argmax(1)]
            oof[c][seed][va] = pred
            rec[c] = macro(real_y[va], pred)
        if (seed, f_i + 1) not in done:
            rows.append(rec); pd.DataFrame(rows).to_csv(ckpt, index=False)
        log(f"  seed={seed:<5} fold={f_i+1}  " + "  ".join(f"{c}={rec[c]:.3f}" for c in CANDS) + f"  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows).sort_values(["seed", "fold"])
res.to_csv(log.path("folds.csv"), index=False)
log("\n== CV macro-F1 over 25 session folds ==")
log(res[CANDS].agg(["mean", "std"]).T.round(4).to_string())
stats = {}
log("\n== Paired vs the synthetic-data model (catboost_v3), Nadeau-Bengio corrected t-test ==")
for c in CANDS[1:]:
    diff, t_stat, p, w = nadeau_bengio_ttest(res[c] - res["catboost_v3"], res.n_train, res.n_val)
    wins = int((res[c] > res["catboost_v3"]).sum())
    ok = diff >= 0.03 and p < 0.05
    stats[c] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4), "p_wilcoxon": round(w, 4),
                "wins": wins, "meets_rule": bool(ok)}
    log(f"  {c:14s} d={diff:+.4f}  p={p:.4f}  Wilcoxon p={w:.4f}  wins {wins}/25  -> {'MEETS RULE' if ok else 'no'}")

# per-class CV (pooled out-of-fold predictions, all seeds)
pc = {}
for c in CANDS:
    yt = np.concatenate([real_y] * len(C.CV_SEEDS)); yp = np.concatenate([oof[c][s] for s in C.CV_SEEDS])
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
    pc[c] = f1
per_class = pd.DataFrame(pc, index=CLASSES); per_class["support"] = [int((real_y == k).sum()) for k in CLASSES]
per_class.round(3).to_csv(log.path("per_class_f1_cv.csv"))
log("\n== Per-class F1, pooled out-of-fold (CV) ==")
log(per_class.round(3).to_string())

# ---- held-out sessions (post-hoc; reported, not used to decide) --------------------------------
log("\n== HELD-OUT TEST SESSIONS (post-hoc) ==")
labels = csv_labels()
te_mask = labels[d["test_idx"]] != C.NORMAL_LABEL
te_rows, y_te = d["test_idx"][te_mask], labels[d["test_idx"]][te_mask]
te_sess = d["session_ids"][te_rows]
sig_te = np.stack([eng.real_sig(r) for r in te_rows])
full_te, G_te, C22_te = np.stack([full_feats(x) for x in sig_te]), F.from_signals(sig_te), c22(sig_te)
P = all_probas(np.arange(len(real_y)), None, C.SEED, 0, sig_te, full_te, G_te, np.hstack([full_te[:, :64], C22_te]), real_y, is_test=True)
test = {}
for c in CANDS:
    pred = np.array(CLASSES)[P[c].argmax(1)]
    pr, rc, f1, sup = precision_recall_fscore_support(y_te, pred, labels=CLASSES, zero_division=0)
    per_sess = {int(s): round(float((pred[te_sess == s] == y_te[te_sess == s]).mean()), 3) for s in np.unique(te_sess)}
    test[c] = {"macro_f1": round(macro(y_te, pred), 4), "accuracy": round(accuracy_score(y_te, pred), 4),
               "per_session_accuracy": per_sess, "per_class_f1": dict(zip(CLASSES, np.round(f1, 3)))}
    pd.DataFrame(confusion_matrix(y_te, pred, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(log.path(f"confusion_test_{c}.csv"))
    log(f"  {c:14s} macro-F1={test[c]['macro_f1']:.4f}  accuracy={test[c]['accuracy']:.4f}  per-session acc={per_sess}")
log("  reference: Phase 1 accuracy 0.677; the synthetic-data report real-only 0.586/0.705, v3+B+FFT 0.557/0.680")
pd.DataFrame({c: test[c]["per_class_f1"] for c in CANDS}).round(3).to_csv(log.path("per_class_f1_test.csv"))

log.save_json("results.json", {"cv_means": res[CANDS].mean().round(4).to_dict(), "cv_std": res[CANDS].std().round(4).to_dict(),
                               "paired_vs_catboost_v3": stats, "test": test})
log.close()
