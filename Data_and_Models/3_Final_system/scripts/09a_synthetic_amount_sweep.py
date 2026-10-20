"""Step 09a - Does MORE synthetic data help? Extends the cell-23 sweep of the synthetic-data notebook beyond 500/class.

The original sweep (26-Sep): 100 -> 500 windows per class, macro-F1 still rising at 500
(0.329 -> 0.351, CatBoost, Config A). Here: 500 (the synthetic-data model), 1000, 2000 per class, with the
step-06 setting (v3 + Config B + FFT + trend features) and three algorithms:
CatBoost (the synthetic-data model), LightGBM, QUANT (raw windows).
Screening design as in the original: seeds [42, 7] x 5 session folds = 10 folds. Within a fold the
synthetic windows are generated ONCE at 2000/class and truncated to 500 / 1000 per class
(the generator produces windows class by class in sequence, so the first N per class are
exactly what N_TARGET = N would produce).
If an amount wins, step 09b confirms it on all 25 folds.

Run from the phase2 folder:   python scripts/09a_synthetic_amount_sweep.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/.kpi_rag_vendor"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src import v3 as V
from src.data import load_splits
from src.evaluation import nadeau_bengio_ttest
from src.utils import StepLog
from src.v3 import IDX_B_FFT, V3, fit_predict_w, full_feats, macro

AMOUNTS = [500, 1000, 2000]
SEEDS = [42, 7]
ALGOS = ["catboost", "lightgbm", "quant"]
V.N_TARGET = max(AMOUNTS)

log = StepLog("09a_synthetic_amount_sweep")
d = load_splits()
fault = d["train_y"] != C.NORMAL_LABEL
real_rows, real_y = d["train_idx"][fault], d["train_y"][fault]
real_sess = d["session_ids"][real_rows]
eng = V3(d)                                        # same parser as the synthetic-data model
eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])
full = np.stack([full_feats(x) for x in sig])
G = F.from_signals(sig)
A_ref = full[:, :64]
host_sess = d["session_ids"][eng.host_rows]
table = lambda fullX, GX: np.hstack([fullX[:, IDX_B_FFT], GX])


def generate(tr, va, seed, f_i):
    val_sess = set(real_sess[va].tolist())
    hosts_ok = eng.host_rows[~np.isin(host_sess, list(val_sess))]
    ch_min = np.minimum(eng.NORMAL_MIN, A_ref[tr][:, 2::4].min(0))
    ch_max = np.maximum(eng.NORMAL_MAX, A_ref[tr][:, 3::4].max(0))
    eff = eng.estimate_effects(real_rows[tr], hosts_ok)
    raw = []
    Xs, Ys = eng.generate(eff, real_rows[tr], hosts_ok, np.random.default_rng(seed * 100 + f_i), ch_min, ch_max,
                          lambda x: (raw.append(x.copy()), full_feats(x))[1])
    return Xs, np.stack(raw), Ys


def take(Ys, amount, n_real_per_class):
    """Indices of the first (amount - n_real) synthetic windows of each class, in generation order."""
    keep = np.zeros(len(Ys), bool)
    for cls in np.unique(Ys):
        idx = np.where(Ys == cls)[0]
        keep[idx[:max(0, amount - n_real_per_class[cls])]] = True
    return keep


def fit_predict(algo, Xtr, ytr, Xva, Str, Sva, seed):
    if algo == "catboost":
        return fit_predict_w(Xtr, ytr, np.ones(len(ytr)), Xva, seed)[1]
    if algo == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=16, verbose=-1).fit(Xtr, ytr).predict(Xva)
    if algo == "quant":
        from aeon.classification.interval_based import QUANTClassifier
        est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy",
                                   class_weight="balanced", random_state=seed, n_jobs=8)
        return QUANTClassifier(estimator=est, random_state=seed).fit(Str, ytr).predict(Sva)


ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for f_i, (tr, va) in enumerate(cv.split(A_ref, real_y, groups=real_sess)):
        if (seed, f_i + 1) in done:
            continue
        t0 = time.time()
        Xs, Ss, Ys = generate(tr, va, seed, f_i)
        Gs = F.from_signals(Ss)
        n_real = {c: int((real_y[tr] == c).sum()) for c in np.unique(real_y[tr])}
        rec = {"seed": seed, "fold": f_i + 1, "n_train": len(tr), "n_val": len(va), "n_syn_2000": len(Ys)}
        for amount in AMOUNTS:
            k = take(Ys, amount, n_real)
            Xtr = np.vstack([table(full[tr], G[tr]), table(Xs[k], Gs[k])])
            ytr = np.concatenate([real_y[tr], Ys[k]])
            Str = np.concatenate([sig[tr], Ss[k]])
            for algo in ALGOS:
                t = time.time()
                rec[f"{algo}@{amount}"] = macro(real_y[va], fit_predict(algo, Xtr, ytr, table(full[va], G[va]), Str, sig[va], seed))
                log(f"    seed={seed} fold={f_i+1} {algo}@{amount}: {rec[f'{algo}@{amount}']:.4f} "
                    f"(train rows {len(ytr)}, {time.time()-t:.0f}s)")
            rec[f"n_syn@{amount}"] = int(k.sum())
        rows.append(rec)
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        log(f"  seed={seed:<5} fold={f_i+1}  " + "  ".join(f"{a}@{m}={rec[f'{a}@{m}']:.3f}" for m in AMOUNTS for a in ALGOS)
            + f"  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows).sort_values(["seed", "fold"])
res.to_csv(log.path("folds.csv"), index=False)
log("\n== Macro-F1, mean over 10 folds (rows: algorithm, columns: windows per class) ==")
tab = pd.DataFrame({m: {a: res[f"{a}@{m}"].mean() for a in ALGOS} for m in AMOUNTS})
log(tab.round(4).to_string())
log("\n== Paired vs 500/class (same algorithm, Nadeau-Bengio corrected t-test, 10 folds) ==")
stats = {}
for a in ALGOS:
    for m in AMOUNTS[1:]:
        diff, t_stat, p, w = nadeau_bengio_ttest(res[f"{a}@{m}"] - res[f"{a}@500"], res.n_train, res.n_val)
        wins = int((res[f"{a}@{m}"] > res[f"{a}@500"]).sum())
        stats[f"{a}@{m}_vs_500"] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4), "wins": wins}
        log(f"  {a:9s} {m:>5}/class vs 500: d={diff:+.4f}  p={p:.3f}  wins {wins}/10")
log.save_json("results.json", {"amounts": AMOUNTS, "means": tab.round(4).to_dict(), "paired": stats})
log.close()
