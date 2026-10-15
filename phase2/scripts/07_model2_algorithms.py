"""Step 07 - Is CatBoost the right algorithm for Model 2? A fair algorithm comparison.

All methods: 935 REAL training fault windows, correctly parsed signals (raw_kpis.npy),
the same 5 seeds x 5 session folds as the reproduced baseline (step 01), macro-F1.
Tabular models get the feature table [Config A | FFT | G1 G2 G3] (240 dims); time-series
models read the raw (16 x 128) windows directly.

    catboost_A        CatBoost, Config A only (reference, = baseline recipe on fixed parse)
    catboost_rich     CatBoost on the 240-dim table
    lightgbm_rich     LightGBM (class_weight=balanced)
    xgboost_rich      XGBoost (balanced sample weights)
    extratrees_rich   ExtraTrees (class_weight=balanced, 500 trees)
    catboost_c22      CatBoost on Config A + catch22 (16 channels x 22)
    minirocket        MiniRocket transform + RidgeClassifierCV (balanced)
    hydra             MultiRocket-Hydra classifier (aeon)
    quant             QUANT classifier (aeon)
Method source: research survey (agent_out/research_telecomts_methods.md), Middlehurst et al.
"Bake off redux" (2024). Each method is compared fold-by-fold with the Phase 1 baseline
(A_phase1 = 0.2908) using the Nadeau-Bengio corrected t-test.

Run from the phase2 folder:   python scripts/07_model2_algorithms.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/.kpi_rag_vendor"))          # isolated aeon install

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config as C
from src import features as F
from src.data import load_splits
from src.evaluation import nadeau_bengio_ttest
from src.models import balanced_class_weights, fault_classifier
from src.utils import StepLog
from src.v3 import fft_feats

log = StepLog("07_model2_algorithms")
d = load_splits()
fault = d["train_y"] != C.NORMAL_LABEL
y = d["train_y"][fault]
groups = d["train_sessions"][fault]
rows = d["train_idx"][fault]
raw = np.load(C.RAW_KPIS, mmap_mode="r")
S = np.asarray(raw[rows], dtype=np.float64)                         # (935, 16, 128)
A = np.stack([np.stack([x.mean(1), x.std(1), x.min(1), x.max(1)], 1).reshape(-1) for x in S])
FFT = np.stack([fft_feats(x) for x in S])
G = F.from_signals(S)
RICH = np.hstack([A, FFT, G])

t0 = time.time()
import pycatch22
C22 = np.stack([np.concatenate([pycatch22.catch22_all(list(ch))["values"] for ch in x]) for x in S])
C22 = np.nan_to_num(C22)
log(f"windows {len(y)} | A {A.shape[1]} | rich {RICH.shape[1]} | catch22 {C22.shape[1]} ({time.time()-t0:.0f}s)")

enc = LabelEncoder().fit(y)
yi = enc.transform(y)


def sample_w(ytr):
    w = balanced_class_weights(ytr)
    return np.array([w[c] for c in ytr])


def run(name, tr, va, seed):
    if name == "catboost_A":
        m = fault_classifier(seed, class_weights=balanced_class_weights(y[tr])); m.fit(A[tr], y[tr])
        return m.predict(A[va]).ravel()
    if name == "catboost_rich":
        m = fault_classifier(seed, class_weights=balanced_class_weights(y[tr])); m.fit(RICH[tr], y[tr])
        return m.predict(RICH[va]).ravel()
    if name == "catboost_c22":
        X = np.hstack([A, C22])
        m = fault_classifier(seed, class_weights=balanced_class_weights(y[tr])); m.fit(X[tr], y[tr])
        return m.predict(X[va]).ravel()
    if name == "lightgbm_rich":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=8, verbose=-1)
        m.fit(RICH[tr], y[tr]); return m.predict(RICH[va])
    if name == "xgboost_rich":
        import xgboost as xgb
        m = xgb.XGBClassifier(random_state=seed, n_jobs=8, tree_method="hist")
        m.fit(RICH[tr], yi[tr], sample_weight=sample_w(y[tr])); return enc.inverse_transform(m.predict(RICH[va]))
    if name == "extratrees_rich":
        m = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=16)
        m.fit(RICH[tr], y[tr]); return m.predict(RICH[va])
    if name == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        tf = MiniRocket(random_state=seed, n_jobs=16).fit(S[tr])
        sc = StandardScaler().fit(tf.transform(S[tr]))
        Ztr, Zva = sc.transform(tf.transform(S[tr])), sc.transform(tf.transform(S[va]))
        m = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced").fit(Ztr, y[tr])
        return m.predict(Zva)
    if name == "hydra":
        from aeon.classification.convolution_based import MultiRocketHydraClassifier
        m = MultiRocketHydraClassifier(random_state=seed, n_jobs=16).fit(S[tr], y[tr]); return m.predict(S[va])
    if name == "quant":
        from aeon.classification.interval_based import QUANTClassifier
        est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy",
                                   class_weight="balanced", random_state=seed, n_jobs=16)
        m = QUANTClassifier(estimator=est, random_state=seed).fit(S[tr], y[tr]); return m.predict(S[va])
    raise ValueError(name)


METHODS = ["catboost_A", "catboost_rich", "lightgbm_rich", "xgboost_rich", "extratrees_rich",
           "catboost_c22", "minirocket", "hydra", "quant"]
base = pd.read_csv(os.path.join(C.RESULTS, "01_baseline", "baseline_folds.csv"))

ckpt = log.path("folds_partial.csv")
out = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["method"], r["seed"], r["fold"]) for r in out}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(A, y, groups=groups), start=1):
        line = []
        for mname in METHODS:
            if (mname, seed, fold) in done:
                continue
            t = time.time()
            pred = run(mname, tr, va, seed)
            f1 = f1_score(y[va], pred, average="macro", zero_division=0)
            out.append({"method": mname, "seed": seed, "fold": fold, "f1": f1,
                        "sec": round(time.time() - t, 1), "n_train": len(tr), "n_val": len(va)})
            line.append(f"{mname}={f1:.3f}")
        pd.DataFrame(out).to_csv(ckpt, index=False)
        if line:
            log(f"  seed={seed:<5} fold={fold}  " + "  ".join(line))

res = pd.DataFrame(out)
res.to_csv(log.path("folds.csv"), index=False)
ref = base.sort_values(["seed", "fold"])
summary, stats = [], {}
for mname in METHODS:
    r = res[res.method == mname].sort_values(["seed", "fold"])
    diff, t_stat, p, w = nadeau_bengio_ttest(r.f1.values - ref.f1_macro.values, ref.n_train.values, ref.n_val.values)
    wins = int((r.f1.values > ref.f1_macro.values).sum())
    stats[mname] = {"mean": round(r.f1.mean(), 4), "std": round(r.f1.std(), 4), "d_vs_phase1": round(diff, 4),
                    "p_corrected": round(p, 4), "wins": wins, "sec_per_fold": round(r.sec.mean(), 1)}
    summary.append((mname, r.f1.mean(), r.f1.std(), diff, p, wins, r.sec.mean()))
log("\n== Macro-F1 over 25 folds vs Phase 1 baseline (0.2908) ==")
for m, mu, sd, dd, p, w, s in sorted(summary, key=lambda t: -t[1]):
    log(f"  {m:16s} {mu:.4f} ± {sd:.4f}   d={dd:+.4f}  p={p:.4f}  wins {w}/25   {s:.0f}s/fold")
log.save_json("results.json", stats)
log.close()
