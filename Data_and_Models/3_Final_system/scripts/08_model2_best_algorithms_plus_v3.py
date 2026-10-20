"""Step 08 - Combine the two best levers for Model 2: the best algorithms (step 07)
trained on real + the v3 synthetic data (step 06 setting).

Settings are read from the previous steps' results, so this script is deterministic:
  - from results/06: the best v3 variant (final / excl_pair / trend / trend_fixed) decides
    whether the pair HNC-Gradual + Buffer Overflow is excluded from augmentation, whether
    the G1 G2 G3 trend features are added, and which parser is used.
  - from results/07: the top-3 algorithms by mean macro-F1 (CatBoost is always included
    as the reference = the synthetic-data model).
Protocol unchanged: 935 real fault windows, 5 seeds x 5 SESSION folds, synthetic windows
rebuilt from training-fold sessions only, validation = real windows only, macro-F1,
Nadeau-Bengio corrected t-test vs the synthetic-data model and vs the Phase 1 baseline.
One post-hoc test check (test fault windows of the 3 held-out sessions).

Run from the phase2 folder:   python scripts/08_model2_best_algorithms_plus_v3.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/.kpi_rag_vendor"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config as C
from src import features as F
from src.data import csv_labels, load_splits
from src.evaluation import nadeau_bengio_ttest
from src.models import balanced_class_weights
from src.utils import StepLog
from src.v3 import IDX_B_FFT, V3, fit_predict_w, full_feats, macro

PAIR = ("High Network Congestion (Gradual Buildup)", "Buffer Overflow (Gradual Buildup)")
log = StepLog("08_model2_best_plus_v3")

# ---- settings from steps 06 and 07 ---------------------------------------------------------
r06 = json.load(open(os.path.join(C.RESULTS, "06_model2_synthetic_data_experiments", "results.json")))
means06 = {k: v for k, v in r06["means"].items() if k != "real_only"}
best06 = max(means06, key=means06.get)
USE_G = best06 in ("trend", "trend_fixed")
FIXED = best06 == "trend_fixed"
EXCLUDE = PAIR if best06 == "excl_pair" else ()
r07 = json.load(open(os.path.join(C.RESULTS, "07_model2_algorithms", "results.json")))
ranked = sorted(r07, key=lambda k: -r07[k]["mean"])
ALGOS = ["catboost"] + [a.replace("_rich", "") for a in ranked if a not in ("catboost_A", "catboost_rich")][:3]
log(f"step 06 best variant: {best06} ({means06[best06]:.4f}) -> G features={USE_G}, fixed parser={FIXED}, "
    f"exclude pair={bool(EXCLUDE)}")
log(f"step 07 ranking: {[(a, r07[a]['mean']) for a in ranked]}")
log(f"algorithms in step 08: {ALGOS}")

# ---- data ---------------------------------------------------------------------------------
d = load_splits()
fault = d["train_y"] != C.NORMAL_LABEL
real_rows, real_y = d["train_idx"][fault], d["train_y"][fault]
real_X = d["train_X"][fault][:, :C.N_FEATURES_A]
real_sess = d["session_ids"][real_rows]
eng = V3(d, fixed_parse=FIXED)
eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])
full = np.stack([full_feats(x) for x in sig])
G = F.from_signals(sig)
A_ref = full[:, :64]
host_sess = d["session_ids"][eng.host_rows]
enc = LabelEncoder().fit(real_y)


def table(fullX, GX):
    return np.hstack([fullX[:, IDX_B_FFT], GX]) if USE_G else fullX[:, IDX_B_FFT]


def synth(tr, va, seed, f_i):
    val_sess = set(real_sess[va].tolist())
    hosts_ok = eng.host_rows[~np.isin(host_sess, list(val_sess))]
    ch_min = np.minimum(eng.NORMAL_MIN, A_ref[tr][:, 2::4].min(0))
    ch_max = np.maximum(eng.NORMAL_MAX, A_ref[tr][:, 3::4].max(0))
    eff = eng.estimate_effects(real_rows[tr], hosts_ok, exclude=EXCLUDE)
    raw = []
    feat = lambda x: (raw.append(x.copy()), full_feats(x))[1]
    Xs, Ys = eng.generate(eff, real_rows[tr], hosts_ok, np.random.default_rng(seed * 100 + f_i), ch_min, ch_max, feat)
    S = np.stack(raw) if raw else np.zeros((0, 16, 128))
    Gs = F.from_signals(S) if len(S) else np.zeros((0, 96))
    return Xs, Gs, S, Ys


def fit_predict(algo, Xtr, ytr, Xva, Str, Sva, seed):
    """Xtr/Xva: feature tables; Str/Sva: raw windows (for time-series models)."""
    if algo == "catboost":
        return fit_predict_w(Xtr, ytr, np.ones(len(ytr)), Xva, seed)[1]
    sw = np.array([balanced_class_weights(ytr)[c] for c in ytr])
    if algo == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=8, verbose=-1).fit(Xtr, ytr)
        return m.predict(Xva)
    if algo == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(random_state=seed, n_jobs=8, tree_method="hist")
        m.fit(Xtr, enc.transform(ytr), sample_weight=sw)
        return enc.inverse_transform(m.predict(Xva))
    if algo == "extratrees":
        return ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=16).fit(Xtr, ytr).predict(Xva)
    if algo == "catboost_c22":
        import pycatch22
        c22 = lambda S: np.nan_to_num(np.stack([np.concatenate([pycatch22.catch22_all(list(ch))["values"] for ch in x]) for x in S]))
        Xt, Xv = np.hstack([Xtr, c22(Str)]), np.hstack([Xva, c22(Sva)])
        return fit_predict_w(Xt, ytr, np.ones(len(ytr)), Xv, seed)[1]
    if algo == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        tf = MiniRocket(random_state=seed, n_jobs=16).fit(Str)
        sc = StandardScaler().fit(tf.transform(Str))
        m = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced").fit(sc.transform(tf.transform(Str)), ytr)
        return m.predict(sc.transform(tf.transform(Sva)))
    if algo == "hydra":
        from aeon.classification.convolution_based import MultiRocketHydraClassifier
        return MultiRocketHydraClassifier(random_state=seed, n_jobs=16).fit(Str, ytr).predict(Sva)
    if algo == "quant":
        from aeon.classification.interval_based import QUANTClassifier
        est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy",
                                   class_weight="balanced", random_state=seed, n_jobs=16)
        return QUANTClassifier(estimator=est, random_state=seed).fit(Str, ytr).predict(Sva)
    raise ValueError(algo)


# ---- nested CV --------------------------------------------------------------------------------
ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for f_i, (tr, va) in enumerate(cv.split(real_X, real_y, groups=real_sess)):
        if (seed, f_i + 1) in done:
            continue
        t0 = time.time()
        Xs, Gs, Ss, Ys = synth(tr, va, seed, f_i)
        Xtr = np.vstack([table(full[tr], G[tr]), table(Xs, Gs)])
        ytr = np.concatenate([real_y[tr], Ys])
        Str = np.concatenate([sig[tr], Ss]) if len(Ss) else sig[tr]
        Xva, Sva, yva = table(full[va], G[va]), sig[va], real_y[va]
        rec = {"seed": seed, "fold": f_i + 1, "n_train": len(tr), "n_val": len(va), "n_syn": len(Ys)}
        for algo in ALGOS:
            t = time.time()
            rec[algo] = macro(yva, fit_predict(algo, Xtr, ytr, Xva, Str, Sva, seed))
            rec[f"{algo}_sec"] = round(time.time() - t, 1)
        rows.append(rec)
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        log(f"  seed={seed:<5} fold={f_i+1}  syn={len(Ys)}  " + "  ".join(f"{a}={rec[a]:.4f}" for a in ALGOS)
            + f"  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows).sort_values(["seed", "fold"])
r06f = pd.read_csv(os.path.join(C.RESULTS, "06_model2_synthetic_data_experiments", "folds.csv")).sort_values(["seed", "fold"])
res = res.merge(r06f[["seed", "fold", "final", "real_only"]].rename(columns={"final": "synthetic_data_model"}), on=["seed", "fold"])
res.to_csv(log.path("folds.csv"), index=False)

log("\n== Macro-F1 over 25 folds ==")
cols = ["real_only", "synthetic_data_model"] + ALGOS
log(res[cols].agg(["mean", "std"]).T.round(4).to_string())
stats = {}
log("\n== Paired comparisons (Nadeau-Bengio corrected t-test) ==")
for ref in ("real_only", "synthetic_data_model"):
    for a in ALGOS:
        diff, t_stat, p, w = nadeau_bengio_ttest(res[a] - res[ref], res.n_train, res.n_val)
        wins = int((res[a] > res[ref]).sum())
        stats[f"{a}_vs_{ref}"] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4), "p_wilcoxon": round(w, 4), "wins": wins}
        log(f"  {a:12s} vs {ref:12s}: d={diff:+.4f}  p={p:.4f}  Wilcoxon p={w:.4f}  wins {wins}/25")
best = max(ALGOS, key=lambda a: res[a].mean())
log(f"\nBest in CV: {best} ({res[best].mean():.4f})")

# ---- one post-hoc test check (best config, all training data) ----------------------------------
log("\n== POST-HOC test check (3 held-out sessions, 300 fault windows; not used to decide) ==")
labels = csv_labels()
te_mask = labels[d["test_idx"]] != C.NORMAL_LABEL
te_rows, y_te = d["test_idx"][te_mask], labels[d["test_idx"]][te_mask]
sig_te = np.stack([eng.real_sig(r) for r in te_rows])
full_te, G_te = np.stack([full_feats(x) for x in sig_te]), F.from_signals(sig_te)
all_tr = np.arange(len(real_y))
ch_min = np.minimum(eng.NORMAL_MIN, A_ref[:, 2::4].min(0)); ch_max = np.maximum(eng.NORMAL_MAX, A_ref[:, 3::4].max(0))
eff = eng.estimate_effects(real_rows, eng.host_rows, exclude=EXCLUDE)
raw = []
Xs, Ys = eng.generate(eff, real_rows, eng.host_rows, np.random.default_rng(C.SEED), ch_min, ch_max,
                      lambda x: (raw.append(x.copy()), full_feats(x))[1])
Ss = np.stack(raw); Gs = F.from_signals(Ss)
Xtr = np.vstack([table(full, G), table(Xs, Gs)]); ytr = np.concatenate([real_y, Ys]); Str = np.concatenate([sig, Ss])
test = {}
for algo in ALGOS:
    pred = fit_predict(algo, Xtr, ytr, table(full_te, G_te), Str, sig_te, C.SEED)
    test[algo] = {"macro_f1": round(f1_score(y_te, pred, average="macro", zero_division=0), 4),
                  "accuracy": round(accuracy_score(y_te, pred), 4)}
    log(f"  {algo:12s} macro-F1={test[algo]['macro_f1']:.4f}  accuracy={test[algo]['accuracy']:.4f}")
log("  reference (synthetic-data report): real-only 0.586 / 0.705; v3+B+FFT 0.557 / 0.680; Phase 1 accuracy 0.677")

log.save_json("results.json", {"settings": {"best06": best06, "use_G": USE_G, "fixed_parser": FIXED,
                                            "exclude_pair": bool(EXCLUDE), "algorithms": ALGOS},
                               "cv_means": res[cols].mean().round(4).to_dict(), "paired": stats,
                               "best_cv": best, "posthoc_test": test})
log.close()
