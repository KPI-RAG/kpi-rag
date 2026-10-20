"""Step 06 - Model 2: reproduce the synthetic-data model experiment, then run the report's "Next Steps".

Starting point (Final Experiments 26-Sep, Synthetic_Data_Report_v2):
    final  = real + synthetic (500/class, 10 classes) with Config B + FFT (400 features)
             CV macro-F1 0.3743 (configBC_records.csv, column B_fft)
Variants, all on the SAME nested 5 seeds x 5 session folds (synthetic data rebuilt from
training-fold sessions only; validation = real windows only):
    final        reproduction of the synthetic-data model           -> must match the original records
    excl_pair    next step 1: no augmentation for HNC Gradual + Buffer Overflow
    trend        next step 2: + trend/timing features G1 G2 G3 (src/features.py, 96)
    trend_fixed  trend + BLER parser fix (correct signals from raw_kpis.npy)
Comparison: each variant vs the real-only baseline (0.2908, step 01) and vs `final`,
Nadeau-Bengio corrected t-test. Checkpointed per fold; safe to re-run (resumes).

Run from the phase2 folder:   python scripts/06_model2_synthetic_data_experiments.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src import features as F
from src.data import load_splits
from src.evaluation import nadeau_bengio_ttest
from src.utils import StepLog
from src.synthetic_data import IDX_B_FFT, N_THREADS, SyntheticGenerator, fit_predict_w, full_feats, macro

PAIR = ("High Network Congestion (Gradual Buildup)", "Buffer Overflow (Gradual Buildup)")
VARIANTS = ["final", "excl_pair", "trend", "trend_fixed"]
TOL = 1e-3

log = StepLog("06_model2_synthetic_data_experiments")
d = load_splits()
fault = d["train_y"] != C.NORMAL_LABEL
real_rows = d["train_idx"][fault]
real_y = d["train_y"][fault]
real_X = d["train_X"][fault][:, :C.N_FEATURES_A]
real_sess = d["session_ids"][real_rows]

orig = pd.read_csv(os.path.join(C.REAL_PARSED_DIR, "configBC_records.csv"))
base = pd.read_csv(os.path.join(C.RESULTS, "01_baseline", "baseline_folds.csv"))

eng = {"bug": SyntheticGenerator(d), "fixed": SyntheticGenerator(d, fixed_parse=True)}
assert np.array_equal(eng["bug"].cache_rows[:int(np.isin(real_y, eng["bug"].SYNTHETIC_FAULTS).sum())],
                      d["train_idx"][np.isin(d["train_y"], eng["bug"].SYNTHETIC_FAULTS)]), "cache layout"
log(f"fault order (fixes the RNG sequence): {eng['bug'].SYNTHETIC_FAULTS}")
log(f"hosts: {len(eng['bug'].host_rows)} | CatBoost threads: {N_THREADS}")

# Jamming windows are not in the signal cache: read their text in ONE pass over the CSV
eng["bug"]._load_text([r for r in real_rows if int(r) not in eng["bug"].ROW2IDX])

# Real-window features for both parsers: [full 656 | G 96]
real = {}
for k, e in eng.items():
    sig = np.stack([e.real_sig(r) for r in real_rows])
    real[k] = {"full": np.stack([full_feats(x) for x in sig]), "G": F.from_signals(sig),
               "A": np.stack([full_feats(x)[:64] for x in sig])}
log(f"Config A from cached signals vs saved train_X: max|diff| = "
    f"{np.abs(real['bug']['full'][:, :64] - real_X).max():.4g} (notebook reported 0.0156)")
host_sess = d["session_ids"][eng["bug"].host_rows]


def synth(e, tr, va, seed, f_i, exclude=(), A_ref=None):
    """Her nested generation for one fold. Returns (full features, G features, labels)."""
    val_sess = set(real_sess[va].tolist())
    hosts_ok = e.host_rows[~np.isin(host_sess, list(val_sess))]
    A = real_X if A_ref is None else A_ref
    ch_min = np.minimum(e.NORMAL_MIN, A[tr][:, 2::4].min(0))
    ch_max = np.maximum(e.NORMAL_MAX, A[tr][:, 3::4].max(0))
    eff = e.estimate_effects(real_rows[tr], hosts_ok, exclude=exclude)
    raw = []
    feat = lambda x: (raw.append(x.copy()), full_feats(x))[1]
    Xs, Ys = e.generate(eff, real_rows[tr], hosts_ok, np.random.default_rng(seed * 100 + f_i),
                        ch_min, ch_max, feat)
    G = F.from_signals(np.stack(raw)) if raw else np.zeros((0, 96))
    return Xs, G, Ys


ckpt = log.path("folds_partial.csv")
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["seed"], r["fold"]) for r in rows}
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for f_i, (tr, va) in enumerate(cv.split(real_X, real_y, groups=real_sess)):
        if (seed, f_i + 1) in done:
            continue
        rec = {"seed": seed, "fold": f_i + 1, "n_train": len(tr), "n_val": len(va)}
        yva, ytr = real_y[va], real_y[tr]
        rb, rf = real["bug"], real["fixed"]

        Xs, Gs, Ys = synth(eng["bug"], tr, va, seed, f_i)
        ya = np.concatenate([ytr, Ys])
        Xa = np.vstack([rb["full"][tr][:, IDX_B_FFT], Xs[:, IDX_B_FFT]])
        rec["final"] = macro(yva, fit_predict_w(Xa, ya, np.ones(len(ya)), rb["full"][va][:, IDX_B_FFT], seed)[1])

        Xg = np.hstack([Xa, np.vstack([rb["G"][tr], Gs])])
        Xg_va = np.hstack([rb["full"][va][:, IDX_B_FFT], rb["G"][va]])
        rec["trend"] = macro(yva, fit_predict_w(Xg, ya, np.ones(len(ya)), Xg_va, seed)[1])

        Xs2, _, Ys2 = synth(eng["bug"], tr, va, seed, f_i, exclude=PAIR)
        ya2 = np.concatenate([ytr, Ys2])
        Xa2 = np.vstack([rb["full"][tr][:, IDX_B_FFT], Xs2[:, IDX_B_FFT]])
        rec["excl_pair"] = macro(yva, fit_predict_w(Xa2, ya2, np.ones(len(ya2)), rb["full"][va][:, IDX_B_FFT], seed)[1])

        Xs3, Gs3, Ys3 = synth(eng["fixed"], tr, va, seed, f_i, A_ref=rf["A"])
        ya3 = np.concatenate([ytr, Ys3])
        Xf = np.hstack([np.vstack([rf["full"][tr][:, IDX_B_FFT], Xs3[:, IDX_B_FFT]]), np.vstack([rf["G"][tr], Gs3])])
        Xf_va = np.hstack([rf["full"][va][:, IDX_B_FFT], rf["G"][va]])
        rec["trend_fixed"] = macro(yva, fit_predict_w(Xf, ya3, np.ones(len(ya3)), Xf_va, seed)[1])

        ref = orig[(orig.seed == seed) & (orig.fold == f_i + 1)].B_fft.iloc[0]
        rec["original_reference"] = ref
        rows.append(rec)
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        ok = abs(rec["final"] - ref) <= TOL
        log(f"  seed={seed:<5} fold={f_i+1}  final={rec['final']:.4f} (original {ref:.4f} "
            f"{'MATCH' if ok else 'DIFF'})  excl_pair={rec['excl_pair']:.4f}  "
            f"trend={rec['trend']:.4f}  trend_fixed={rec['trend_fixed']:.4f}")
        if not ok and len(rows) == 1:
            log("STOP: the reproduction of the synthetic-data model does not match on the first fold.")
            log.close()
            sys.exit(2)

res = pd.DataFrame(rows).merge(base[["seed", "fold", "f1_macro"]].rename(columns={"f1_macro": "real_only"}),
                               on=["seed", "fold"]).sort_values(["seed", "fold"])
res.to_csv(log.path("folds.csv"), index=False)
n_match = int((np.abs(res["final"] - res["original_reference"]) <= TOL).sum())
log(f"\nReproduction of the synthetic-data model: {n_match}/25 folds match the original records "
    f"(mean {res['final'].mean():.4f} vs original {res['original_reference'].mean():.4f})")

log("\n== Macro-F1 over 25 folds ==")
cols = ["real_only", "final"] + VARIANTS[1:]
log(res[cols].agg(["mean", "std"]).T.round(4).to_string())

stats = {}
log("\n== Paired comparisons (Nadeau-Bengio corrected t-test) ==")
for ref_col in ("real_only", "final"):
    for v in VARIANTS:
        if v == ref_col:
            continue
        diff, t_stat, p, w = nadeau_bengio_ttest(res[v] - res[ref_col], res.n_train, res.n_val)
        wins = int((res[v] > res[ref_col]).sum())
        stats[f"{v}_vs_{ref_col}"] = {"mean_diff": round(diff, 4), "p_corrected": round(p, 4),
                                      "p_wilcoxon": round(w, 4), "folds_better": wins}
        log(f"  {v:12s} vs {ref_col:9s}: d={diff:+.4f}  p={p:.4f}  wins {wins}/25")

log.save_json("results.json", {"reproduction_matches": n_match,
                               "means": res[cols].mean().round(4).to_dict(), "paired": stats})
log.close()
