"""Step 12c-1 - Session-level uncertainty for the key comparisons (no retraining).

With 10 fault sessions the unit of replication is the SESSION, not the window. This script reports, from the
existing per-fold results:
  1. Bootstrap confidence intervals of the mean paired difference, resampling FOLDS (each fold = a set of
     held-out sessions) with replacement, 10,000 draws, per seed-fold pair.
  2. The same for Model 1 (step 03 fix_both vs Phase 1; step 10a adopted+G vs adopted).
  3. Power analysis: with the observed SD of the paired fold differences, how many folds (≈ sessions)
     would be needed for a paired t-test to detect the observed gain at alpha 0.05, power 0.8.
Runs locally or on the server (reads results/*/folds.csv).  Output: results/12c_session_statistics/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.utils import StepLog

log = StepLog("12c_session_statistics")
R = C.RESULTS
rng = np.random.default_rng(C.SEED)


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    means = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())


def n_needed(d, alpha=0.05, power=0.8):
    """Paired t-test sample size for the observed mean/SD of differences (Cohen's d = mean/sd)."""
    d = np.asarray(d, float)
    eff = d.mean() / d.std(ddof=1)
    if eff <= 0:
        return np.inf, eff
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z / eff) ** 2)), eff


rows = []
def compare(name, a, b, unit):
    d = np.asarray(a) - np.asarray(b)
    m, lo, hi, p_le0 = boot_ci(d)
    n_req, eff = n_needed(d)
    rows.append({"comparison": name, "unit": unit, "n": len(d), "mean_diff": round(m, 4), "ci95_low": round(lo, 4),
                 "ci95_high": round(hi, 4), "p_boot(diff<=0)": round(p_le0, 4), "cohen_d": round(eff, 3),
                 "folds_needed_alpha05_power80": n_req, "sessions_equiv": (n_req * 2 if np.isfinite(n_req) else np.inf)})
    log(f"  {name:48s} d={m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(d<=0)={p_le0:.3f}  Cohen d={eff:.2f}  "
        f"folds needed={n_req}")

log("== Model 2 (macro-F1, 25 seed-folds; each fold holds out ~2 fault sessions) ==")
f06 = pd.read_csv(os.path.join(R, "06_model2_synthetic_data_experiments", "folds.csv")).sort_values(["seed", "fold"])
compare("the synthetic-data model (v3+B+FFT) vs real-only baseline", f06["final"], f06["real_only"], "seed-fold")
compare("final (+trend) vs real-only baseline", f06["trend"], f06["real_only"], "seed-fold")
compare("final (+trend) vs the synthetic-data model", f06["trend"], f06["final"], "seed-fold")
f09 = pd.read_csv(os.path.join(R, "09c_model2_candidates", "folds.csv")).sort_values(["seed", "fold"])
compare("QUANT real-only vs the synthetic-data model", f09["quant_real"], f09["catboost_v3"], "seed-fold")
compare("vote(CB+trend, QUANT) vs the synthetic-data model", f09["vote_2"], f09["catboost_v3"], "seed-fold")

log("\n== Model 1 (PR-AUC, 25 seed-folds) ==")
f03 = pd.read_csv(os.path.join(R, "03_model1_fixes", "folds.csv"))
p1 = f03[f03.variant == "phase1"].sort_values(["seed", "fold"]); fb = f03[f03.variant == "fix_both"].sort_values(["seed", "fold"])
compare("fix_both vs Phase 1 (PR-AUC)", fb["pr_auc"].values, p1["pr_auc"].values, "seed-fold")
compare("fix_both vs Phase 1 (false alarms, sign flipped)", -fb["false_alarms"].values / 1000, -p1["false_alarms"].values / 1000, "seed-fold (k windows)")
f10a = pd.read_csv(os.path.join(R, "10a_model1_shift", "folds.csv"))
ad = f10a[f10a.variant == "adopted"].sort_values(["seed", "fold"]); ag = f10a[f10a.variant == "adopted+G"].sort_values(["seed", "fold"])
compare("adopted+G vs adopted (PR-AUC)", ag["pr_auc"].values, ad["pr_auc"].values, "seed-fold")

log("\n== Per-fold (= per held-out session group) view of the final Model 2 vs the synthetic-data model, seed-averaged ==")
fold_sessions = {1: "16, 18 (+Normal)", 2: "14, 32", 3: "2, 15", 4: "0, 12", 5: "11, 19"}
per_fold = f06.groupby("fold")[["real_only", "final", "trend"]].mean()
per_fold["sessions (seed 42)"] = per_fold.index.map(fold_sessions)
log(per_fold.round(4).to_string())

out = pd.DataFrame(rows)
out.to_csv(log.path("bootstrap_ci.csv"), index=False)
log("\nReading guide: a 95% CI that excludes 0 ≈ significant at 5 %; 'folds needed' = how many seed-folds the "
    "paired test would need at the observed effect size (≈ 2 fault sessions per fold).")
log.save_json("results.json", {"comparisons": rows})
log.close()
