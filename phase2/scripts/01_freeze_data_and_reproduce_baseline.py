"""Step 01 - Freeze the canonical data and reproduce the Phase 1 baseline.

1. SHA-256 fingerprint of every input file  -> results/01_baseline/manifest.json
2. Integrity checks (row order, no session leakage, 935 fault windows)
3. Model 2 baseline: real-only Config A, CatBoost defaults + balanced class weights,
   StratifiedGroupKFold(5) by session x 5 seeds. Must reproduce 0.2908 +- 0.005.

Run from the phase2 folder:   python scripts/01_freeze_data_and_reproduce_baseline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.data import csv_labels, integrity_checks, load_splits, sha256
from src.models import balanced_class_weights, fault_classifier
from src.utils import StepLog, library_versions

log = StepLog("01_baseline")

# 1. Fingerprint inputs ---------------------------------------------------------
log("\n== 1. Input files ==")
inputs = [os.path.join(C.DATA_DIR, fn) for fn in C.SPLIT_FILES.values()] + [C.RAW_META, C.RAW_KPIS]
manifest = {"versions": library_versions(), "files": {}}
for p in inputs:
    rel = os.path.relpath(p, C.ROOT)
    manifest["files"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    log(f"  {rel:58s} {manifest['files'][rel]['sha256'][:16]}")

# 2. Integrity ------------------------------------------------------------------
log("\n== 2. Integrity checks ==")
d = load_splits()
integrity_checks(d, csv_labels(), log)

# 3. Baseline reproduction --------------------------------------------------------
log("\n== 3. Model 2 baseline (5 seeds x 5 folds) ==")
fault = d["train_y"] != C.NORMAL_LABEL
X = d["train_X"][fault][:, :C.N_FEATURES_A]
y = d["train_y"][fault]
groups = d["train_sessions"][fault]

rows = []
for seed in C.CV_SEEDS:
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), start=1):
        clf = fault_classifier(seed, class_weights=balanced_class_weights(y[tr]))
        clf.fit(X[tr], y[tr])
        f1 = f1_score(y[va], clf.predict(X[va]).ravel(), average="macro", zero_division=0)
        rows.append({"seed": seed, "fold": fold, "f1_macro": f1,
                     "n_train": len(tr), "n_val": len(va)})
        log(f"  seed={seed:<5} fold={fold}  F1-macro={f1:.4f}")

res = pd.DataFrame(rows)
res.to_csv(log.path("baseline_folds.csv"), index=False)
mean, std = res["f1_macro"].mean(), res["f1_macro"].std(ddof=0)
passed = abs(mean - C.BASELINE_MEAN) <= C.BASELINE_TOL
log(f"\nBaseline F1-macro: {mean:.4f} +- {std:.4f}   (Phase 1 record: "
    f"{C.BASELINE_MEAN} +- {C.BASELINE_STD})  ->  {'REPRODUCED' if passed else 'NOT REPRODUCED'}")

manifest["baseline"] = {"f1_macro_mean": round(mean, 4), "f1_macro_std": round(std, 4),
                        "reproduced": bool(passed)}
log.save_json("manifest.json", manifest)
log.close()
sys.exit(0 if passed else 1)
