"""Step 09z - Build the synthetic-window cache once (speed-up for every later step).

Generates the synthetic windows (500 per class, the synthetic-data model setting) for all
5 seeds x 5 session folds and stores them in results/cache/synth/. Later experiments load
them instead of re-running the ~10-minute single-threaded generator per fold.
Runs 3 folds in parallel (fits the 16 GB container limit). Identical output to the inline
generation used in steps 06 and 08 (same seeds, hosts, bounds and effects).

Run from the phase2 folder:   python scripts/09z_build_synth_cache.py
"""
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.data import load_splits
from src.utils import StepLog
from src.synthetic_data import SyntheticGenerator, synth_cache_path, synthetic_windows

N_TARGET, TAG, WORKERS = 500, "bug", 3
_state = {}


def _init():
    d = load_splits()
    fault = d["train_y"] != C.NORMAL_LABEL
    eng = SyntheticGenerator(d)
    _state.update(d=d, eng=eng, real_rows=d["train_idx"][fault], real_y=d["train_y"][fault],
                  real_sess=d["session_ids"][d["train_idx"][fault]],
                  A_ref=d["train_X"][fault][:, :C.N_FEATURES_A],
                  host_sess=d["session_ids"][eng.host_rows])


def _job(args):
    seed, fold, tr, va = args
    t = time.time()
    S, Y = synthetic_windows(_state["eng"], _state["real_rows"], _state["real_sess"], _state["A_ref"],
                             _state["host_sess"], tr, va, seed, fold, N_TARGET, TAG)
    return seed, fold, len(Y), round(time.time() - t)


if __name__ == "__main__":
    log = StepLog("09z_synth_cache")
    d = load_splits()
    fault = d["train_y"] != C.NORMAL_LABEL
    X, y, g = d["train_X"][fault][:, :C.N_FEATURES_A], d["train_y"][fault], d["session_ids"][d["train_idx"][fault]]
    jobs = []
    for seed in C.CV_SEEDS:
        cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
        for f_i, (tr, va) in enumerate(cv.split(X, y, groups=g)):
            if not os.path.exists(synth_cache_path(seed, f_i + 1, N_TARGET, TAG)):
                jobs.append((seed, f_i + 1, tr, va))
    log(f"folds to generate: {len(jobs)} / 25  (workers={WORKERS}, n_target={N_TARGET})")
    with Pool(WORKERS, initializer=_init) as pool:
        for seed, fold, n, sec in pool.imap_unordered(_job, jobs):
            log(f"  seed={seed:<5} fold={fold}  synthetic={n}  ({sec}s)")
    total = sum(os.path.getsize(synth_cache_path(s, f, N_TARGET, TAG))
                for s in C.CV_SEEDS for f in range(1, 6)) / 1e6
    log(f"cache complete: 25 folds, {total:.0f} MB in results/cache/synth/")

    # Equivalence check: seed 42 fold 1 regenerated inline (as steps 06/08 did) == cache
    from src.synthetic_data import full_feats
    _init()
    s = _state
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=42)
    tr, va = next(iter(cv.split(X, y, groups=g)))
    val_sess = set(s["real_sess"][va].tolist())
    hosts_ok = s["eng"].host_rows[~np.isin(s["host_sess"], list(val_sess))]
    ch_min = np.minimum(s["eng"].NORMAL_MIN, s["A_ref"][tr][:, 2::4].min(0))
    ch_max = np.maximum(s["eng"].NORMAL_MAX, s["A_ref"][tr][:, 3::4].max(0))
    eff = s["eng"].estimate_effects(s["real_rows"][tr], hosts_ok)
    Xs, Ys = s["eng"].generate(eff, s["real_rows"][tr], hosts_ok, np.random.default_rng(4200), ch_min, ch_max, full_feats)
    z = np.load(synth_cache_path(42, 1, N_TARGET, TAG), allow_pickle=True)
    Xc = np.stack([full_feats(x) for x in z["S"]])
    ok = len(Ys) == len(z["Y"]) and np.array_equal(Ys, z["Y"]) and np.array_equal(Xs, Xc)
    log(f"equivalence check (seed 42, fold 1): inline {len(Ys)} windows vs cache {len(z['Y'])} -> "
        f"{'IDENTICAL' if ok else 'MISMATCH'}  max|diff| = {np.max(np.abs(Xs - Xc)) if len(Ys) == len(z['Y']) else 'n/a'}")
    assert ok, "cache does not reproduce inline generation"
    log.close()
