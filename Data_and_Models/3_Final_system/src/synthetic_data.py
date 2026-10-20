"""the synthetic-data method + FFT + Config B (Final Experiments, 26-Sep).

Ported VERBATIM from Synthetic_Data_Part1.ipynb (Part 7, cells 16b-18, 21, 22a, 28, 29) so
that the synthetic-data model results reproduce exactly. Only the notebook globals are wired up here
(loaded from the saved synthetic-data artifacts in Models Improvement/Synthetic/):
    signal_cache_v3.npz  real signals of the 750 synthetic-class fault windows + 4200 hosts
    effects_v3.pkl       LEVEL_FLOOR / STD_FLOOR used by the generator
    fault_recipes.json   fault order (it fixes the random-number sequence)
    context.csv          zone / application of every window
Every window passed through this module is a REAL window; synthetic windows are produced
inside the training folds only (nested), never in validation.

Note: windows are parsed with the Phase 1 extractor (`config_a_phase1_parser` replica in
data.py), exactly as in the notebook. Step 07 measures the effect of the parser fix.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from . import config as C
from .data import _phase1_window

NUMERICAL_CHANNELS = ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS",
    "UL_NPRB", "UL_SNR", "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer",
    "PRBs_DL_Current", "PRBs_UL_Current", "PRB_Utilization_DL",
    "PRB_Utilization_UL", "UL_NumberOfPackets", "DL_NumberOfPackets"]
PHYS_LIMITS = {
    "RSRP": (-140, -45), "DL_BLER": (0, 1), "DL_MCS": (0, 27),
    "UL_BLER": (0, 1), "UL_MCS": (0, 27), "UL_NPRB": (0, 100),
    "UL_SNR": (-20, 40), "TX_Bytes": (0, None), "RX_Bytes": (0, None),
    "Estimated_UL_Buffer": (0, None), "PRBs_DL_Current": (0, 100),
    "PRBs_UL_Current": (0, 100), "PRB_Utilization_DL": (0, 100),
    "PRB_Utilization_UL": (0, 100), "UL_NumberOfPackets": (0, None),
    "DL_NumberOfPackets": (0, None),
}
MIN_OUTSIDE = 16
ALPHA = 0.001
LOG_CAP = 5.0
ADDITIVE = {"RSRP", "UL_SNR", "DL_MCS", "UL_MCS", "DL_BLER", "UL_BLER"}
KEYS = ["diff", "log_ratio", "log_std_ratio", "slope", "seg_std", "seg_mean"]
N_TARGET = 500                                    # final setting (cells 27-29)
N_THREADS = max(1, (os.cpu_count() or 4) - 2)     # as in cell 29
IDX_B_FFT = np.r_[0:320, 576:656]                 # Config B (A + patchwise) + FFT = 400


class SyntheticGenerator:
    """Holds the notebook state that estimate_effects() / generate() read as globals."""

    def __init__(self, d, fixed_parse=False):
        """fixed_parse=True: same rows, but signals from the correctly parsed raw_kpis.npy
        (BLER parser bug removed) for hosts, donors and real windows."""
        sd = C.REAL_PARSED_DIR
        self.fixed_parse = fixed_parse
        self.meta = pd.read_csv(C.RAW_META)
        self.context = pd.read_csv(os.path.join(sd, "context.csv"))
        self.LABELS = self.meta["label"].values
        self.session_id_by_row = d["session_ids"]
        with open(os.path.join(sd, "fault_recipes.json")) as f:
            self.SYNTHETIC_FAULTS = list(json.load(f).keys())
        with open(os.path.join(sd, "effects_v3.pkl"), "rb") as f:
            pk = pickle.load(f)
        self.LEVEL_FLOOR, self.STD_FLOOR = pk["LEVEL_FLOOR"], pk["STD_FLOOR"]
        cache = np.load(os.path.join(sd, "signal_cache_v3.npz"))
        self.cache_rows, self.cache_sig = cache["rows"], cache["signals"]
        if fixed_parse:
            self.raw = np.load(C.RAW_KPIS, mmap_mode="r")
            self.cache_sig = np.asarray(self.raw[self.cache_rows], dtype=np.float64)
        self.ROW2IDX = {int(r): i for i, r in enumerate(self.cache_rows)}
        train_y = d["train_y"]
        n_fault10 = int(np.isin(train_y, self.SYNTHETIC_FAULTS).sum())
        self.host_rows = self.cache_rows[n_fault10:]                 # cell 21 layout
        train_X = d["train_X"]
        nm = train_y == C.NORMAL_LABEL
        self.NORMAL_MIN = np.array([np.percentile(train_X[nm, c * 4 + 2], 0.1) for c in range(16)])
        self.NORMAL_MAX = np.array([np.percentile(train_X[nm, c * 4 + 3], 99.9) for c in range(16)])
        self._text = {}

    # --- signals ------------------------------------------------------------------
    def sig(self, r):
        return self.cache_sig[self.ROW2IDX[int(r)]]

    def real_sig(self, r):
        """Cached signal if available, else the Phase 1 extractor (as in the notebook)."""
        if int(r) in self.ROW2IDX:
            return self.sig(r)
        if self.fixed_parse:
            return np.asarray(self.raw[int(r)], dtype=np.float64)
        if int(r) not in self._text:
            self._load_text([r])
        return _phase1_window(self._text[int(r)])

    def _load_text(self, rows):
        want = set(int(r) for r in rows) - set(self._text)
        for chunk in pd.read_csv(C.CSV_PATH, usecols=["KPIs"], chunksize=2000):
            hit = chunk.index.intersection(list(want))
            self._text.update({int(k): v for k, v in chunk.loc[hit, "KPIs"].items()})
            if want <= set(self._text):
                break

    # --- notebook helpers (verbatim) ---------------------------------------------------
    @staticmethod
    def clip_channel(arr, ch):
        lo, hi = PHYS_LIMITS.get(ch, (None, None))
        return np.clip(arr, lo, hi)

    @staticmethod
    def fit_slope(t, y):
        return np.polyfit(t, y, 1)[0] if len(y) > 2 else 0.0

    def seg_bounds(self, r):
        s, e = int(self.meta.loc[r, "anom_start"]), int(self.meta.loc[r, "anom_end"])
        return (0, 127) if (s < 0 or e < s) else (s, e)

    def compute_effect(self, x, s, e, ref=None):
        LEVEL_FLOOR, STD_FLOOR = self.LEVEL_FLOOR, self.STD_FLOOR
        inside = np.zeros(128, bool); inside[s:e + 1] = True
        t_in, t_out = np.where(inside)[0], np.where(~inside)[0]
        diff = np.zeros(16)
        lr, lsr, slp = np.full(16, np.nan), np.full(16, np.nan), np.full(16, np.nan)
        for c, ch in enumerate(NUMERICAL_CHANNELS):
            seg = x[c, inside]
            if ref is None:
                out = x[c, ~inside]
                m_out, sd_out = out.mean(), out.std()
                raw = self.fit_slope(t_in, seg) - self.fit_slope(t_out, out)
            else:
                m_out, sd_out = ref[0][c], ref[1][c]
                raw = self.fit_slope(t_in, seg)
            m_in, sd_in = seg.mean(), seg.std()
            diff[c] = m_in - m_out
            if m_in > LEVEL_FLOOR[c] and m_out > LEVEL_FLOOR[c]:
                lr[c] = np.clip(np.log(m_in / m_out), -LOG_CAP, LOG_CAP)
            if sd_in > STD_FLOOR[c] and sd_out > STD_FLOOR[c]:
                lsr[c] = np.clip(np.log(sd_in / sd_out), -LOG_CAP, LOG_CAP)
            if ch in ADDITIVE:
                slp[c] = raw
            elif m_out > LEVEL_FLOOR[c]:
                slp[c] = np.clip(raw / m_out, -LOG_CAP / 128, LOG_CAP / 128)
        return diff, lr, lsr, slp, x[:, inside].std(1), x[:, inside].mean(1)

    def estimate_effects(self, train_rows, hosts, exclude=()):
        context, sig, LABELS = self.context, self.sig, self.LABELS
        hz, ha = context["zone"].values[hosts], context["app"].values[hosts]
        ref_of = lambda rows: (np.median(np.stack([sig(r) for r in rows]).mean(2), 0),
                               np.median(np.stack([sig(r) for r in rows]).std(2), 0))
        REF_f = {(z, a): ref_of(hosts[(hz == z) & (ha == a)]) for z, a in sorted(set(zip(hz, ha)))}
        REFZ_f = {z: ref_of(hosts[hz == z]) for z in sorted(set(hz))}
        eff = {}
        for fault in self.SYNTHETIC_FAULTS:
            if fault in exclude:
                continue
            rows = train_rows[LABELS[train_rows] == fault]
            rec, donors, within_idx = {k: [] for k in KEYS}, [], []
            for r in rows:
                s, e = self.seg_bounds(r)
                if (128 - (e - s + 1)) >= MIN_OUTSIDE:
                    vals = self.compute_effect(sig(r), s, e)
                    within_idx.append(len(donors))
                else:
                    z, a = context.loc[r, "zone"], context.loc[r, "app"]
                    ref = REF_f.get((z, a), REFZ_f.get(z))
                    if ref is None:
                        continue
                    vals = self.compute_effect(sig(r), s, e, ref)
                for k, v in zip(KEYS, vals):
                    rec[k].append(v)
                donors.append(r)
            rec = {k: np.array(v) for k, v in rec.items()}
            wi, affected = np.array(within_idx, int), np.zeros(16, bool)
            for c in range(16):
                for arr in (rec["diff"][wi, c], rec["log_std_ratio"][wi, c], rec["slope"][wi, c]):
                    v = arr[np.isfinite(arr)]
                    if len(v) < 10 or np.allclose(v, 0):
                        continue
                    try:
                        if wilcoxon(v).pvalue < ALPHA:
                            affected[c] = True
                            break
                    except ValueError:
                        pass
            eff[fault] = {**rec, "affected": affected, "rows_used": np.array(donors)}
        return eff

    def generate(self, eff, train_rows, hosts, rng, ch_min, ch_max, featurize):
        """Synthetic windows up to N_TARGET per class; featurize: (16,128) -> vector."""
        context, sig, LABELS = self.context, self.sig, self.LABELS
        LEVEL_FLOOR, STD_FLOOR, sess_of = self.LEVEL_FLOOR, self.STD_FLOOR, self.session_id_by_row
        hz, ha = context["zone"].values[hosts], context["app"].values[hosts]
        X, Y = [], []
        for fault in self.SYNTHETIC_FAULTS:
            if fault not in eff:
                continue
            E = eff[fault]
            if E["affected"].sum() == 0 or len(E["rows_used"]) == 0:
                continue
            f_rows = train_rows[LABELS[train_rows] == fault]
            f_sess = sess_of[f_rows]
            for _ in range(max(0, N_TARGET - len(f_rows))):
                k = rng.integers(len(E["rows_used"]))
                donor = E["rows_used"][k]
                same = f_rows[f_sess == sess_of[donor]]
                s, e = self.seg_bounds(same[rng.integers(len(same))])
                z, a = context.loc[donor, "zone"], context.loc[donor, "app"]
                pool = hosts[(hz == z) & (ha == a)]
                if len(pool) == 0:
                    pool = hosts[hz == z]
                if len(pool) == 0:
                    pool = hosts
                x = sig(pool[rng.integers(len(pool))]).copy()
                tc = np.arange(e - s + 1) - (e - s) / 2.0
                for c, ch in enumerate(NUMERICAL_CHANNELS):
                    if not E["affected"][c]:
                        continue
                    seg = x[c, s:e + 1]
                    m, sd = seg.mean(), seg.std()
                    lr, lsr, slp = E["log_ratio"][k, c], E["log_std_ratio"][k, c], E["slope"][k, c]
                    new_m = m * np.exp(lr) if (ch not in ADDITIVE and np.isfinite(lr) and m > LEVEL_FLOOR[c]) \
                        else m + E["diff"][k, c]
                    if sd > STD_FLOOR[c] and np.isfinite(lsr):
                        dev = (seg - m) * np.exp(lsr)
                    elif sd <= STD_FLOOR[c]:
                        d_sd = E["seg_std"][k, c]
                        if ch not in ADDITIVE and E["seg_mean"][k, c] > LEVEL_FLOOR[c]:
                            d_sd = d_sd / E["seg_mean"][k, c] * max(abs(new_m), LEVEL_FLOOR[c])
                        dev = rng.normal(0, d_sd, len(seg))
                    else:
                        dev = seg - m
                    trend = 0.0
                    if np.isfinite(slp):
                        trend = slp * (1.0 if ch in ADDITIVE else max(abs(m), LEVEL_FLOOR[c])) * tc
                    x[c, s:e + 1] = np.clip(self.clip_channel(new_m + dev + trend, ch), ch_min[c], ch_max[c])
                X.append(featurize(x)); Y.append(fault)
        return np.array(X, dtype=np.float32), np.array(Y)


# --- features (cell 29, verbatim) -------------------------------------------------------
def config_a(x):
    return np.stack([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1).reshape(-1).astype(np.float32)


def fft_feats(x):
    xc = x - x.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(xc, axis=1))
    return np.concatenate([mag[:, 1:5], mag[:, 1:].sum(1, keepdims=True)], axis=1).reshape(-1)


def patch_stats(x, n=8):
    parts = np.array_split(x, n, axis=1)
    return np.concatenate([np.stack([p.mean(1), p.std(1)], axis=1) for p in parts], axis=1).reshape(-1)


def diff_stats(x, n=8):
    return patch_stats(np.diff(x, axis=1), n)


def full_feats(x):
    """[Config A 64 | patchwise 256 | differences 256 | FFT 80] = 656"""
    return np.concatenate([config_a(x), patch_stats(x), diff_stats(x), fft_feats(x)]).astype(np.float32)


def fit_predict_w(X_tr, y_tr, w_tr, X_va, seed):
    """Cell 29: CatBoost defaults, class-balanced sample weights."""
    classes_ = np.unique(y_tr)
    eff_ = {c: w_tr[y_tr == c].sum() for c in classes_}
    total = sum(eff_.values())
    cw = {c: total / (len(classes_) * eff_[c]) for c in classes_}
    sw = w_tr * np.array([cw[c] for c in y_tr])
    clf = CatBoostClassifier(random_seed=seed, verbose=0, thread_count=N_THREADS,
                             allow_writing_files=False)
    clf.fit(X_tr, y_tr, sample_weight=sw)
    return clf, clf.predict(X_va).ravel()


def macro(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


# --- synthetic-window cache -----------------------------------------------------------------
def synth_cache_path(seed, fold, n_target=500, tag="bug", exclude=()):
    name = f"s{seed}_f{fold}_n{n_target}_{tag}{'_excl' if exclude else ''}.npz"
    return os.path.join(C.RESULTS, "cache", "synth", name)


def synthetic_windows(eng, real_rows, real_sess, A_ref, host_sess, tr, va, seed, fold,
                      n_target=500, tag="bug", exclude=()):
    """Nested synthetic-data generation for one CV fold, cached on disk.

    Returns (S, Y): raw synthetic windows (n, 16, 128) and labels. Generation is identical to
    the notebook (same RNG seed = seed*100 + fold-1, same hosts, bounds and effects); the cache
    only avoids repeating the ~10 min of single-threaded Python per fold.
    """
    import sys as _sys
    this = _sys.modules[__name__]
    path = synth_cache_path(seed, fold, n_target, tag, exclude)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["S"], z["Y"]
    val_sess = set(real_sess[va].tolist())
    hosts_ok = eng.host_rows[~np.isin(host_sess, list(val_sess))]
    ch_min = np.minimum(eng.NORMAL_MIN, A_ref[tr][:, 2::4].min(0))
    ch_max = np.maximum(eng.NORMAL_MAX, A_ref[tr][:, 3::4].max(0))
    eff = eng.estimate_effects(real_rows[tr], hosts_ok, exclude=exclude)
    old = this.N_TARGET
    this.N_TARGET = n_target
    raw = []
    try:
        _, Y = eng.generate(eff, real_rows[tr], hosts_ok, np.random.default_rng(seed * 100 + (fold - 1)),
                            ch_min, ch_max, lambda x: (raw.append(x.copy()), np.zeros(1, np.float32))[1])
    finally:
        this.N_TARGET = old
    S = np.stack(raw) if raw else np.zeros((0, 16, 128))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, S=S, Y=Y)          # float64: bit-identical to inline generation
    return S, Y
