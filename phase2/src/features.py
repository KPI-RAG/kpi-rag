"""New per-window feature groups (step 05), grounded in the TelecomTS paper.

Every feature uses only the current (16, 128) window and fixed constants, so it is safe
under session-grouped CV and computable at inference. Design: scratchpad proposal by the
feature-engineering agent, reviewed against step 04 (see results/05_*/NOTES.md).

G1  within-window segment shift (32): largest noise-normalised level step over fixed split
    points + log std-ratio at that split. Paper Table 9 (constant addition/multiplication
    applied to a segment). Level-free.
G2  temporal shape (48): Legendre P1..P3 coefficients of each standardised channel
    (trend, convexity, S-shape). Paper Table 9 (linear/exponential/logistic growth, decay).
G3  multi-KPI co-movement (16): how many KPIs shift, in which groups, how in sync, and
    cross-channel correlations. Paper Sec 4.1 / Fig 6 vs Table 10 (faults hit 7-13 KPIs).
"""
import hashlib
import os

import numpy as np

from . import config as C

EPS = 1e-8
SPLITS = np.array([16, 32, 48, 64, 80, 96, 112])
SHIFT_ACTIVE = 5.0                       # |G1 shift| above this = "KPI shifted" (fixed constant)
RADIO, RESOURCE, TRAFFIC = [0, 1, 2, 3, 4, 6], [5, 10, 11, 12, 13], [7, 8, 9, 14, 15]
PAIRS = [(6, 4), (2, 1), (4, 3), (7, 15), (9, 14), (5, 13)]   # SNR-MCS, MCS-BLER (DL, UL), ...
CH = ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR", "TX_Bytes",
      "RX_Bytes", "Est_UL_Buffer", "PRBs_DL", "PRBs_UL", "PRB_Util_DL", "PRB_Util_UL",
      "UL_Packets", "DL_Packets"]

NAMES = {
    "G1": [f"{c}_shift" for c in CH] + [f"{c}_logstdratio" for c in CH],
    "G2": [f"{c}_{k}" for k in ("trend", "convexity", "sshape") for c in CH],
    "G3": ["n_shift_radio", "n_shift_resource", "n_shift_traffic", "n_shift_total",
           "radio_share", "onset_spread", "onset_agree"]
          + [f"corr_{CH[i]}_{CH[j]}" for i, j in PAIRS]
          + ["diff_mean_abs_corr", "diff_top_eig_frac", "diff_radio_traffic_corr"],
}


def _g1(x):
    n, c, L = x.shape
    cs = np.concatenate([np.zeros((n, c, 1)), np.cumsum(x, 2)], 2)
    cs2 = np.concatenate([np.zeros((n, c, 1)), np.cumsum(x * x, 2)], 2)
    t = SPLITS
    m_before = cs[:, :, t] / t
    m_after = (cs[:, :, [L]] - cs[:, :, t]) / (L - t)
    v_before = np.maximum(cs2[:, :, t] / t - m_before ** 2, 0)
    v_after = np.maximum((cs2[:, :, [L]] - cs2[:, :, t]) / (L - t) - m_after ** 2, 0)
    sig = np.diff(x, axis=2).std(2) / np.sqrt(2) + EPS                    # (n, c) noise scale
    S = (m_after - m_before) / (sig[:, :, None] * np.sqrt(1 / t + 1 / (L - t)))
    k = np.abs(S).argmax(2)                                               # (n, c)
    take = lambda a: np.take_along_axis(a, k[:, :, None], 2)[:, :, 0]
    shift = take(S)
    lvr = np.abs(np.log((np.sqrt(take(v_after)) + EPS) / (np.sqrt(take(v_before)) + EPS)))
    return np.concatenate([shift, lvr], 1), shift, SPLITS[k]


def _g2(x):
    t = np.linspace(-1, 1, x.shape[2])
    P = np.stack([np.ones_like(t), t, 0.5 * (3 * t ** 2 - 1), 0.5 * (5 * t ** 3 - 3 * t)], 1)
    z = (x - x.mean(2, keepdims=True)) / (x.std(2, keepdims=True) + EPS)
    coef = z @ np.linalg.pinv(P).T                                        # (n, c, 4)
    return coef[:, :, 1:].transpose(0, 2, 1).reshape(len(x), -1)          # trend|convex|sshape


def _corr(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(-1) / den, np.nan)


def _g3(x, shift, tstar):
    act = np.abs(shift) > SHIFT_ACTIVE
    n_r, n_s, n_t = act[:, RADIO].sum(1), act[:, RESOURCE].sum(1), act[:, TRAFFIC].sum(1)
    n_tot = act.sum(1)
    k = np.maximum(n_tot, 1)
    mean_t = (act * tstar).sum(1) / k
    var_t = (act * (tstar - mean_t[:, None]) ** 2).sum(1) / k
    spread = np.where(n_tot >= 2, np.sqrt(var_t), np.nan)       # onset spread of shifted KPIs
    agree = np.max(np.stack([(act & (tstar == t)).sum(1) for t in SPLITS], 1), 1) / (n_tot + 1)
    pairs = np.stack([_corr(x[:, i], x[:, j]) for i, j in PAIRS], 1)
    D = np.diff(x, axis=2)
    Dz = (D - D.mean(2, keepdims=True)) / (D.std(2, keepdims=True) + EPS)
    Cm = np.einsum("nct,ndt->ncd", Dz, Dz) / D.shape[2]
    idx = np.arange(Cm.shape[1])
    Cm[:, idx, idx] = 0
    mean_abs = np.abs(Cm).sum((1, 2)) / (16 * 15)
    top_eig = np.linalg.eigvalsh(Cm + np.eye(16))[:, -1] / 16
    rt = np.abs(Cm[:, RADIO][:, :, TRAFFIC]).mean((1, 2))
    return np.column_stack([n_r, n_s, n_t, n_tot, n_r / (n_tot + 1), spread, agree,
                            pairs, mean_abs, top_eig, rt])


def compute(rows, groups=("G1", "G2", "G3"), chunk=4000):
    """New features for CSV rows, from the correctly parsed raw signals. Cached on disk."""
    rows = np.asarray(rows, dtype=int)
    key = hashlib.sha1(rows.tobytes() + ",".join(groups).encode()).hexdigest()[:12]
    cache = os.path.join(C.RESULTS, "cache", f"features_{'_'.join(groups)}_{key}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    raw = np.load(C.RAW_KPIS, mmap_mode="r")
    out = []
    for i in range(0, len(rows), chunk):
        x = np.asarray(raw[rows[i:i + chunk]], dtype=np.float64)
        g1, shift, tstar = _g1(x)
        parts = {"G1": g1, "G2": _g2(x), "G3": _g3(x, shift, tstar)}
        out.append(np.concatenate([parts[g] for g in groups], 1))
    X = np.concatenate(out)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, X)
    return X


def from_signals(x):
    """G1|G2|G3 (96) for a batch of windows x: (n, 16, 128). No caching."""
    x = np.asarray(x, dtype=np.float64)
    g1, shift, tstar = _g1(x)
    return np.concatenate([g1, _g2(x), _g3(x, shift, tstar)], 1)


def names(groups=("G1", "G2", "G3")):
    return [n for g in groups for n in NAMES[g]]
