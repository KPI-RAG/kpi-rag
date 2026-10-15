"""Loading the canonical session-split data and checking its integrity."""
import hashlib
import os
import re

import numpy as np
import pandas as pd

from . import config as C


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_splits():
    """Return a dict with train_X, train_y, train_idx, test_idx, session_ids."""
    d = {}
    for key, fn in C.SPLIT_FILES.items():
        d[key] = np.load(os.path.join(C.DATA_DIR, fn), allow_pickle=True)
    d["train_y"]   = d["train_y"].astype(str)
    d["train_idx"] = d["train_idx"].astype(int)
    d["test_idx"]  = d["test_idx"].astype(int)
    d["train_sessions"] = d["session_ids"][d["train_idx"]]
    d["test_sessions"]  = d["session_ids"][d["test_idx"]]
    return d


def csv_labels():
    """Label of every CSV row (cached by the Phase-1 synthetic notebook)."""
    return pd.read_csv(C.RAW_META)["label"].values.astype(str)


def integrity_checks(d, labels, log=print):
    """Hard checks - raise AssertionError if the data are not what we expect."""
    order = float(np.mean(labels[d["train_idx"]] == d["train_y"]))
    log(f"Check 1  CSV label order vs train_y   : {order * 100:.2f}% (must be 100%)")
    assert order == 1.0, "row order mismatch"

    overlap = set(np.unique(d["train_sessions"])) & set(np.unique(d["test_sessions"]))
    log(f"Check 2  train/test session overlap   : {sorted(overlap) or 'none'} (must be none)")
    assert not overlap, "session leakage between train and test"

    is_fault = d["train_y"] != C.NORMAL_LABEL
    log(f"Check 3  training fault windows       : {int(is_fault.sum())} (expected 935)")
    assert int(is_fault.sum()) == 935
    log(f"         training sessions            : {len(np.unique(d['train_sessions']))}")
    log(f"         sessions containing faults   : {len(np.unique(d['train_sessions'][is_fault]))}")


def _stats(x):
    """(n, 16, 128) signals -> (n, 64) Config A: [ch0_mean, ch0_std, ch0_min, ch0_max, ch1_mean, ...]."""
    feats = np.stack([x.mean(2), x.std(2), x.min(2), x.max(2)], axis=2)
    return feats.reshape(len(x), -1)


def config_a_from_raw(rows):
    """Config A from the CORRECTLY parsed signals in raw_kpis.npy."""
    raw = np.load(C.RAW_KPIS, mmap_mode="r")
    return _stats(np.asarray(raw[rows], dtype=np.float64))


# --- Exact replica of the Phase 1 feature parser -------------------------------------
# Phase 1 extracted numbers with the regex below. It mis-reads numbers printed as
# "1.e-05" (no digit after the dot) as TWO values, 1 and -5, and then keeps only the
# first 128 values. This corrupts the BLER channels of ~2% of windows. The replica is
# needed to rebuild test features exactly as Phase 1 did (for a like-for-like check).
_PHASE1_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_CHANNELS = ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
             "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "PRBs_DL_Current",
             "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL",
             "UL_NumberOfPackets", "DL_NumberOfPackets"]


def _phase1_window(kpi_text, size=128):
    x = np.zeros((len(_CHANNELS), size))
    for c, ch in enumerate(_CHANNELS):
        m = re.search(rf"'{ch}':\s*array\(\[(.*?)\]", str(kpi_text), re.DOTALL)
        if m:
            v = np.array([float(t) for t in _PHASE1_NUM.findall(m.group(1))[:size]])
            x[c, :len(v)] = v
    return x


def config_a_phase1_parser(rows):
    """Config A for CSV rows, rebuilt with the Phase 1 parser (bug included).

    Reads only the needed rows of telecomts.csv; result cached in results/cache/.
    """
    rows = np.asarray(rows, dtype=int)
    cache = os.path.join(C.RESULTS, "cache",
                         f"configA_phase1parser_{hashlib.sha1(rows.tobytes()).hexdigest()[:12]}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    wanted = set(rows.tolist())
    text = {}
    for chunk in pd.read_csv(C.CSV_PATH, usecols=["KPIs"], chunksize=2000):
        hit = chunk.index.intersection(list(wanted))
        text.update(chunk.loc[hit, "KPIs"].to_dict())
    X = _stats(np.stack([_phase1_window(text[r]) for r in rows]))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, X)
    return X


def features(d, part, fixed):
    """Config A for 'train' or 'test'. fixed=False -> Phase 1 features (BLER bug);
    fixed=True -> correctly parsed features from raw_kpis.npy."""
    rows = d["train_idx"] if part == "train" else d["test_idx"]
    if fixed:
        return config_a_from_raw(rows)
    if part == "train":
        return d["train_X"][:, :C.N_FEATURES_A]
    return config_a_phase1_parser(rows)


def check_phase1_parser(d, n=2000, log=print):
    """Prove the replica parser reproduces the saved train_X exactly (corrupted rows too)."""
    rebuilt = config_a_phase1_parser(d["train_idx"][:n])
    saved = d["train_X"][:n, :C.N_FEATURES_A]
    correct = config_a_from_raw(d["train_idx"][:n])
    ok = np.allclose(rebuilt, saved, rtol=1e-6, atol=1e-6)
    n_bug = int((~np.isclose(correct, saved, rtol=1e-3, atol=1e-4)).any(1).sum())
    log(f"Phase 1 parser replica vs saved train_X ({n} rows, {n_bug} with the BLER bug): "
        f"max|diff| = {np.max(np.abs(rebuilt - saved)):.2e}  -> {'EXACT' if ok else 'MISMATCH'}")
    assert ok, "Phase 1 parser replica does not reproduce train_X"
