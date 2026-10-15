"""Model construction - identical recipes to the Phase 1 submission."""
from collections import Counter

import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils import resample

from . import config as C


def balanced_class_weights(y):
    """weight_c = n / (n_classes * n_c), as used in Phase 1."""
    counts = Counter(y)
    return {c: len(y) / (len(counts) * n) for c, n in counts.items()}


def fault_classifier(seed, params=None, class_weights=None):
    """Model 2 (11-class fault type). params=None -> CatBoost defaults (CV harness)."""
    return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=C.THREADS,
                              allow_writing_files=False, class_weights=class_weights,
                              **(params or {}))


def anomaly_detector(seed=C.SEED, class_weighting=True):
    """Model 1 (binary Normal vs anomaly) with the Phase 1 tuned parameters.

    class_weighting=True reproduces Phase 1 (auto_class_weights="Balanced" on top of the
    80/20 rebalancing). False keeps only the rebalancing - the imbalance is corrected once.
    """
    params = dict(C.MODEL1_PARAMS)
    if not class_weighting:
        params.pop("auto_class_weights")
    return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=C.THREADS,
                              allow_writing_files=False, **params)


def rebalance_80_20(rows, y_bin, seed=C.SEED):
    """Phase 1 recipe: keep all anomalies, downsample Normal to 4x their count.

    rows: candidate row indices (e.g. a training fold). Returns the kept indices.
    """
    normal = rows[y_bin[rows] == 0]
    anomaly = rows[y_bin[rows] == 1]
    target = 4 * len(anomaly)
    if target < len(normal):
        normal = resample(normal, n_samples=target, replace=False, random_state=seed)
    keep = np.concatenate([normal, anomaly])
    np.random.RandomState(seed).shuffle(keep)
    return keep
