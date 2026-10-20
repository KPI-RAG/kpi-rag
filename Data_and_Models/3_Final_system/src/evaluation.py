"""Metrics, threshold analysis and paired statistics."""
import numpy as np
from scipy.stats import t as t_dist, wilcoxon
from sklearn.metrics import fbeta_score, precision_recall_curve


def binary_report(prob, y, threshold):
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"threshold": round(float(threshold), 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-12), 4),
            "f2": round(fbeta_score(y, pred, beta=2, zero_division=0), 4)}


def threshold_max_precision(prob, y, min_recall):
    """Highest-precision threshold whose recall is still >= min_recall."""
    prec, rec, thr = precision_recall_curve(y, prob)
    ok = rec[:-1] >= min_recall
    return float(thr[ok][np.argmax(prec[:-1][ok])])


def threshold_f2(prob, y):
    prec, rec, thr = precision_recall_curve(y, prob)
    f2 = 5 * prec[:-1] * rec[:-1] / np.clip(4 * prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f2)])


def nadeau_bengio_ttest(diffs, n_train, n_val):
    """Corrected resampled t-test for repeated CV (Nadeau & Bengio, 2003).

    diffs: per-fold score differences (method - baseline); n_train/n_val: fold sizes.
    Returns (mean diff, t statistic, two-sided p, Wilcoxon p).
    """
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    ratio = float(np.mean(np.asarray(n_val) / np.asarray(n_train)))
    t_stat = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
    p = 2 * (1 - t_dist.cdf(abs(t_stat), df=k - 1))
    w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return float(d.mean()), float(t_stat), float(p), float(w)
