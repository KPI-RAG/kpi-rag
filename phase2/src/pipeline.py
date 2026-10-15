"""Final KPI-RAG Layer 2 pipeline: Model 1 -> Model 2 -> layer2_output JSON (Phase 1 schema).

The schema is byte-compatible with Phase 1 (`layer2_output_sessionsplit.json`, Raneem's cell 65):
    {window_index, anomaly_type, confidence, shap_top3[{channel, feature, shap_value, shap_effect,
     feature_vs_normal[, note]}], signal_statistics{<KPI>_mean/_std/_min/_max x 16}}
Changes vs Phase 1 are internal only: correct parser, no double weighting, trend features,
synthetic-trained Model 2, optional reject class. One behavioural fix: Phase 1 chose the SHAP source
by GROUND TRUTH (true anomalies -> Model 2 SHAP, false positives -> Model 1 SHAP), which is not
available at inference; here the choice follows the models' own output: a window Model 2 rejects as
"Normal" gets Model 1 SHAP with the Phase 1 note, every other flagged window gets Model 2 SHAP.
"""
import json
import os

import numpy as np

from . import config as C
from . import features as F
from .v3 import IDX_B_FFT, full_feats

CHANNELS = ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR", "TX_Bytes", "RX_Bytes",
            "Estimated_UL_Buffer", "PRBs_DL_Current", "PRBs_UL_Current", "PRB_Utilization_DL",
            "PRB_Utilization_UL", "UL_NumberOfPackets", "DL_NumberOfPackets"]
STATS = ["mean", "std", "min", "max"]
A_NAMES = [f"{c}_{s}" for c in CHANNELS for s in STATS]
PATCH_NAMES = [f"{c}_patch{p}_{s}" for p in range(8) for c in CHANNELS for s in ("mean", "std")]
FFT_NAMES = [f"{c}_fft_{k}" for c in CHANNELS for k in ("band1", "band2", "band3", "band4", "energy")]
G_NAMES = F.names()
M2_NAMES = [([*A_NAMES, *PATCH_NAMES][i]) for i in range(320)] + FFT_NAMES + G_NAMES   # B+FFT+G = 496
M1_NAMES = A_NAMES + G_NAMES                                                              # A+G = 160

# Idle-link indicator (step 12a): fixed physical thresholds, not fitted on labels.
IDLE_TX, IDLE_PRB, IDLE_PKT = 1000.0, 1.0, 1.0        # bytes/sample, % PRB utilisation DL, packets/sample
IDLE_NOTE = "idle_link_ambiguous"


def idle_link(A):
    """True for windows with (almost) no traffic: TX_Bytes mean < IDLE_TX, PRB_Util_DL mean < IDLE_PRB,
    UL+DL packets mean < IDLE_PKT. A = Config A matrix (n, 64) in A_NAMES order."""
    col = {n: i for i, n in enumerate(A_NAMES)}
    return (A[:, col["TX_Bytes_mean"]] < IDLE_TX) & (A[:, col["PRB_Utilization_DL_mean"]] < IDLE_PRB) &            (A[:, col["UL_NumberOfPackets_mean"]] + A[:, col["DL_NumberOfPackets_mean"]] < IDLE_PKT)


_SHORT = {"Est_UL_Buffer": "Estimated_UL_Buffer", "PRBs_DL": "PRBs_DL_Current", "PRBs_UL": "PRBs_UL_Current",
          "PRB_Util_DL": "PRB_Utilization_DL", "PRB_Util_UL": "PRB_Utilization_UL",
          "UL_Packets": "UL_NumberOfPackets", "DL_Packets": "DL_NumberOfPackets"}


def channel_of(feature_name):
    """KPI channel a feature belongs to, or None for cross-channel features (n_shift_*, onset_*, diff_*).
    corr_<A>_<B> features are attributed to their first channel A."""
    name = feature_name[5:] if feature_name.startswith("corr_") else feature_name
    for short, full in sorted(_SHORT.items(), key=lambda kv: -len(kv[0])):   # G features use short names
        if name.startswith(short):
            return full
    for ch in sorted(CHANNELS, key=len, reverse=True):      # longest first: "PRB_Utilization_DL" before "PRBs_DL"
        if name.startswith(ch):
            return ch
    return None


def config_a(x):
    return np.stack([x.mean(1), x.std(1), x.min(1), x.max(1)], 1).reshape(-1)


def model1_features(S, with_G=True):
    A = np.stack([config_a(x) for x in S])
    return np.hstack([A, F.from_signals(S)]) if with_G else A


def model2_features(S):
    full = np.stack([full_feats(x) for x in S])
    return np.hstack([full[:, IDX_B_FFT], F.from_signals(S)])


def _top3(sv, x_row, names, normal_means, effect_fn, note=None):
    """Top-3 KPI contributors: features with no single channel (cross-KPI co-movement) are skipped so
    that `channel` is always one of the 16 KPI names, as in Phase 1."""
    abs_v = np.abs(sv)
    out = []
    for i in np.argsort(abs_v)[::-1]:
        if channel_of(names[i]) is None:
            continue
        if len(out) == 3:
            break
        if not np.isfinite(x_row[i]) or not np.isfinite(normal_means[i]):
            continue                                     # undefined feature (e.g. correlation of a flat KPI): not reportable
        e = {"channel": channel_of(names[i]), "feature": names[i], "shap_value": float(abs_v[i]),
             "shap_effect": effect_fn(sv[i]),
             "feature_vs_normal": "above_normal_mean" if x_row[i] > normal_means[i] else "below_normal_mean"}
        if note:
            e["note"] = note
        out.append(e)
    return out


def run_layer2(m1, m2, S, window_index, A_raw, normal_means_m1, normal_means_m2, with_G=True,
               threshold=C.MODEL1_THRESHOLD, reject_label=C.NORMAL_LABEL, S_m2=None, reject_tau=None,
               calibrator=None, flag_idle=True):
    """Build the Layer 2 records for windows S (n,16,128). A_raw = Config A stats for signal_statistics.
    S_m2: optional signals for Model 2 (Phase 1 parser, as Model 2 was trained); defaults to S.
    reject_tau: if Model 2 has a reject class, reject iff P(Normal) >= tau (None = argmax)."""
    import shap
    S2 = S if S_m2 is None else S_m2
    X1 = model1_features(S, with_G)
    p1 = m1.predict_proba(X1)
    flagged = p1[:, 1] >= threshold                      # decision on the raw probability (as validated)
    if calibrator is not None:                           # confidence reported as a calibrated probability
        pc = np.clip(calibrator.predict(p1[:, 1]), 0, 1)
        p1 = np.column_stack([1 - pc, pc])
    idle = idle_link(X1[:, :64]) if flag_idle else np.zeros(len(S), bool)
    records = []
    X2 = model2_features(S2[flagged]) if flagged.any() else None
    if X2 is not None:
        prob2 = m2.predict_proba(X2)
        classes2 = [str(c) for c in m2.classes_]
        if reject_tau is not None and reject_label in classes2:
            ni = classes2.index(reject_label)
            best_fault = np.array(classes2)[np.argmax(np.where(np.arange(len(classes2)) == ni, -1, prob2), axis=1)]
            pred2 = np.where(prob2[:, ni] >= reject_tau, reject_label, best_fault)
        else:
            pred2 = m2.predict(X2).ravel().astype(str)
        ex2 = shap.TreeExplainer(m2)
        sv2 = ex2.shap_values(X2)                       # (n, n_feat, n_class)
        ex1 = shap.TreeExplainer(m1)
        sv1 = ex1.shap_values(X1[flagged])
        if np.ndim(sv1) == 3:
            sv1 = sv1[:, :, 1]
    j = 0
    for i in range(len(S)):
        stats = {f"{c}_{s}": float(A_raw[i, k * 4 + t]) for k, c in enumerate(CHANNELS) for t, s in enumerate(STATS)}
        rec = {"window_index": int(window_index[i]), "anomaly_type": C.NORMAL_LABEL,
               "confidence": float(p1[i].max()), "shap_top3": [], "signal_statistics": stats}
        if flagged[i]:
            fault = pred2[j]
            ci = classes2.index(fault)
            if fault == reject_label:                   # Model 2 rejected the alarm
                rec["confidence"] = float(prob2[j, ci])
                rec["shap_top3"] = _top3(sv1[j], X1[flagged][j], M1_NAMES if with_G else A_NAMES, normal_means_m1,
                                         lambda v: "toward_anomaly_detection", note="false_positive_model1_shap")
            else:
                rec["anomaly_type"] = fault
                rec["confidence"] = float(prob2[j, ci])
                rec["shap_top3"] = _top3(sv2[j, :, ci], X2[j], M2_NAMES, normal_means_m2,
                                         lambda v: "toward_predicted_class" if v > 0 else "away_from_predicted_class")
            if idle[i]:                                  # label kept; evidence is "no traffic" only -> review flag
                for e in rec["shap_top3"]:
                    e["note"] = (e["note"] + ";" + IDLE_NOTE) if e.get("note") else IDLE_NOTE
            j += 1
        records.append(rec)
    return records


def validate(records, n_expected):
    """The checks Raneem's cell 65 ran, plus schema checks. Returns (ok, report dict)."""
    rep = {"records": len(records), "expected": n_expected}
    ok = len(records) == n_expected
    keys = {"window_index", "anomaly_type", "confidence", "shap_top3", "signal_statistics"}
    ok &= all(set(r) == keys for r in records)
    rep["schema_ok"] = ok
    an = [r for r in records if r["anomaly_type"] != C.NORMAL_LABEL]
    rep["normal"] = len(records) - len(an); rep["anomalous"] = len(an)
    eff = {"toward_predicted_class", "away_from_predicted_class", "toward_anomaly_detection"}
    dirs = {"above_normal_mean", "below_normal_mean"}
    bad = 0
    for r in records:
        for e in r["shap_top3"]:
            bad += (e["channel"] not in CHANNELS) + (e["shap_effect"] not in eff) + (e["feature_vs_normal"] not in dirs)
        bad += (not (0.0 <= r["confidence"] <= 1.0)) + (len(r["signal_statistics"]) != 64)
        bad += (r["anomaly_type"] != C.NORMAL_LABEL and len(r["shap_top3"]) != 3)   # every fault label has 3 entries
        bad += (len(r["shap_top3"]) not in (0, 3))
    rep["invalid_fields"] = int(bad)
    rep["rejected_alarms_with_model1_shap"] = sum(1 for r in records if r["anomaly_type"] == C.NORMAL_LABEL and r["shap_top3"])
    ok &= bad == 0
    rep["ok"] = bool(ok)
    return ok, rep
