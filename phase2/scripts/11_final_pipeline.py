"""Step 11 - Train the final models on all 28 training sessions and produce the Layer 2 JSON.

Choices are read from the previous steps:
    Model 1 : results/10_cascade_reject/results.json -> model1_variant (adopted | adopted+G | jitter)
    Model 2 : Farida's recipe + trend (decision 2026-10-13); reject class only if step 10 promoted it
Outputs (results/final/):
    model1.cbm, model2.cbm, feature_spec.json, normal_means.json,
    layer2_output_sessionsplit_phase2.json  (6,312 test windows, Phase 1 schema),
    validation.json (schema checks + side-by-side counts vs Phase 1's JSON + reload reproducibility)

Run from the phase2 folder:   python scripts/11_final_pipeline.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/.kpi_rag_vendor"))

import numpy as np
from catboost import CatBoostClassifier

from src import config as C
from src import features as F
from src import pipeline as P
from src.data import csv_labels, load_splits
from src.models import anomaly_detector, rebalance_80_20
from src.utils import StepLog, library_versions
from src.v3 import V3, fit_predict_w

log = StepLog("final")
OUT = log.dir
d = load_splits()
labels = csv_labels()
raw = np.load(C.RAW_KPIS, mmap_mode="r")
r10 = json.load(open(os.path.join(C.RESULTS, "10_cascade_reject", "results.json")))
M1 = r10["model1_variant"]
WITH_G = M1 != "adopted"
REJECT = r10["promotion"]["hard1000"]["promoted"]       # reject class only if step 10 promoted it
TAU = None
pol = os.path.join(C.RESULTS, "10c_reject_policy", "policy.json")
if REJECT and os.path.exists(pol):                    # step 10c: reject iff P(Normal) >= tau
    TAU = json.load(open(pol)).get("reject_tau")
log(f"Model 1 variant: {M1} | Model 2: Farida's recipe + trend | reject class: {REJECT} | reject tau: {TAU}")

# ---- Model 1 -----------------------------------------------------------------------------------------
y_bin = (d["train_y"] != C.NORMAL_LABEL).astype(int)
S_tr = np.asarray(raw[d["train_idx"]], dtype=np.float64)
X1 = P.model1_features(S_tr, WITH_G)
keep = rebalance_80_20(np.arange(len(y_bin)), y_bin)
Xk, yk = X1[keep], y_bin[keep]
if M1 == "jitter":
    normal_tr = np.where(y_bin == 0)[0]; st = d["train_sessions"][normal_tr]
    med = np.stack([np.median(X1[normal_tr[st == s]][:, 0:64][:, 0::4], 0) for s in np.unique(st)])
    delta = np.random.default_rng(C.SEED * 7 + 1).normal(0, 1, (len(keep), 16)) * med.std(0)
    Xj = Xk.copy()
    for k in (0, 2, 3):
        Xj[:, [c * 4 + k for c in range(16)]] += delta
    Xk, yk = np.vstack([Xk, Xj]), np.concatenate([yk, yk])
m1 = anomaly_detector(C.SEED, class_weighting=C.MODEL1_CLASS_WEIGHTING)
m1.fit(Xk, yk)
m1.save_model(os.path.join(OUT, "model1.cbm"))
normal_means_m1 = np.nanmean(X1[y_bin == 0], axis=0)   # NaN-safe (undefined co-movement features)
log(f"Model 1 trained on {len(yk)} rows ({X1.shape[1]} features) and saved")

# ---- Model 2 -----------------------------------------------------------------------------------------
fault = y_bin == 1
real_rows, real_y = d["train_idx"][fault], d["train_y"][fault]
eng = V3(d)
eng._load_text([r for r in real_rows if int(r) not in eng.ROW2IDX])
sig = np.stack([eng.real_sig(r) for r in real_rows])            # Phase 1 parser signals (as in the cache)
X2 = P.model2_features(sig)
z = np.load(os.path.join(C.RESULTS, "cache", "synth", f"all_s{C.SEED}_n500_bug.npz"), allow_pickle=True)
Xs, Ys = P.model2_features(z["S"]), z["Y"]
Xtr, ytr = [X2, Xs], [real_y, Ys]
if REJECT:
    oof = np.load(os.path.join(C.RESULTS, "10_cascade_reject", "oof_model1_prob.npy"))
    normal_idx = np.where(y_bin == 0)[0]
    hard = normal_idx[np.argsort(oof[normal_idx])[::-1][:1000]]
    rows = d["train_idx"][hard]
    eng._load_text([r for r in rows if int(r) not in eng.ROW2IDX])
    Xtr.append(P.model2_features(np.stack([eng.real_sig(r) for r in rows]))); ytr.append(np.array([C.NORMAL_LABEL] * 1000))
Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
m2, _ = fit_predict_w(Xtr, ytr, np.ones(len(ytr)), Xtr[:1], C.SEED)
m2.save_model(os.path.join(OUT, "model2.cbm"))
# "normal means" for Model 2 features: from the Phase 1-parsed Normal training windows (sample of 3000)
rng = np.random.default_rng(C.SEED)
samp = rng.choice(np.where(y_bin == 0)[0], 3000, replace=False)
eng._load_text([r for r in d["train_idx"][samp] if int(r) not in eng.ROW2IDX])
normal_means_m2 = np.nanmean(P.model2_features(np.stack([eng.real_sig(r) for r in d["train_idx"][samp]])), axis=0)
log(f"Model 2 trained on {len(ytr)} rows ({Xtr.shape[1]} features; classes {len(np.unique(ytr))}) and saved")

spec = {"model1": {"variant": M1, "features": P.M1_NAMES if WITH_G else P.A_NAMES, "threshold": C.MODEL1_THRESHOLD,
                   "params": C.MODEL1_PARAMS, "class_weighting": C.MODEL1_CLASS_WEIGHTING, "parser": "correct (raw_kpis.npy)"},
        "model2": {"recipe": "Farida v3 500/class + Config B + FFT + G1G2G3", "features": P.M2_NAMES,
                   "reject_class": REJECT, "reject_tau": TAU, "calibrated_confidence": True, "idle_link_flag": [P.IDLE_TX, P.IDLE_PRB, P.IDLE_PKT], "parser": "Phase 1 regex replica (matches the synthetic cache)"},
        "versions": library_versions()}
json.dump(spec, open(os.path.join(OUT, "feature_spec.json"), "w"), indent=2)
json.dump({"model1": normal_means_m1.tolist(), "model2": normal_means_m2.tolist()}, open(os.path.join(OUT, "normal_means.json"), "w"))

# ---- Layer 2 JSON for the 6,312 test windows -----------------------------------------------------------
t0 = time.time()
te = d["test_idx"]
S_te_m1 = np.asarray(raw[te], dtype=np.float64)                  # Model 1: correct parser
eng._load_text([r for r in te if int(r) not in eng.ROW2IDX])
S_te_m2 = np.stack([eng.real_sig(r) for r in te])                 # Model 2: Phase 1 parser (as trained)
A_raw = np.stack([P.config_a(x) for x in S_te_m1])
X1_te = P.model1_features(S_te_m1, WITH_G)
flagged_idx = np.where(m1.predict_proba(X1_te)[:, 1] >= C.MODEL1_THRESHOLD)[0]
import pickle
cal_path = os.path.join(OUT, "model1_calibrator.pkl")               # step 12a: isotonic on OOF probabilities
CAL = pickle.load(open(cal_path, "rb")) if os.path.exists(cal_path) else None
log(f"calibrated confidence: {CAL is not None} | idle-link flag: True")
records = P.run_layer2(m1, m2, S_te_m1, te, A_raw, normal_means_m1, normal_means_m2, WITH_G, S_m2=S_te_m2, reject_tau=TAU,
                       calibrator=CAL, flag_idle=True)
out_json = os.path.join(OUT, "layer2_output_sessionsplit_phase2.json")
json.dump(records, open(out_json, "w"), indent=2)
log(f"Layer 2 JSON written: {len(records)} records ({time.time()-t0:.0f}s)")

# ---- validation --------------------------------------------------------------------------------------
ok, rep = P.validate(records, len(te))
y_te = labels[te]
an = [r for r in records if r["anomaly_type"] != C.NORMAL_LABEL]
pred_map = {r["window_index"]: r["anomaly_type"] for r in records}
pred = np.array([pred_map[int(i)] for i in te])
is_fault = y_te != C.NORMAL_LABEL
rep["test_truth"] = {"faults": int(is_fault.sum()), "normal": int((~is_fault).sum())}
rep["cascade"] = {"flagged_as_fault": int((pred != C.NORMAL_LABEL).sum()),
                  "true_faults_labelled_fault": int(((pred != C.NORMAL_LABEL) & is_fault).sum()),
                  "false_alarms_labelled_fault": int(((pred != C.NORMAL_LABEL) & ~is_fault).sum()),
                  "exact_fault_type_correct": int((pred == y_te)[is_fault].sum()),
                  "catchall_share_cci_mild_among_false_alarms": round(float((pred[~is_fault] == C.CATCHALL_LABEL).sum() / max(((pred != C.NORMAL_LABEL) & ~is_fault).sum(), 1)), 3)}
FAMILY = {"High Network Congestion (Gradual Buildup)": "congestion", "Buffer Overflow (Gradual Buildup)": "congestion",
          "High Network Congestion (Sudden Spike)": "congestion"}
fam = lambda a: np.array([FAMILY.get(x, x) for x in a])
rep["class_grouping_reporting_complement"] = {
    "congestion_family": sorted(FAMILY), "exact_11_class_accuracy_on_true_faults": round(float((pred == y_te)[is_fault].mean()), 4),
    "family_level_accuracy_on_true_faults": round(float((fam(pred) == fam(y_te))[is_fault].mean()), 4),
    "note": "labels in the JSON stay the 11 classes; the family metric is a reporting complement (Farida's next step 3)"}
rep["idle_link_flagged_records"] = sum(1 for r in records if any(P.IDLE_NOTE in e.get("note", "") for e in r["shap_top3"]))
# Phase 1 side by side (Raneem's JSON, copied from Drive into results/final/)
p1 = os.path.join(OUT, "phase1_layer2_reference.json")
if os.path.exists(p1):
    old = json.load(open(p1))
    oan = [r for r in old if r["anomaly_type"] != "Normal"]
    rep["phase1_json"] = {"records": len(old), "anomalous": len(oan), "normal": len(old) - len(oan)}
rep["phase1_report"] = {"anomalous": 1374, "normal": 4938, "true_with_model2_shap": 269, "false_positives": 1105}
# reload reproducibility
m1b = CatBoostClassifier().load_model(os.path.join(OUT, "model1.cbm"))
m2b = CatBoostClassifier().load_model(os.path.join(OUT, "model2.cbm"))
same1 = np.array_equal(m1b.predict_proba(X1_te)[:, 1] >= C.MODEL1_THRESHOLD, m1.predict_proba(X1_te)[:, 1] >= C.MODEL1_THRESHOLD)
X2_flag = P.model2_features(S_te_m2[flagged_idx])
same2 = np.array_equal(m2b.predict(X2_flag).ravel(), m2.predict(X2_flag).ravel())
rep["reload_reproduces_predictions"] = bool(same1 and same2)
json.dump(rep, open(os.path.join(OUT, "validation.json"), "w"), indent=2)
log("validation: " + json.dumps(rep, indent=1))
log.close()
