"""Tests for Phase 2 data files."""
import json
import pytest


# ── layer2_output_sessionsplit_phase2.json ─────────────────────────────────────

def test_layer2_phase2_exists():
    import os
    assert os.path.exists("data/processed/layer2_output_sessionsplit_phase2.json")

def test_layer2_phase2_has_6312_records():
    with open("data/processed/layer2_output_sessionsplit_phase2.json") as f:
        d = json.load(f)
    assert len(d) == 6312

def test_layer2_phase2_has_correct_schema():
    with open("data/processed/layer2_output_sessionsplit_phase2.json") as f:
        d = json.load(f)
    required_keys = {"window_index", "anomaly_type", "confidence", "shap_top3", "signal_statistics"}
    for record in d[:10]:
        assert required_keys.issubset(record.keys())

def test_layer2_phase2_has_382_anomalous():
    with open("data/processed/layer2_output_sessionsplit_phase2.json") as f:
        d = json.load(f)
    anomalous = [w for w in d if w.get("anomaly_type") != "Normal"]
    assert len(anomalous) == 382

def test_layer2_phase2_includes_jamming():
    with open("data/processed/layer2_output_sessionsplit_phase2.json") as f:
        d = json.load(f)
    jamming = [w for w in d if w.get("anomaly_type") == "Jamming"]
    assert len(jamming) == 93

def test_layer2_phase2_shap_has_feature_vs_normal():
    """Phase 2 SHAP records must include feature_vs_normal for direction logic."""
    with open("data/processed/layer2_output_sessionsplit_phase2.json") as f:
        d = json.load(f)
    anomalous = [w for w in d if w.get("anomaly_type") != "Normal" and w.get("shap_top3")]
    sample = anomalous[0]["shap_top3"][0]
    assert "feature_vs_normal" in sample
    assert sample["feature_vs_normal"] in ("above_normal_mean", "below_normal_mean")


# ── rca_evidence.json v2 ──────────────────────────────────────────────────────

def test_rca_evidence_v2_has_289_records():
    with open("data/processed/rca_evidence.json") as f:
        d = json.load(f)
    assert len(d) == 289

def test_rca_evidence_v2_no_jamming():
    """Rodina removed Jamming from RCA — it has no 3GPP mapping."""
    with open("data/processed/rca_evidence.json") as f:
        d = json.load(f)
    has_jamming = any(r.get("predicted_fault") == "Jamming" for r in d)
    assert not has_jamming

def test_rca_evidence_v2_has_affected_kpis():
    """Rodina added alignment_table_affected_kpis to v2."""
    with open("data/processed/rca_evidence.json") as f:
        d = json.load(f)
    assert "alignment_table_affected_kpis" in d[0]

def test_rca_evidence_v2_has_expected_kpis():
    with open("data/processed/rca_evidence.json") as f:
        d = json.load(f)
    assert "alignment_table_expected_kpis" in d[0]

def test_rca_evidence_v2_has_10_fault_types():
    with open("data/processed/rca_evidence.json") as f:
        d = json.load(f)
    faults = set(r.get("predicted_fault") for r in d if r.get("predicted_fault"))
    assert len(faults) == 10

def test_rca_evidence_v2_window_index_is_int():
    with open("data/processed/rca_evidence.json") as f:
        d = json.load(f)
    for r in d[:10]:
        assert isinstance(r["window_index"], int)
