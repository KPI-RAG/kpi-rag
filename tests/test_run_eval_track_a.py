import json
import pytest
from scripts.run_eval_track_a import compute_cascade_metrics, run_track_a
from src.config_loader import load_config


def test_compute_cascade_metrics_small():
    preds = {1: "Normal", 2: "Antenna Failure", 3: "Buffer Overflow (Gradual Buildup)", 4: "Jamming", 5: "Normal"}
    truth = {2: "Antenna Failure", 3: "High Network Congestion (Gradual Buildup)", 5: "Jamming"}
    m = compute_cascade_metrics(preds, truth)
    assert (m["true_faults"], m["flagged_as_fault"], m["true_faults_detected"], m["false_alarms"]) == (3, 3, 2, 1)
    assert m["exact_fault_type_correct"] == 1
    assert m["family_accuracy_on_all_true_faults"] == pytest.approx(2 / 3)   # buffer ~ congestion family


def test_track_a_reproduces_validation_json(tmp_path):
    """Recomputed cascade numbers must match Farida's validation.json."""
    res = run_track_a(load_config(), str(tmp_path / "a.json"), with_retrieval=False)["cascade"]
    with open("data/processed/validation.json", encoding="utf-8") as f:
        v = json.load(f)
    assert res["true_faults"] == v["test_truth"]["faults"]
    assert res["flagged_as_fault"] == v["cascade"]["flagged_as_fault"]
    assert res["true_faults_detected"] == v["cascade"]["true_faults_labelled_fault"]
    assert res["exact_fault_type_correct"] == v["cascade"]["exact_fault_type_correct"]
    assert res["family_accuracy_on_all_true_faults"] == pytest.approx(
        v["class_grouping_reporting_complement"]["family_level_accuracy_on_true_faults"], abs=1e-4)
