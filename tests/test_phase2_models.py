"""Tests for Phase 2 model files and calibrator."""
import os
import json
import pickle
import pytest
import math


# ── model1.cbm ────────────────────────────────────────────────────────────────

def test_model1_exists():
    assert os.path.exists("data/models/model1.cbm")

def test_model1_loads():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model1.cbm")
    assert m.tree_count_ > 0

def test_model1_has_64_features():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model1.cbm")
    assert len(m.feature_names_) == 64

def test_model1_is_binary():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model1.cbm")
    assert len(m.classes_) == 2


# ── model2.cbm ────────────────────────────────────────────────────────────────

def test_model2_exists():
    assert os.path.exists("data/models/model2.cbm")

def test_model2_loads():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model2.cbm")
    assert m.tree_count_ > 0

def test_model2_has_496_features():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model2.cbm")
    assert len(m.feature_names_) == 496

def test_model2_has_11_classes():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model2.cbm")
    assert len(m.classes_) == 11

def test_model2_classes_include_jamming():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model2.cbm")
    assert "Jamming" in list(m.classes_)

def test_model2_classes_no_normal():
    """Normal traffic should NOT be a fault class."""
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model("data/models/model2.cbm")
    assert "Normal" not in list(m.classes_)


# ── model1_calibrator.pkl ─────────────────────────────────────────────────────

def test_calibrator_exists():
    assert os.path.exists("data/models/model1_calibrator.pkl")

def test_calibrator_loads():
    with open("data/models/model1_calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    assert cal is not None

def test_calibrator_is_isotonic():
    """Farida used IsotonicRegression as the calibrator."""
    from sklearn.isotonic import IsotonicRegression
    with open("data/models/model1_calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    assert isinstance(cal, IsotonicRegression)


# ── normal_means.json ─────────────────────────────────────────────────────────

def test_normal_means_exists():
    assert os.path.exists("data/processed/normal_means.json")

def test_normal_means_has_model1_and_model2():
    with open("data/processed/normal_means.json") as f:
        nm = json.load(f)
    assert "model1" in nm
    assert "model2" in nm

def test_normal_means_model1_has_64_values():
    with open("data/processed/normal_means.json") as f:
        nm = json.load(f)
    vals = nm["model1"]
    assert isinstance(vals, list)
    assert len(vals) == 64

def test_normal_means_model2_has_496_values():
    with open("data/processed/normal_means.json") as f:
        nm = json.load(f)
    vals = nm["model2"]
    assert isinstance(vals, list)
    assert len(vals) == 496

def test_normal_means_no_nan():
    """NaN in normal_means causes wrong SHAP directions."""
    with open("data/processed/normal_means.json") as f:
        nm = json.load(f)
    for model_key, vals in nm.items():
        if isinstance(vals, list):
            nan_count = sum(1 for v in vals if isinstance(v, float) and math.isnan(v))
            assert nan_count == 0, f"{model_key} has {nan_count} NaN values"


# ── feature_spec.json ─────────────────────────────────────────────────────────

def test_feature_spec_exists():
    assert os.path.exists("data/processed/feature_spec.json")

def test_feature_spec_has_model1_and_model2():
    with open("data/processed/feature_spec.json") as f:
        fs = json.load(f)
    assert "model1" in fs
    assert "model2" in fs

def test_feature_spec_model1_has_64_features():
    with open("data/processed/feature_spec.json") as f:
        fs = json.load(f)
    assert len(fs["model1"]["features"]) == 64

def test_feature_spec_model2_has_496_features():
    with open("data/processed/feature_spec.json") as f:
        fs = json.load(f)
    assert len(fs["model2"]["features"]) == 496

def test_feature_spec_model2_has_patch_features():
    """Phase 2 model2 adds patch0-7 temporal features."""
    with open("data/processed/feature_spec.json") as f:
        fs = json.load(f)
    features = fs["model2"]["features"]
    patch_features = [f for f in features if "patch" in f]
    assert len(patch_features) > 0

def test_feature_spec_model2_has_fft_features():
    """Phase 2 model2 adds FFT spectral features."""
    with open("data/processed/feature_spec.json") as f:
        fs = json.load(f)
    features = fs["model2"]["features"]
    fft_features = [f for f in features if "fft" in f]
    assert len(fft_features) > 0


# ── calibrator integration ───────────────────────────────────────────────────

def test_calibrator_produces_valid_probability():
    """Calibrated confidence must be in [0, 1]."""
    import pickle, numpy as np
    with open("data/models/model1_calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    test_inputs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    outputs = cal.predict(test_inputs)
    assert all(0.0 <= p <= 1.0 for p in outputs), f"Out of range: {outputs}"

def test_calibrator_is_monotonic():
    """Higher raw prob must give higher or equal calibrated prob."""
    import pickle, numpy as np
    with open("data/models/model1_calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    probs = np.linspace(0.1, 0.9, 9)
    calibrated = cal.predict(probs)
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i+1] + 1e-6, \
            f"Not monotonic at {probs[i]:.1f}: {calibrated[i]:.3f} > {calibrated[i+1]:.3f}"

def test_load_models_returns_calibrator():
    """load_models must return 3-tuple including calibrator."""
    from src.config_loader import load_config
    from src.ml_model import load_models
    cfg = load_config()
    result = load_models(cfg)
    assert len(result) == 3, f"Expected 3-tuple (m1, m2, cal), got {len(result)}"
    m1, m2, cal = result
    assert cal is not None, "Calibrator should not be None"

