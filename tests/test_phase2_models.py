"""Tests for Phase 2 model files (the calibrator was removed)."""
import os
import json
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


# ── model1_calibrator.pkl (removed) ───────────────────────────────────────────

def test_calibrator_removed():
    """Farida removed degenerate calibrator in Phase 2 final."""
    assert not os.path.exists("data/models/model1_calibrator.pkl"), \
        "Calibrator should not exist — was removed as degenerate"


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


# ── load_models ──────────────────────────────────────────────────────────────

def test_load_models_returns_two_tuple():
    """load_models returns (m1, m2) — no calibrator."""
    from src.config_loader import load_config
    from src.ml_model import load_models
    cfg = load_config()
    result = load_models(cfg)
    assert len(result) == 2, f"Expected (m1, m2) tuple, got {len(result)} items"
    m1, m2 = result
    assert m1 is not None
    assert m2 is not None


