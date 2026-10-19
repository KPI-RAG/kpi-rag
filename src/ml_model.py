"""
Offline inference utilities for KPI-RAG Phase 2.

ARCHITECTURE NOTE:
    Layer 2 (feature extraction + CatBoost + SHAP) is computed offline
    by Farida/Raneem and saved to:
    data/processed/layer2_output_sessionsplit_phase2.json

    The pipeline, dashboard, and Track C consume this precomputed JSON.
    This module is NOT imported by those scripts in the current architecture.

    model1_calibrator.pkl was removed by Farida (Phase 2 final) because
    the IsotonicRegression mapping was degenerate (all probs -> 0.085).
    Raw model probabilities are used instead.

    Feature names in both .cbm files are anonymous integers ('0','1','2').
    Column order must be inferred from data/processed/feature_spec.json.

FUTURE INTEGRATION:
    To wire online inference: implement build_features(window_dict) -> np.ndarray
    using feature_spec.json column ordering, then call predict().
"""
import logging
import numpy as np
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def load_models(cfg: dict) -> tuple[CatBoostClassifier, CatBoostClassifier]:
    """Returns (model1, model2). Calibrator removed — see data/processed/log.txt."""
    models_cfg = cfg.get("models", {})
    m1 = CatBoostClassifier()
    m1.load_model(models_cfg.get("model1_path", "data/models/model1.cbm"))
    logger.info("Loaded model1: %d trees, %d features", m1.tree_count_, len(m1.feature_names_))
    m2 = CatBoostClassifier()
    m2.load_model(models_cfg.get("model2_path", "data/models/model2.cbm"))
    logger.info("Loaded model2: %d trees, %d features, %d classes",
                m2.tree_count_, len(m2.feature_names_), len(m2.classes_))
    return m1, m2


def compute_confidence(model1: CatBoostClassifier, features) -> float:
    """Raw model1 P(anomalous) for one 64-feature row."""
    return float(model1.predict_proba(features)[0][1])


def predict(model1: CatBoostClassifier, model2: CatBoostClassifier, features1, features2) -> tuple[str, float]:
    """Two-stage cascade: ("Normal" or the model2 class, model1 P(anomalous)).

    The returned confidence is model1's P(anomalous) even when a fault class is
    returned; the Phase 2 JSON's ``confidence`` field is defined differently.
    """
    confidence = compute_confidence(model1, features1)
    if confidence < 0.5:
        return "Normal", confidence
    fault_idx = int(np.argmax(model2.predict_proba(features2)[0]))
    return str(model2.classes_[fault_idx]), confidence
