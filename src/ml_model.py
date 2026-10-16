"""CatBoost model loading and calibrated inference for Layer 1 and Layer 2."""
import logging
import os
import pickle
import numpy as np
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def load_models(cfg: dict):
    """Load model1, model2, and the model1 calibrator.

    Returns:
        tuple[CatBoostClassifier, CatBoostClassifier, object | None]:
            (model1, model2, calibrator)
    """
    models_cfg = cfg.get("models", {})
    model1_path = models_cfg.get("model1_path", "data/models/model1.cbm")
    model2_path = models_cfg.get("model2_path", "data/models/model2.cbm")
    calibrator_path = models_cfg.get(
        "calibrator_path",
        "data/models/model1_calibrator.pkl",
    )

    m1 = CatBoostClassifier()
    m1.load_model(model1_path)

    m2 = CatBoostClassifier()
    m2.load_model(model2_path)

    calibrator = None
    if os.path.exists(calibrator_path):
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
        logger.info("Loaded calibrator from %s", calibrator_path)
    else:
        logger.warning(
            "Calibrator not found at %s — using raw probabilities",
            calibrator_path,
        )

    return m1, m2, calibrator


def compute_confidence(model1: CatBoostClassifier, features, calibrator=None) -> float:
    """Compute calibrated anomaly confidence from model1 binary probabilities.

    Args:
        model1: Binary CatBoostClassifier (Normal vs Anomalous).
        features: Input feature vector/matrix matching model1 feature names (64 features).
        calibrator: Fitted IsotonicRegression calibrator instance or None.

    Returns:
        float: Calibrated probability in [0.0, 1.0].
    """
    proba = model1.predict_proba(features)[0]
    raw_confidence = float(proba[1])
    if calibrator is not None:
        confidence = float(calibrator.predict(np.array([raw_confidence]))[0])
        confidence = float(np.clip(confidence, 0.0, 1.0))
    else:
        confidence = raw_confidence
    return confidence


def calibrate_confidence(raw_confidence: float, calibrator=None) -> float:
    """Calibrate a single raw confidence score using the isotonic calibrator."""
    if calibrator is not None:
        confidence = float(calibrator.predict(np.array([raw_confidence]))[0])
        return float(np.clip(confidence, 0.0, 1.0))
    return float(raw_confidence)


def predict(
    model1: CatBoostClassifier,
    model2: CatBoostClassifier,
    features1,
    features2,
    calibrator=None,
) -> tuple[str, float]:
    """Run full two-layer inference pipeline.

    Args:
        model1: Binary CatBoostClassifier for anomaly detection.
        model2: 11-class CatBoostClassifier for fault classification.
        features1: 64-feature input for model1.
        features2: 496-feature input for model2.
        calibrator: IsotonicRegression model1 calibrator or None.

    Returns:
        tuple[str, float]: (predicted_anomaly_type, calibrated_confidence)
    """
    confidence = compute_confidence(model1, features1, calibrator=calibrator)
    raw_proba = model1.predict_proba(features1)[0]
    raw_confidence = float(raw_proba[1])

    if raw_confidence < 0.5:
        return "Normal", confidence

    m2_proba = model2.predict_proba(features2)[0]
    fault_idx = int(np.argmax(m2_proba))
    predicted_fault = model2.classes_[fault_idx]
    return str(predicted_fault), confidence
