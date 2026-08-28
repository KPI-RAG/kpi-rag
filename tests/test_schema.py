import pytest
from pydantic import ValidationError
from src.schema import AnomalyType, ClassifierOutput, LLMExplanation


def test_anomaly_type_length():
    assert len(AnomalyType) == 11


@pytest.mark.parametrize("value", [
    "Jamming",
    "Antenna Failure"
])
def test_anomaly_type_valid(value):
    assert AnomalyType(value) == value


def test_anomaly_type_invalid():
    with pytest.raises(ValueError):
        AnomalyType("Normal")


def test_classifier_output_valid():
    payload = {
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"}
        ],
        "signal_statistics": {
            "RSRP_mean": -105, "RSRP_std": 3.2, "RSRP_min": -112, "RSRP_max": -98,
            "DL_BLER_mean": 0.35, "DL_BLER_std": 0.08, "DL_BLER_min": 0.21, "DL_BLER_max": 0.51,
        }
    }
    obj = ClassifierOutput(**payload)
    assert obj.anomaly_type == "Antenna Failure"


def test_classifier_output_predicted_fault_type_alias():
    """predicted_fault_type must be accepted as alias for anomaly_type."""
    payload = {
        "predicted_fault_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"},
        ],
        "signal_statistics": {}
    }
    obj = ClassifierOutput(**payload)
    assert obj.anomaly_type == "Antenna Failure"


def test_classifier_output_invalid_confidence():
    payload = {
        "anomaly_type": "Antenna Failure",
        "confidence": 1.5,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"}
        ],
        "signal_statistics": {}
    }
    with pytest.raises(ValidationError):
        ClassifierOutput(**payload)


def test_classifier_output_invalid_shap_len():
    payload = {
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"}
        ],
        "signal_statistics": {}
    }
    with pytest.raises(ValidationError):
        ClassifierOutput(**payload)


def test_llm_explanation_valid():
    payload = {
        "root_cause": "Test root cause",
        "gpp_reference": "TS 38.300",
        "oran_component": "O-DU",
        "recommended_action": "Restart DU",
        "reasoning_trace": "Because of X and Y.",
        "reference_valid": True,
        "template_generated": False
    }
    obj = LLMExplanation(**payload)
    assert isinstance(obj.template_generated, bool)


def test_validate_3gpp_ref():
    from src.utils import validate_3gpp_ref
    # Existing formats — must still pass
    assert validate_3gpp_ref("TS 38.104") is True
    assert validate_3gpp_ref("TS 39.999") is False
    assert validate_3gpp_ref("38.104") is False
    # Rodina's new reference formats — must now pass
    assert validate_3gpp_ref("TS 38.141-1") is True
    assert validate_3gpp_ref("TR 38.901") is True
    assert validate_3gpp_ref("TS 28.552") is True
    assert validate_3gpp_ref("TS 38.133") is True
    assert validate_3gpp_ref("TS 38.321") is True
    assert validate_3gpp_ref("TS 38.314") is True