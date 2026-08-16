"""Test 01 — Schema validation for all 11 fault types."""
import pytest
from src.schema import AnomalyType, ClassifierOutput


def test_all_fault_types_valid_schema(all_payloads):
    """All 11 fault types must produce valid ClassifierOutput objects."""
    assert len(all_payloads) == 11, f"Expected 11 fault types, got {len(all_payloads)}"

    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        assert payload.anomaly_type.value == fault_name
        assert 0.0 <= payload.confidence <= 1.0
        assert len(payload.shap_top3) == 3
        for s in payload.shap_top3:
            assert s.direction in ("above_normal", "below_normal")
            assert isinstance(s.shap_value, float)


def test_jamming_is_valid_anomaly_type():
    """Jamming must be a valid AnomalyType enum member."""
    jt = AnomalyType("Jamming")
    assert jt.value == "Jamming"


def test_shap_sign_matches_direction(all_payloads):
    """Positive shap_value → above_normal, negative → below_normal."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        for s in payload.shap_top3:
            if s.shap_value > 0:
                assert s.direction == "above_normal", (
                    f"{fault_name}: {s.channel} has positive SHAP "
                    f"({s.shap_value}) but direction={s.direction}"
                )
            elif s.shap_value < 0:
                assert s.direction == "below_normal", (
                    f"{fault_name}: {s.channel} has negative SHAP "
                    f"({s.shap_value}) but direction={s.direction}"
                )
