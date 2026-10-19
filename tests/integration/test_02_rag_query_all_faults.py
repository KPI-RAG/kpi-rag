"""Test 02 — RAG query construction for all 11 fault types."""
import pytest
from src.schema import ClassifierOutput
from src.rag_query import build_query


def test_build_query_is_evidence_only(all_payloads):
    """build_query() carries KPI evidence but never the fault label (the label is a metadata filter)."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        query = build_query(payload)
        assert payload.anomaly_type.value not in query, (
            f"Query for {fault_name} leaks the fault label"
        )
        assert "Primary affected KPIs" in query, (
            f"Query for {fault_name} missing 'Primary affected KPIs'"
        )
        assert "Signal direction" in query, (
            f"Query for {fault_name} missing 'Signal direction'"
        )


def test_build_query_shap_direction_labels(all_payloads):
    """Antenna Failure query must contain both direction labels."""
    data = all_payloads["Antenna Failure"]
    payload = ClassifierOutput(**data)
    query = build_query(payload)
    # RSRP has negative SHAP → "below normal"
    assert "below normal" in query
    # DL_BLER has positive SHAP → "above normal"
    assert "above normal" in query


def test_build_query_jamming_no_crash(all_payloads):
    """Jamming payload must not crash build_query()."""
    data = all_payloads["Jamming"]
    payload = ClassifierOutput(**data)
    query = build_query(payload)
    assert "Primary affected KPIs" in query and "Jamming" not in query


def test_query_is_nonempty_string(all_payloads):
    """All 11 queries must be non-trivial strings (>50 chars)."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        query = build_query(payload)
        assert len(query) > 50, (
            f"Query for {fault_name} too short: {len(query)} chars"
        )
