"""Test 05 — LLM explainer for all 11 fault types (mocked LLM)."""
import json
import pytest
from unittest.mock import patch
from src.schema import ClassifierOutput, LLMExplanation
from src.llm_explainer import (
    load_alignment_table,
    explain,
    explain_condition,
)


MOCK_RESPONSE = json.dumps({
    "root_cause": "Fault detected via KPI deviation",
    "3gpp_reference": "TS 38.104",
    "oran_component": "WG4 Open Fronthaul",
    "recommended_action": "Inspect affected component",
    "reasoning_trace": "KPI patterns indicate fault",
})


@pytest.fixture(scope="module")
def alignment():
    return load_alignment_table("configs/alignment_table.json")


NON_JAMMING_FAULTS = [
    "Antenna Failure",
    "Buffer Overflow (Gradual Buildup)",
    "Co-Channel Interference (Mild)",
    "Co-Channel Interference (Severe)",
    "Faulty RF Filters (Temporal)",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
    "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)",
    "Resource Allocation Bugs",
]


def test_explain_all_10_non_jamming_faults(all_payloads, cfg, alignment):
    """explain() must produce valid LLMExplanation for all 10 non-Jamming faults."""
    for fault_name in NON_JAMMING_FAULTS:
        data = all_payloads[fault_name]
        payload = ClassifierOutput(**data)
        with patch("src.llm_explainer.call_llm") as mock:
            mock.return_value = MOCK_RESPONSE
            result = explain(payload, [], cfg, alignment)
        assert isinstance(result, LLMExplanation), (
            f"{fault_name}: returned {type(result)}"
        )
        assert result.root_cause
        assert result.oran_component
        assert isinstance(result.template_generated, bool)
        assert isinstance(result.reference_valid, bool)


def test_explain_jamming_uses_ticket_only(all_payloads, cfg, alignment):
    """Jamming has no alignment entry — explain() must not crash."""
    assert alignment.get("Jamming") is None

    data = all_payloads["Jamming"]
    payload = ClassifierOutput(**data)
    with patch("src.llm_explainer.call_llm") as mock:
        mock.return_value = MOCK_RESPONSE
        result = explain(payload, [], cfg, alignment)
    assert isinstance(result, LLMExplanation)
    # Jamming: reference_valid may be False (TS 38.104 is valid format
    # but may fail alignment table lookup for Jamming)


def test_explain_condition1_all_faults(all_payloads, cfg, alignment):
    """explain_condition(condition=1) must not crash for any fault."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        with patch("src.llm_explainer.call_llm") as mock:
            mock.return_value = MOCK_RESPONSE
            result = explain_condition(payload, [], cfg, alignment, condition=1)
        assert isinstance(result, LLMExplanation), (
            f"C1 {fault_name}: returned {type(result)}"
        )


def test_explain_condition2_all_faults(all_payloads, cfg, alignment):
    """explain_condition(condition=2) must not crash for any fault."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        with patch("src.llm_explainer.call_llm") as mock:
            mock.return_value = MOCK_RESPONSE
            result = explain_condition(payload, [], cfg, alignment, condition=2)
        assert isinstance(result, LLMExplanation), (
            f"C2 {fault_name}: returned {type(result)}"
        )


def test_explain_condition3_all_faults(all_payloads, cfg, alignment):
    """explain_condition(condition=3) must not crash for any fault."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        with patch("src.llm_explainer.call_llm") as mock:
            mock.return_value = MOCK_RESPONSE
            result = explain_condition(payload, [], cfg, alignment, condition=3)
        assert isinstance(result, LLMExplanation), (
            f"C3 {fault_name}: returned {type(result)}"
        )


def test_template_fallback_on_llm_failure(all_payloads, cfg, alignment):
    """If call_llm raises, template_generated=True for all 11 faults."""
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        with patch("src.llm_explainer.call_llm") as mock:
            mock.side_effect = ValueError("LLM unavailable")
            result = explain(payload, [], cfg, alignment)
        assert isinstance(result, LLMExplanation), (
            f"Fallback {fault_name}: returned {type(result)}"
        )
        assert result.template_generated is True, (
            f"{fault_name}: template_generated should be True on LLM failure"
        )
