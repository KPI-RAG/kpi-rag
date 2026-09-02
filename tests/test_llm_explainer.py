import pytest
import json
from unittest.mock import patch, MagicMock
from src.schema import ClassifierOutput, RetrievedTicket, LLMExplanation
from src.llm_explainer import (
    load_alignment_table,
    build_prompt,
    build_prompt_condition1,
    build_prompt_condition2,
    call_llm,
    parse_response,
    validate_citation,
    explain,
    explain_condition
)

@pytest.fixture
def alignment():
    return load_alignment_table("configs/alignment_table.json")

@pytest.fixture
def sample_payload():
    return ClassifierOutput(**{
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"}
        ],
        "signal_statistics": {
            "RSRP_mean": -105, "RSRP_std": 3.2,  "RSRP_min": -112, "RSRP_max": -98,
            "DL_BLER_mean": 0.35, "DL_BLER_std": 0.08, "DL_BLER_min": 0.21, "DL_BLER_max": 0.51,
        }
    })


@pytest.fixture
def cfg():
    return {
        "llm": {
            "backend": "ollama",
            "ollama_model": "llama3",
            "temperature": 0.0,
            "max_retries": 2
        }
    }

def test_load_alignment_table(alignment):
    assert len(alignment) == 10
    # All canonical fault names present
    assert "Antenna Failure" in alignment
    assert "Buffer Overflow (Gradual Buildup)" in alignment
    assert "Co-Channel Interference (Mild)" in alignment
    assert "Co-Channel Interference (Severe)" in alignment
    assert "Doppler Shift (Severe)" in alignment
    assert "Faulty Handover Algorithm (Too Frequent)" in alignment
    assert "Faulty RF Filters (Temporal)" in alignment
    assert "High Network Congestion (Gradual Buildup)" in alignment
    assert "High Network Congestion (Sudden Spike)" in alignment
    assert "Resource Allocation Bugs" in alignment
    assert "Jamming" not in alignment
    # Normalized fields (added by load_alignment_table)
    entry = alignment["Antenna Failure"]
    assert "3gpp_ts" in entry
    assert "clause" in entry
    assert "oran_component" in entry
    # Rodina's rich field must be preserved
    assert "causal_mechanism" in entry
    assert entry["3gpp_ts"] == "TS 38.141-1"

def test_build_prompt(sample_payload, alignment):
    # With tickets
    tickets = [
        RetrievedTicket(ticket_id="1", content="Antenna issue text", anomaly_type="Antenna Failure", similarity_score=0.9)
    ]
    prompt = build_prompt(sample_payload, tickets, alignment)
    assert isinstance(prompt, str)
    assert "Antenna Failure" in prompt
    assert "TS 38.141-1" in prompt
    assert "RSRP: below normal" in prompt
    assert "DL_BLER: above normal" in prompt
    
    # Empty tickets
    prompt_empty = build_prompt(sample_payload, [], alignment)
    assert "No similar incidents" in prompt_empty

def test_parse_response():
    clean_json = '{"root_cause":"x", "3gpp_reference":"x", "oran_component":"x", "recommended_action":"x", "reasoning_trace":"x"}'
    parsed = parse_response(clean_json)
    assert parsed["root_cause"] == "x"
    
    md_json = f"```json\n{clean_json}\n```"
    parsed_md = parse_response(md_json)
    assert parsed_md["root_cause"] == "x"
    
    with pytest.raises(ValueError):
        parse_response("just some text")

def test_validate_citation(alignment):
    from src.utils import validate_3gpp_ref

    # --- validate_3gpp_ref format checks ---
    assert validate_3gpp_ref("TS 38.104") is True      # old format still valid
    assert validate_3gpp_ref("TS 38.141-1") is True    # Rodina: Antenna Failure / CCI / RF Filters
    assert validate_3gpp_ref("TR 38.901") is True       # Rodina: Doppler Shift
    assert validate_3gpp_ref("TS 28.552") is True       # Rodina: Congestion rows
    assert validate_3gpp_ref("TS 38.133") is True       # Rodina: Faulty Handover
    assert validate_3gpp_ref("TS 38.321") is True       # Rodina: Buffer Overflow / Resource Bugs
    assert validate_3gpp_ref("TS 39.999") is False      # series out of range
    assert validate_3gpp_ref("38.104") is False         # missing prefix

    # --- validate_citation: ref must pass format AND be in alignment table ---
    # Known valid (all in new alignment table)
    assert validate_citation("TS 38.141-1", alignment) is True
    assert validate_citation("TS 38.321", alignment) is True
    assert validate_citation("TR 38.901", alignment) is True
    assert validate_citation("TS 38.133", alignment) is True
    assert validate_citation("TS 28.552", alignment) is True
    # Invalid: format ok but not in table
    assert validate_citation("TS 38.104", alignment) is False
    assert validate_citation("TS 38.999", alignment) is False
    # Invalid: format fails
    assert validate_citation("TS 39.999", alignment) is False

@patch("src.llm_explainer.call_llm")
def test_explain_path1_success(mock_call, sample_payload, alignment, cfg):
    mock_call.return_value = '''{
        "root_cause": "Physical antenna failure causing RSRP degradation",
        "3gpp_reference": "TS 38.141-1",
        "oran_component": "O-RAN WG4 Open Fronthaul",
        "recommended_action": "Inspect antenna connector and RF cable",
        "reasoning_trace": "RSRP below threshold indicates antenna issue"
    }'''
    
    res = explain(sample_payload, [], cfg, alignment)
    assert isinstance(res, LLMExplanation)
    assert res.template_generated is False
    assert res.reference_valid is True

@patch("src.llm_explainer.call_llm")
def test_explain_path2_fallback(mock_call, sample_payload, alignment, cfg):
    mock_call.side_effect = ValueError("fake error")
    
    res = explain(sample_payload, [], cfg, alignment)
    assert isinstance(res, LLMExplanation)
    assert res.template_generated is True
    assert res.reference_valid is False
    assert res.root_cause == "Antenna Failure detected via KPI deviation"
    # Template fallback uses entry["3gpp_ts"] from alignment (Rodina's table)
    assert res.gpp_reference == "TS 38.141-1"

@patch("src.llm_explainer.call_llm")
def test_explain_path3_hallucinated_reference(mock_call, sample_payload, alignment, cfg):
    mock_call.return_value = '''{
        "root_cause": "Physical antenna failure causing RSRP degradation",
        "3gpp_reference": "TS 38.999",
        "oran_component": "O-RAN WG4 Open Fronthaul",
        "recommended_action": "Inspect antenna connector and RF cable",
        "reasoning_trace": "RSRP below threshold indicates antenna issue"
    }'''
    
    res = explain(sample_payload, [], cfg, alignment)
    assert isinstance(res, LLMExplanation)
    assert res.template_generated is False
    assert res.reference_valid is False


# ─── Track C condition tests ──────────────────────────────────────

@patch("src.llm_explainer.call_llm")
def test_explain_condition1_no_tickets(mock_call, sample_payload, alignment, cfg):
    """Condition 1: label only — prompt has no tickets or standards section."""
    mock_call.return_value = '''{
        "root_cause": "Antenna degradation from physical damage",
        "3gpp_reference": "TS 38.104",
        "oran_component": "WG4 Open Fronthaul",
        "recommended_action": "Check antenna connectors",
        "reasoning_trace": "RSRP drop indicates antenna issue"
    }'''

    res = explain_condition(sample_payload, [], cfg, alignment, condition=1)
    assert isinstance(res, LLMExplanation)
    assert res.template_generated is False

    # Verify prompt content: condition 1 must NOT contain standards or retrieval
    prompt = build_prompt_condition1(sample_payload)
    assert "3GPP" not in prompt
    assert "Retrieved" not in prompt
    assert "Standards reference" not in prompt
    assert "Antenna Failure" in prompt
    assert "RSRP" in prompt


@patch("src.llm_explainer.call_llm")
def test_explain_condition2_tickets_no_table(mock_call, sample_payload, alignment, cfg):
    """Condition 2: label + tickets — prompt has tickets but no alignment table."""
    mock_tickets = [
        RetrievedTicket(
            ticket_id="T1",
            content="Antenna connector loose causing RSRP degradation",
            anomaly_type="Antenna Failure",
            similarity_score=0.42,
        )
    ]
    mock_call.return_value = '''{
        "root_cause": "Loose antenna connector",
        "3gpp_reference": "TS 38.104",
        "oran_component": "WG4",
        "recommended_action": "Tighten connector",
        "reasoning_trace": "Historical ticket confirms connector issue"
    }'''

    res = explain_condition(sample_payload, mock_tickets, cfg, alignment, condition=2)
    assert isinstance(res, LLMExplanation)
    assert res.template_generated is False

    # Verify prompt content: condition 2 has tickets but NO alignment clause
    prompt = build_prompt_condition2(sample_payload, mock_tickets)
    assert "Antenna connector loose" in prompt  # ticket content present
    assert "Retrieved similar incidents" in prompt
    assert "Standards reference" not in prompt
    assert "clause" not in prompt


def test_explain_condition_invalid(sample_payload, alignment, cfg):
    """Invalid condition number must raise ValueError."""
    with pytest.raises(ValueError, match="condition must be 1, 2, or 3"):
        explain_condition(sample_payload, [], cfg, alignment, condition=99)


