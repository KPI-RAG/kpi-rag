"""Test 07 — Full end-to-end pipeline for all 11 fault types."""
import json
import pytest
from unittest.mock import patch, MagicMock
from src.schema import ClassifierOutput, LLMExplanation, RetrievedTicket
from src.kg_indexer import get_collection
from scripts.run_pipeline import run


MOCK_RESPONSE = json.dumps({
    "root_cause": "Fault detected via KPI deviation",
    "3gpp_reference": "TS 38.104",
    "oran_component": "WG4 Open Fronthaul",
    "recommended_action": "Inspect affected component",
    "reasoning_trace": "KPI patterns indicate fault",
})


@pytest.fixture(scope="module")
def has_chromadb(cfg):
    """Check if real ChromaDB index is available."""
    try:
        collection = get_collection(cfg)
        return collection.count() > 0
    except Exception:
        return False


def _run_with_mocks(payload, cfg, has_chromadb):
    """Run pipeline with mocked LLM, optionally mocked ChromaDB."""
    if has_chromadb:
        # Real ChromaDB, mocked LLM only
        with patch("src.llm_explainer.call_llm") as mock_llm:
            mock_llm.return_value = MOCK_RESPONSE
            return run(payload, cfg)
    else:
        # Mock both ChromaDB and LLM
        mock_tickets = [
            RetrievedTicket(
                ticket_id="mock_001",
                content="Mock ticket for testing pipeline flow",
                anomaly_type=payload.anomaly_type.value,
                similarity_score=0.85,
            )
        ]
        with patch("src.llm_explainer.call_llm") as mock_llm, \
             patch("scripts.run_pipeline.get_collection") as mock_col, \
             patch("scripts.run_pipeline.query_from_classifier_output") as mock_query:
            mock_llm.return_value = MOCK_RESPONSE
            mock_col.return_value = MagicMock()
            mock_query.return_value = (mock_tickets, False)
            return run(payload, cfg)


def test_full_pipeline_antenna_failure(cfg, all_payloads, has_chromadb):
    """Full pipeline must produce LLMExplanation for Antenna Failure."""
    data = all_payloads["Antenna Failure"]
    payload = ClassifierOutput(**data)
    result = _run_with_mocks(payload, cfg, has_chromadb)

    assert isinstance(result, LLMExplanation)
    assert result.root_cause
    assert result.oran_component
    assert isinstance(result.template_generated, bool)
    assert isinstance(result.reference_valid, bool)


def test_full_pipeline_all_11_faults(cfg, all_payloads, has_chromadb):
    """Full pipeline must work for all 11 fault types without crashing."""
    results = {}
    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        result = _run_with_mocks(payload, cfg, has_chromadb)
        results[fault_name] = result
        assert isinstance(result, LLMExplanation), (
            f"{fault_name} returned {type(result)}"
        )

    print(f"\nPipeline results for {len(results)} fault types:")
    for fault, r in results.items():
        print(f"  {fault}: valid={r.reference_valid} template={r.template_generated}")


def test_jamming_pipeline_no_crash(cfg, all_payloads, has_chromadb):
    """Jamming has no alignment entry — pipeline must complete."""
    data = all_payloads["Jamming"]
    payload = ClassifierOutput(**data)
    result = _run_with_mocks(payload, cfg, has_chromadb)

    assert isinstance(result, LLMExplanation)
    # gpp_reference may be empty for Jamming — that is OK


def test_pipeline_low_confidence_warning(cfg, all_payloads):
    """Pipeline must produce LLMExplanation even with empty retrieval."""
    data = all_payloads["Antenna Failure"]
    payload = ClassifierOutput(**data)

    with patch("src.llm_explainer.call_llm") as mock_llm, \
         patch("scripts.run_pipeline.get_collection") as mock_col, \
         patch("scripts.run_pipeline.query_from_classifier_output") as mock_query:
        mock_llm.return_value = MOCK_RESPONSE
        mock_col.return_value = MagicMock()
        mock_query.return_value = ([], True)  # empty tickets, low_conf=True
        result = run(payload, cfg)

    assert isinstance(result, LLMExplanation)
