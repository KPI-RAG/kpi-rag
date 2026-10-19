"""Regression tests for the third end-to-end review (one test per fixed defect)."""
import json
from unittest.mock import patch

import pytest

from src.llm_explainer import (
    _is_non_retryable, _is_rate_limit, _build_shap_summary, _run_with_retry, parse_response,
    LLMResponseError, load_alignment_table,
)
from src.schema import ClassifierOutput


def _reply(ref):
    return json.dumps({"root_cause": "x", "3gpp_reference": ref, "oran_component": "x",
                       "recommended_action": "x", "reasoning_trace": "x"})


@pytest.fixture
def payload():
    return ClassifierOutput(anomaly_type="Antenna Failure", confidence=0.9, signal_statistics={}, shap_top3=[
        {"channel": "RSRP", "shap_value": 0.42, "feature_vs_normal": "below_normal_mean", "shap_effect": "toward_predicted_class"},
        {"channel": "DL_MCS", "shap_value": 0.20, "feature_vs_normal": "below_normal_mean", "shap_effect": "away_from_predicted_class"},
        {"channel": "UL_SNR", "shap_value": 0.10, "feature_vs_normal": "above_normal_mean"},
    ])


@pytest.mark.parametrize("raw,expected", [
    ("TS38.331", "TS 38.331"), ("ts 38.331", "TS 38.331"), ("TS  38.331", "TS 38.331"),
    ("3GPP TR 38.901 §7.6.6", "TR 38.901"), ("TS 38.141-1, TS 38.104", "TS 38.141-1"),
])
def test_citation_formats_are_normalised(raw, expected):
    assert parse_response(_reply(raw))["gpp_reference"] == expected


@pytest.mark.parametrize("bad", [None, ["TS 38.331"], 38.331])
def test_non_string_reference_is_a_malformed_reply(bad):
    with pytest.raises(LLMResponseError):
        parse_response(_reply(bad))


class _Err(Exception):
    def __init__(self, msg, status=None, code=None):
        super().__init__(msg)
        self.status_code, self.code = status, code


@pytest.mark.parametrize("err,non_retryable", [
    (_Err("model not found", status=404), True),
    (_Err("API key not valid", code=400), True),     # google-genai ClientError
    (KeyError("groq_model"), True),
    (_Err("unauthorized", status=401), True),
    (_Err("rate limited", status=429), False),
    (_Err("server error", status=503), False),
    (_Err("timeout", status=408), False),
    (LLMResponseError("No valid JSON found in response"), False),
])
def test_retry_classification(err, non_retryable):
    assert _is_non_retryable(err) is non_retryable


def test_malformed_reply_mentioning_429_is_not_a_rate_limit():
    assert _is_rate_limit(LLMResponseError("value 429 out of range")) is False
    assert _is_rate_limit(Exception("429 Too Many Requests")) is True


@patch("src.llm_explainer.time.sleep")
@patch("src.llm_explainer.call_llm")
def test_long_retry_after_gives_up_instead_of_sleeping(mock_call, mock_sleep, payload):
    mock_call.side_effect = _Err("429 RESOURCE_EXHAUSTED. Please retry in 3600s.", code=429)
    cfg = {"llm": {"max_retries": 3, "backoff_max_s": 60}}
    res = _run_with_retry("p", cfg, payload, load_alignment_table())
    assert res.template_generated is True
    assert "3600" in res.fallback_reason
    assert mock_call.call_count == 1 and not mock_sleep.called


@patch("src.llm_explainer.call_llm")
def test_zero_max_retries_still_calls_the_llm(mock_call, payload):
    mock_call.return_value = _reply("TS 38.141-1")
    res = _run_with_retry("p", {"llm": {"max_retries": 0}}, payload, load_alignment_table())
    assert mock_call.call_count == 1 and res.template_generated is False


def test_shap_summary_shows_magnitude_and_effect(payload):
    s = _build_shap_summary(payload)
    assert "RSRP: below normal (|SHAP|=0.42, toward predicted class)" in s
    assert "away from predicted class" in s
    assert "SHAP=+" not in s


def test_rca_block_has_no_standards_or_evidence_status():
    from src.rca_loader import RCALoader
    loader = RCALoader("data/processed/rca_evidence.json")
    handover = next(w for w, r in loader._index.items() if r["predicted_fault"].startswith("Faulty Handover"))
    block = loader.get_prompt_context(handover)
    assert "TS 38.133" not in block and "Standards grounding" not in block   # alignment table is the single source
    assert "observed_only" not in block and "supporting" not in block
    assert "Observed KPI means" in block


def test_track_c_baselines():
    from scripts.run_eval_track_c import citation_baselines
    al = load_alignment_table()
    recs = [
        {"window_index": 1, "true_fault": "Antenna Failure", "predicted_fault": "Antenna Failure"},
        {"window_index": 2, "true_fault": "Co-Channel Interference (Mild)", "predicted_fault": "Faulty RF Filters (Temporal)"},
        {"window_index": 3, "true_fault": "Faulty Handover Algorithm (Too Frequent)", "predicted_fault": "Normal"},
    ]
    b = citation_baselines(recs, al)
    assert b["n"] == 3
    assert b["copy_predicted_row_correct"] == 2   # RF Filters' row (TS 38.141-1) also covers CCI Mild
    assert b["constant_best_standard"] == "TS 38.141-1" and b["constant_correct"] == 2


def test_scores_without_reference_valid_are_rejected(tmp_path):
    from src.evaluator import load_scores_from_jsonl
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"explanation_id": "w1_c1", "condition": 1, "fault_type": "Antenna Failure",
                             "citation_validity": 4, "fault_specificity": 4, "actionability": 4,
                             "causal_soundness": 4}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference_valid"):
        load_scores_from_jsonl(str(p))


def test_build_index_refuses_to_clear_with_no_tickets():
    from scripts import build_index as bi
    cfg = {"data": {"raw_path": "x", "indices_path": "y", "train_idx_file": "z", "processed_path": "p"},
           "rag": {"embedding_model": "m"}}
    with patch.object(bi, "load_jsonl_files", return_value=[]), patch.object(bi, "apply_train_split", return_value=[]), \
         patch.object(bi, "get_collection") as get_col, patch.object(bi, "clear_collection") as clear:
        with pytest.raises(RuntimeError, match="refusing"):
            bi.build_index(cfg, rebuild=True)
        clear.assert_not_called()
        get_col.assert_not_called()


def test_cascade_metrics_handle_nothing_flagged():
    from scripts.run_eval_track_a import compute_cascade_metrics
    m = compute_cascade_metrics({1: "Normal", 2: "Normal"}, {1: "Antenna Failure"})
    assert m["binary_recall"] == 0.0 and m["binary_precision"] is None


def test_dashboard_browse_class_with_one_window(monkeypatch):
    """Phase 2 has a single Sudden Spike alarm; the window picker used to crash on it (slider min == max)."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    from pathlib import Path
    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Browse Dataset Windows").run()
    at.sidebar.selectbox[0].set_value("High Network Congestion (Sudden Spike)").run()
    assert not at.exception
