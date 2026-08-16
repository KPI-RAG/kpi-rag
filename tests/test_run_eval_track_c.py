import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from scripts.run_eval_track_c import (
    run_track_c,
    main,
    build_synthetic_payload,
    TRACK_C_FAULTS,
    FAULT_SIGNAL_PROFILES,
    FAULT_SHAP_PROFILES,
)
from src.schema import AnomalyType, ClassifierOutput, LLMExplanation


def test_track_c_faults_excludes_jamming():
    """Jamming must be excluded from Track C evaluation."""
    fault_values = [ft.value for ft in TRACK_C_FAULTS]
    assert "Jamming" not in fault_values
    assert len(TRACK_C_FAULTS) == 10


def test_fault_profiles_cover_all_track_c_faults():
    """Every Track C fault type must have signal and SHAP profiles."""
    for ft in TRACK_C_FAULTS:
        assert ft.value in FAULT_SIGNAL_PROFILES, f"Missing signal profile: {ft.value}"
        assert ft.value in FAULT_SHAP_PROFILES, f"Missing SHAP profile: {ft.value}"
        assert len(FAULT_SHAP_PROFILES[ft.value]) == 3, f"SHAP must have 3 entries: {ft.value}"


def test_build_synthetic_payload():
    """Synthetic payloads must be valid ClassifierOutput objects."""
    for ft in TRACK_C_FAULTS:
        payload = build_synthetic_payload(ft)
        assert isinstance(payload, ClassifierOutput)
        assert payload.anomaly_type == ft
        assert payload.confidence == 0.85
        assert len(payload.shap_top3) == 3


@patch("scripts.run_eval_track_c.explain_condition")
@patch("scripts.run_eval_track_c.query_from_classifier_output")
@patch("scripts.run_eval_track_c.load_alignment_table")
@patch("scripts.run_eval_track_c.get_collection")
def test_run_track_c(mock_get_col, mock_load_align, mock_query, mock_explain, tmp_path, caplog):
    mock_get_col.return_value = MagicMock()
    mock_load_align.return_value = {"Antenna Failure": {"3gpp_ts": "TS 38.104"}}
    mock_query.return_value = ([], False)

    mock_explanation = LLMExplanation(
        root_cause="test cause",
        gpp_reference="TS 38.104",
        oran_component="WG4",
        recommended_action="test action",
        reasoning_trace="test trace",
        reference_valid=True,
        template_generated=False,
    )
    mock_explain.return_value = mock_explanation

    cfg = {
        "rag": {"embedding_model": "test", "top_k": 5, "cosine_threshold": 0.35,
                "chroma_db_path": "test", "collection_name": "test"},
        "llm": {"backend": "ollama", "max_retries": 2, "temperature": 0.1},
    }

    out_dir = str(tmp_path / "output")

    with caplog.at_level(logging.INFO):
        run_track_c(out_dir, cfg, n_per_fault=1)

    # 10 faults × 1 sample × 3 conditions = 30 explain calls
    assert mock_explain.call_count == 30

    # Verify output files
    explanations_path = tmp_path / "output" / "track_c_explanations.jsonl"
    scores_path = tmp_path / "output" / "track_c_scores_template.jsonl"

    assert explanations_path.exists()
    assert scores_path.exists()

    explanations = [json.loads(l) for l in explanations_path.read_text().strip().split("\n")]
    scores = [json.loads(l) for l in scores_path.read_text().strip().split("\n")]

    assert len(explanations) == 30
    assert len(scores) == 30

    # Each condition should have exactly 10 explanations
    for c in (1, 2, 3):
        cond_items = [e for e in explanations if e["condition"] == c]
        assert len(cond_items) == 10

    # Condition field validated
    conditions_seen = {e["condition"] for e in explanations}
    assert conditions_seen == {1, 2, 3}

    # Scores template has placeholder zeros
    for s in scores:
        assert s["citation_validity"] == 0.0
        assert s["fault_specificity"] == 0.0

    # Log output should contain auto-metrics
    assert "Condition 1:" in caplog.text
    assert "Condition 2:" in caplog.text
    assert "Condition 3:" in caplog.text


@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_success(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.output = "data/processed/"
    args.config = "config.yaml"
    args.n_per_fault = 3
    args.dry_run = False
    mock_args.return_value = args

    main()

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs[1]["n_per_fault"] == 3


@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_dry_run(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.output = "data/processed/"
    args.config = "config.yaml"
    args.n_per_fault = 3
    args.dry_run = True
    mock_args.return_value = args

    main()

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs[1]["n_per_fault"] == 1


@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_value_error(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.output = "out/"
    args.config = "config.yaml"
    args.n_per_fault = 3
    args.dry_run = False
    mock_args.return_value = args

    mock_run.side_effect = ValueError("Empty")

    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1


@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_file_not_found(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.output = "out/"
    args.config = "config.yaml"
    args.n_per_fault = 3
    args.dry_run = False
    mock_args.return_value = args

    mock_run.side_effect = FileNotFoundError("File not found")

    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
