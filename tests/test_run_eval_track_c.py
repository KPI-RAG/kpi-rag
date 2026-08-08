import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from scripts.run_eval_track_c import run_track_c, main
from src.evaluator import TrackBResults, TrackCResults

@pytest.fixture
def mock_track_b():
    return TrackBResults(
        scores=[],
        mean_citation_validity=4.0,
        mean_fault_specificity=4.0,
        mean_actionability=4.0,
        mean_causal_soundness=4.0,
        mean_overall=4.0,
        citation_validity_rate=0.8,
        n=10,
        meets_threshold=True
    )

@pytest.fixture
def mock_track_c():
    return TrackCResults(
        condition1_mean=2.0,
        condition2_mean=3.0,
        condition3_mean=4.0,
        condition1_citation_rate=0.1,
        condition2_citation_rate=0.2,
        condition3_citation_rate=0.8,
        delta_2v1=1.0,
        delta_3v2=1.0,
        delta_3v1=2.0
    )

@patch("scripts.run_eval_track_c.load_scores_from_jsonl")
@patch("scripts.run_eval_track_c.compute_track_b")
@patch("scripts.run_eval_track_c.compute_track_c")
def test_run_track_c(mock_compute_c, mock_compute_b, mock_load, mock_track_b, mock_track_c, tmp_path, caplog):
    mock_load.return_value = ["fake_score"]
    mock_compute_b.return_value = mock_track_b
    mock_compute_c.return_value = mock_track_c
    
    out_file = tmp_path / "out.json"
    cfg = {"fake": "config"}
    
    with caplog.at_level(logging.INFO):
        run_track_c("fake.jsonl", str(out_file), cfg)
    
    mock_load.assert_called_once_with("fake.jsonl")
    mock_compute_b.assert_called_once_with(["fake_score"])
    mock_compute_c.assert_called_once_with(["fake_score"])
    
    assert "Delta 3v2 (grounding effect)   : +1.00" in caplog.text
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert "track_b" in data
    assert "track_c" in data
    assert data["track_b"]["n"] == 10
    assert data["track_c"]["delta_3v2"] == 1.0

@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_success(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.scores = "scores.jsonl"
    args.output = "out.json"
    args.config = "config.yaml"
    mock_args.return_value = args
    
    main()
    
    mock_run.assert_called_once()

@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_empty_scores(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.scores = "scores.jsonl"
    args.output = "out.json"
    args.config = "config.yaml"
    mock_args.return_value = args
    
    mock_run.side_effect = ValueError("Empty scores")
    
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1

@patch("scripts.run_eval_track_c.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_c.setup_logging")
@patch("scripts.run_eval_track_c.load_config")
@patch("scripts.run_eval_track_c.run_track_c")
def test_main_file_not_found(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.scores = "missing.jsonl"
    args.output = "out.json"
    args.config = "config.yaml"
    mock_args.return_value = args
    
    mock_run.side_effect = FileNotFoundError("File not found")
    
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
