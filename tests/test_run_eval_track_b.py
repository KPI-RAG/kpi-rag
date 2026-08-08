import pytest
import json
from unittest.mock import patch, MagicMock
from scripts.run_eval_track_b import run_track_b, main
from src.evaluator import TrackBResults

@pytest.fixture
def mock_results():
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

@patch("scripts.run_eval_track_b.load_scores_from_jsonl")
@patch("scripts.run_eval_track_b.compute_track_b")
def test_run_track_b(mock_compute, mock_load, mock_results, tmp_path):
    mock_load.return_value = ["fake_score"]
    mock_compute.return_value = mock_results
    
    out_file = tmp_path / "out.json"
    cfg = {"fake": "config"}
    
    run_track_b("fake.jsonl", str(out_file), cfg)
    
    mock_load.assert_called_once_with("fake.jsonl")
    mock_compute.assert_called_once_with(["fake_score"])
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert "track_b" in data
    assert data["track_b"]["n"] == 10
    assert "track_c" in data
    assert data["track_c"] is None

@patch("scripts.run_eval_track_b.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_b.setup_logging")
@patch("scripts.run_eval_track_b.load_config")
@patch("scripts.run_eval_track_b.run_track_b")
def test_main_success(mock_run, mock_load_config, mock_setup, mock_args):
    args = MagicMock()
    args.scores = "scores.jsonl"
    args.output = "out.json"
    args.config = "config.yaml"
    mock_args.return_value = args
    
    main()
    
    mock_run.assert_called_once()

@patch("scripts.run_eval_track_b.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_b.setup_logging")
@patch("scripts.run_eval_track_b.load_config")
@patch("scripts.run_eval_track_b.run_track_b")
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

@patch("scripts.run_eval_track_b.argparse.ArgumentParser.parse_args")
@patch("scripts.run_eval_track_b.setup_logging")
@patch("scripts.run_eval_track_b.load_config")
@patch("scripts.run_eval_track_b.run_track_b")
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
