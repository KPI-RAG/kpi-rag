import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from scripts.run_pipeline import run, main
from src.schema import ClassifierOutput, LLMExplanation

@pytest.fixture
def fake_cfg():
    return {"rag": {"test": 1}}

@pytest.fixture
def sample_payload():
    return ClassifierOutput(**{
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "direction": "below_normal"},
            {"channel": "DL_BLER", "shap_value":  0.28, "direction": "above_normal"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "direction": "below_normal"}
        ],
        "signal_statistics": {
            "RSRP":    {"mean": -105, "std": 3.2,  "min": -112, "max": -98},
            "DL_BLER": {"mean": 0.35, "std": 0.08, "min": 0.21, "max": 0.51}
        }
    })

@pytest.fixture
def mock_explanation():
    return LLMExplanation(
        root_cause="Physical antenna failure causing RSRP degradation",
        gpp_reference="TS 38.104",
        oran_component="O-RAN WG4 Open Fronthaul",
        recommended_action="Inspect antenna connector and RF cable",
        reasoning_trace="RSRP below threshold indicates antenna issue",
        reference_valid=True,
        template_generated=False
    )

@patch("scripts.run_pipeline.get_collection")
@patch("scripts.run_pipeline.query_from_classifier_output")
@patch("scripts.run_pipeline.load_alignment_table")
@patch("scripts.run_pipeline.explain")
def test_run_low_conf(mock_explain, mock_load_align, mock_query, mock_get_col, sample_payload, fake_cfg, mock_explanation, caplog):
    mock_get_col.return_value = "fake_col"
    mock_query.return_value = (["ticket1"], True)
    mock_load_align.return_value = {"fake": "align"}
    mock_explain.return_value = mock_explanation
    
    with caplog.at_level(logging.WARNING):
        res = run(sample_payload, fake_cfg)
        
    mock_get_col.assert_called_once_with(fake_cfg)
    mock_query.assert_called_once_with(sample_payload, "fake_col", fake_cfg)
    mock_load_align.assert_called_once_with("configs/alignment_table.json")
    mock_explain.assert_called_once_with(sample_payload, ["ticket1"], fake_cfg, {"fake": "align"})
    
    assert isinstance(res, LLMExplanation)
    assert res == mock_explanation
    assert "Low retrieval confidence for Antenna Failure" in caplog.text

@patch("scripts.run_pipeline.get_collection")
@patch("scripts.run_pipeline.query_from_classifier_output")
@patch("scripts.run_pipeline.load_alignment_table")
@patch("scripts.run_pipeline.explain")
def test_run_high_conf(mock_explain, mock_load_align, mock_query, mock_get_col, sample_payload, fake_cfg, mock_explanation, caplog):
    mock_get_col.return_value = "fake_col"
    mock_query.return_value = (["ticket1"], False)
    mock_load_align.return_value = {"fake": "align"}
    mock_explain.return_value = mock_explanation
    
    with caplog.at_level(logging.WARNING):
        res = run(sample_payload, fake_cfg)
        
    assert "Low retrieval confidence for Antenna Failure" not in caplog.text

@patch("scripts.run_pipeline.argparse.ArgumentParser.parse_args")
@patch("scripts.run_pipeline.setup_logging")
@patch("scripts.run_pipeline.load_config")
@patch("scripts.run_pipeline.run")
def test_main_success(mock_run, mock_load_config, mock_setup, mock_args, tmp_path, mock_explanation):
    in_file = tmp_path / "in.json"
    out_file = tmp_path / "out.json"
    cfg_file = "custom.yaml"
    
    payload_dict = {
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "direction": "below_normal"},
            {"channel": "DL_BLER", "shap_value":  0.28, "direction": "above_normal"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "direction": "below_normal"}
        ],
        "signal_statistics": {
            "RSRP":    {"mean": -105, "std": 3.2,  "min": -112, "max": -98},
            "DL_BLER": {"mean": 0.35, "std": 0.08, "min": 0.21, "max": 0.51}
        }
    }
    with open(in_file, "w") as f:
        json.dump(payload_dict, f)
        
    args = MagicMock()
    args.input = str(in_file)
    args.output = str(out_file)
    args.config = cfg_file
    mock_args.return_value = args
    
    mock_run.return_value = mock_explanation
    
    main()
    
    mock_load_config.assert_called_once_with(cfg_file)
    
    with open(out_file, "r") as f:
        out_data = json.load(f)
        
    assert "root_cause" in out_data
    assert out_data["root_cause"] == mock_explanation.root_cause

@patch("scripts.run_pipeline.argparse.ArgumentParser.parse_args")
@patch("scripts.run_pipeline.setup_logging")
@patch("scripts.run_pipeline.load_config")
def test_main_missing_input(mock_load_config, mock_setup, mock_args, tmp_path):
    args = MagicMock()
    args.input = str(tmp_path / "nonexistent.json")
    args.output = "out.json"
    args.config = "config.yaml"
    mock_args.return_value = args
    
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1

@patch("scripts.run_pipeline.argparse.ArgumentParser.parse_args")
@patch("scripts.run_pipeline.setup_logging")
@patch("scripts.run_pipeline.load_config")
def test_main_validation_error(mock_load_config, mock_setup, mock_args, tmp_path):
    in_file = tmp_path / "in.json"
    with open(in_file, "w") as f:
        f.write('{"invalid": "schema"}')
        
    args = MagicMock()
    args.input = str(in_file)
    args.output = "out.json"
    args.config = "config.yaml"
    mock_args.return_value = args
    
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
