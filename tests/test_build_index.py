import pytest
from unittest.mock import patch, MagicMock
from scripts.build_index import build_index, main

@pytest.fixture
def fake_cfg():
    return {
        "data": {
            "raw_path": "fake/raw/path",
            "indices_path": "fake/indices/path"
        },
        "rag": {
            "embedding_model": "fake-model"
        }
    }

@patch("scripts.build_index.load_jsonl_files")
@patch("scripts.build_index.filter_anomalous")
@patch("scripts.build_index.extract_tickets")
@patch("scripts.build_index.apply_train_split")
@patch("scripts.build_index.get_collection")
@patch("scripts.build_index.index_tickets")
def test_build_index(
    mock_index, mock_get_col, mock_apply, mock_extract, mock_filter, mock_load, fake_cfg
):
    mock_load.return_value = ["raw1", "raw2"]
    mock_filter.return_value = ["anom1"]
    mock_extract.return_value = [{"ticket_id": "1"}]
    mock_apply.return_value = [{"ticket_id": "1"}]
    
    mock_col = MagicMock()
    mock_col.name = "test_col"
    mock_get_col.return_value = mock_col
    
    mock_index.return_value = 1
    
    result = build_index(fake_cfg)
    
    mock_load.assert_called_once_with("fake/raw/path")
    mock_filter.assert_called_once_with(["raw1", "raw2"])
    mock_extract.assert_called_once_with(["anom1"])
    mock_apply.assert_called_once_with([{"ticket_id": "1"}], "fake/indices/path")
    mock_get_col.assert_called_once_with(fake_cfg)
    mock_index.assert_called_once_with([{"ticket_id": "1"}], mock_col, "fake-model")
    
    assert result == 1
    assert isinstance(result, int)

@patch("scripts.build_index.argparse.ArgumentParser.parse_args")
@patch("scripts.build_index.setup_logging")
@patch("scripts.build_index.load_config")
@patch("scripts.build_index.build_index")
def test_main_default_config(mock_build, mock_load_config, mock_setup_log, mock_parse_args):
    mock_args = MagicMock()
    mock_args.config = "configs/config.yaml"
    mock_parse_args.return_value = mock_args
    
    mock_build.return_value = 10
    
    main()
    
    mock_load_config.assert_called_once_with("configs/config.yaml")
    mock_setup_log.assert_called_once()
    mock_build.assert_called_once()

@patch("scripts.build_index.argparse.ArgumentParser.parse_args")
@patch("scripts.build_index.setup_logging")
@patch("scripts.build_index.load_config")
@patch("scripts.build_index.build_index")
def test_main_override_config(mock_build, mock_load_config, mock_setup_log, mock_parse_args):
    mock_args = MagicMock()
    mock_args.config = "custom.yaml"
    mock_parse_args.return_value = mock_args
    
    mock_build.return_value = 5
    
    main()
    
    mock_load_config.assert_called_once_with("custom.yaml")
