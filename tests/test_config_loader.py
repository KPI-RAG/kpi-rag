import pytest
import yaml
from src.config_loader import load_config

def test_load_config_valid():
    cfg = load_config()
    assert isinstance(cfg, dict)
    for key in ["rag", "llm", "data", "evaluation", "dashboard"]:
        assert key in cfg
    assert cfg["rag"]["embedding_model"] == "all-MiniLM-L6-v2"
    assert cfg["data"]["random_state"] == 42

def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")

def test_load_config_missing_key(tmp_path):
    invalid_config = {
        "llm": {},
        "data": {},
        "evaluation": {},
        "dashboard": {}
    }
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(invalid_config, f)
        
    with pytest.raises(ValueError, match="Missing required top-level key: rag"):
        load_config(str(config_file))
