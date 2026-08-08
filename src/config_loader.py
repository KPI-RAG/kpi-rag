import yaml

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    required_keys = ["rag", "llm", "data", "evaluation", "dashboard"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level key: {key}")
            
    return config
