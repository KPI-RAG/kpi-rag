import yaml

def load_config(path: str = "configs/config.yaml") -> dict:
    """Load and validate a YAML configuration file.

    Reads the YAML file at the given path, parses it into a Python dict,
    and verifies that all required top-level keys are present.

    Args:
        path (str): Path to the YAML config file.
            Defaults to ``"configs/config.yaml"``.

    Returns:
        dict: The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ValueError: If any required top-level key is missing from the config.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    required_keys = ["rag", "llm", "data", "evaluation", "dashboard"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level key: {key}")
            
    return config
