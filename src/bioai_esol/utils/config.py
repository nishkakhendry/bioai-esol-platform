import yaml
from pathlib import Path

def load_config(path: str):
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)