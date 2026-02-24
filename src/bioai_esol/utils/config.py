import yaml
from pathlib import Path

def load_config(path: str):
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def flatten_dict(d, parent_key="", sep="_"):
    flat_dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v
    return flat_dict