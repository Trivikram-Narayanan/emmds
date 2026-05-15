"""
EMMDS Config — Loads and provides access to settings.yaml.
"""

import yaml
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_config: dict = {}


def load_config(path: str | Path = _CONFIG_PATH) -> dict:
    """Load YAML config from disk."""
    global _config
    with open(path, "r") as f:
        _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    """Return loaded config, loading if needed."""
    if not _config:
        load_config()
    return _config


def get(key_path: str, default: Any = None) -> Any:
    """
    Dot-notation access to config values.
    Example: get("training.cv_folds") → 5
    """
    cfg = get_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val
