"""
EMMDS Helpers — Shared utility functions.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: str | Path) -> None:
    """Persist a trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str | Path) -> Any:
    """Load a persisted model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model


def save_json(data: dict, path: str | Path) -> None:
    """Save a dict as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)
    logger.info(f"JSON saved → {path}")


def load_json(path: str | Path) -> dict:
    """Load JSON from disk."""
    with open(path, "r") as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Handle non-serializable types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def timestamp() -> str:
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns fallback on zero denominator."""
    return numerator / denominator if denominator != 0 else fallback


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict using dot notation."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
