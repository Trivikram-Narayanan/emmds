"""
EMMDS Model Saver
Save and load trained models with accompanying metadata.
Uses joblib for binary serialisation and JSON for metadata.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("outputs/models")


def _serialise(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Non-serialisable: {type(obj)}")


class ModelSaver:
    """
    Saves trained models + metadata bundles.
    Each save creates two files:
      - <name>.joblib  — the serialised model
      - <name>.json    — metadata (metrics, feature names, params, ...)
    """

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Any,
        name: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Save a model and its metadata.

        Args:
            model:    Fitted sklearn-compatible model
            name:     File stem (e.g. "random_forest")
            metadata: Arbitrary metadata dict to accompany the model

        Returns:
            {"model_path": ..., "meta_path": ...}
        """
        ts = datetime.now().isoformat()
        model_path = self.model_dir / f"{name}.joblib"
        meta_path  = self.model_dir / f"{name}.json"

        joblib.dump(model, model_path)

        meta = {
            "name":      name,
            "saved_at":  ts,
            "model_type": type(model).__name__,
            **(metadata or {}),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=_serialise)

        logger.info(f"Model saved → {model_path}")
        return {"model_path": str(model_path), "meta_path": str(meta_path)}

    def save_all(
        self,
        models: dict,
        eval_results: dict = None,
        trust_scores: dict = None,
        feature_names: list = None,
    ) -> dict:
        """
        Save every model in a dict with associated metadata.

        Returns:
            {model_name: save_paths_dict}
        """
        paths = {}
        for name, model in models.items():
            metadata = {
                "feature_names": feature_names or [],
                "metrics": (eval_results or {}).get(name, {}),
                "trust_score": (trust_scores or {}).get(name),
            }
            # Strip non-serialisable items (e.g. confusion matrix lists are fine)
            metadata["metrics"] = {
                k: v for k, v in metadata["metrics"].items()
                if not isinstance(v, list)
            }
            paths[name] = self.save(model, name, metadata)
        logger.info(f"Saved {len(paths)} models to {self.model_dir}/")
        return paths

    def load(self, name: str) -> tuple:
        """
        Load a model and its metadata by name.

        Returns:
            (model, metadata_dict)
        """
        model_path = self.model_dir / f"{name}.joblib"
        meta_path  = self.model_dir / f"{name}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        logger.info(f"Model loaded ← {model_path}")
        return model, meta

    def load_all(self) -> dict:
        """
        Load all models from the model directory.

        Returns:
            {stem_name: {"model": ..., "metadata": ...}}
        """
        result = {}
        for p in self.model_dir.glob("*.joblib"):
            model, meta = self.load(p.stem)
            result[p.stem] = {"model": model, "metadata": meta}
        logger.info(f"Loaded {len(result)} model(s) from {self.model_dir}/")
        return result

    def list_saved_models(self) -> list:
        """Return names of all saved models."""
        return [p.stem for p in self.model_dir.glob("*.joblib")]
