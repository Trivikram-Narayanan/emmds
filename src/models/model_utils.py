"""
EMMDS Model Utils
Cloning, serialization, and helper functions for models.
"""

import numpy as np
from sklearn.base import clone
from pathlib import Path
from typing import Any
from src.utils.helpers import save_model, load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clone_model(model) -> Any:
    """Return a fresh, unfitted clone of an sklearn model."""
    return clone(model)


def save_all_models(trained_models: dict, output_dir: str = "outputs/models") -> dict:
    """
    Persist all trained models to disk.

    Returns:
        {model_name: saved_path}
    """
    paths = {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        path = out / f"{name}.joblib"
        save_model(model, path)
        paths[name] = str(path)

    logger.info(f"Saved {len(paths)} models to {output_dir}/")
    return paths


def load_all_models(model_dir: str = "outputs/models") -> dict:
    """
    Load all .joblib model files from a directory.

    Returns:
        {stem_name: loaded_model}
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models = {}
    for path in model_dir.glob("*.joblib"):
        models[path.stem] = load_model(path)

    logger.info(f"Loaded {len(models)} models from {model_dir}/")
    return models


def predict_single(
    model,
    X_single: np.ndarray,
    return_proba: bool = True,
) -> dict:
    """
    Run prediction on a single sample.

    Returns:
        {"prediction": ..., "probabilities": [...] or None}
    """
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    pred = model.predict(X_single)[0]
    proba = None

    if return_proba and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_single)[0].tolist()
        except Exception:
            pass

    return {
        "prediction": pred.item() if hasattr(pred, "item") else pred,
        "probabilities": proba,
    }


def get_model_params(model) -> dict:
    """Return hyperparameters of an sklearn model."""
    if hasattr(model, "get_params"):
        return model.get_params()
    return {}
