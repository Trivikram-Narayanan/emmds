"""
EMMDS Trainer
Trains all registered models on the provided data.
Returns a dict of {model_name: fitted_model}.
"""

import time
import numpy as np
from typing import Optional
from src.models.model_registry import get_all_models
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains all models in the registry on the given training data.
    Tracks training time per model and handles failures gracefully.
    """

    def __init__(self):
        self.trained_models: dict = {}
        self.training_times: dict = {}
        self.failed_models: list = []

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        enabled_models: Optional[list] = None,
    ) -> dict:
        """
        Train all enabled models.

        Args:
            X_train:        Training features (numpy array)
            y_train:        Training labels
            enabled_models: Optional override list of model names

        Returns:
            {model_name: fitted_model}
        """
        models = get_all_models(enabled_only=True)
        if enabled_models:
            models = {k: v for k, v in models.items() if k in enabled_models}

        logger.info(f"Training {len(models)} model(s) on {X_train.shape[0]} samples...")

        for name, model in models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                elapsed = round(time.time() - start, 3)

                self.trained_models[name] = model
                self.training_times[name] = elapsed
                logger.info(f"  ✅ {name:25s} trained in {elapsed:.3f}s")

            except Exception as e:
                self.failed_models.append(name)
                logger.error(f"  ❌ {name:25s} failed: {e}")

        logger.info(
            f"Training complete | "
            f"Successful: {len(self.trained_models)} | "
            f"Failed: {len(self.failed_models)}"
        )
        return self.trained_models

    def get_training_summary(self) -> dict:
        return {
            "trained": list(self.trained_models.keys()),
            "failed": self.failed_models,
            "training_times": self.training_times,
            "total_time": round(sum(self.training_times.values()), 3),
        }
