"""
EMMDS Evaluator
Applies all metrics to every trained model on test data.
"""

import numpy as np
from typing import Optional
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates all trained models on held-out test data.
    Stores per-model results and surfaces them for ranking.
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        self.results: dict = {}

    def evaluate_all(
        self,
        trained_models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Evaluate every model in trained_models against (X_test, y_test).

        Returns:
            {model_name: {metrics dict}}
        """
        logger.info(f"Evaluating {len(trained_models)} model(s)...")

        for name, model in trained_models.items():
            try:
                y_pred = model.predict(X_test)
                y_prob = None

                if self.task == "classification" and hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)
                    except Exception:
                        pass

                if self.task == "classification":
                    m = classification_metrics(y_test, y_pred, y_prob)
                else:
                    m = regression_metrics(y_test, y_pred)

                self.results[name] = m
                primary = m.get("f1", m.get("r2", "N/A"))
                logger.info(f"  ✅ {name:25s} | primary metric: {primary}")

            except Exception as e:
                logger.error(f"  ❌ {name:25s} evaluation failed: {e}")
                self.results[name] = {}

        return self.results

    def get_results(self) -> dict:
        return self.results

    def get_predictions(self, model, X_test: np.ndarray) -> dict:
        """Get predictions + probabilities from a single model."""
        y_pred = model.predict(X_test)
        result = {"predictions": y_pred.tolist()}
        if hasattr(model, "predict_proba"):
            try:
                result["probabilities"] = model.predict_proba(X_test).tolist()
            except Exception:
                pass
        return result
