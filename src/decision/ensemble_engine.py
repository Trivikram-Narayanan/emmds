"""
EMMDS Ensemble Engine
Combines the top-K trusted models into a voting / averaging ensemble.
Outperforms any single model when models are diverse and well-calibrated.
"""

import numpy as np
from scipy.stats import mode as scipy_mode
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleEngine:
    """
    Builds a lightweight ensemble from the top-K models ranked by trust score.

    Supports:
      - majority_vote   (classification)
      - soft_vote       (classification, requires predict_proba)
      - mean_average    (regression)
    """

    def __init__(self, top_k: int = 3, method: str = "soft_vote"):
        self.top_k   = top_k
        self.method  = method
        self.members: dict = {}   # {name: model}
        self.weights: dict = {}   # {name: trust_score}

    def build(
        self,
        trained_models: dict,
        trust_scores:   dict,
        task:           str = "classification",
    ) -> "EnsembleEngine":
        """
        Select the top-K models by trust score and build the ensemble.

        Args:
            trained_models: {name: fitted_model}
            trust_scores:   {name: float}
            task:           "classification" or "regression"

        Returns:
            self (for chaining)
        """
        if task == "regression":
            self.method = "mean_average"

        # Sort by trust, pick top-K
        ranked = sorted(trust_scores.items(), key=lambda x: x[1], reverse=True)
        top    = [(n, s) for n, s in ranked if n in trained_models][:self.top_k]

        self.members = {n: trained_models[n] for n, _ in top}
        self.weights = {n: s                  for n, s in top}

        # If soft_vote requested but some models lack predict_proba, fall back
        if self.method == "soft_vote":
            can_proba = all(hasattr(m, "predict_proba") for m in self.members.values())
            if not can_proba:
                logger.warning("Some ensemble members lack predict_proba — using majority_vote")
                self.method = "majority_vote"

        logger.info(
            f"Ensemble built ({self.method}) | "
            f"Members: {list(self.members.keys())} | "
            f"Weights: {self.weights}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction.

        Args:
            X: Feature matrix (samples × features)

        Returns:
            Predicted labels or values
        """
        if not self.members:
            raise RuntimeError("Ensemble not built. Call .build() first.")

        if self.method == "soft_vote":
            return self._soft_vote(X)
        elif self.method == "majority_vote":
            return self._majority_vote(X)
        elif self.method == "mean_average":
            return self._mean_average(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of class probabilities (soft vote only)."""
        if self.method != "soft_vote":
            raise NotImplementedError("predict_proba only supported for soft_vote")

        total_weight = sum(self.weights.values())
        weighted_sum = None

        for name, model in self.members.items():
            w     = self.weights.get(name, 1.0) / total_weight
            proba = model.predict_proba(X)
            if weighted_sum is None:
                weighted_sum = w * proba
            else:
                weighted_sum += w * proba

        return weighted_sum

    # ── Internal methods ──────────────────────────────────────────

    def _soft_vote(self, X: np.ndarray) -> np.ndarray:
        """Weighted average probabilities → argmax."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _majority_vote(self, X: np.ndarray) -> np.ndarray:
        """Simple majority vote across all members."""
        preds = np.stack([m.predict(X) for m in self.members.values()], axis=0)
        majority, _ = scipy_mode(preds, axis=0, keepdims=False)
        return majority.flatten()

    def _mean_average(self, X: np.ndarray) -> np.ndarray:
        """Mean of regression predictions."""
        preds = np.stack([m.predict(X) for m in self.members.values()], axis=0)
        return preds.mean(axis=0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task:   str = "classification",
    ) -> dict:
        """
        Evaluate the ensemble on a test set.
        Returns the same metrics dict as the single-model evaluator.
        """
        from src.evaluation.metrics import classification_metrics, regression_metrics

        y_pred = self.predict(X_test)

        if task == "classification":
            y_prob = None
            if self.method == "soft_vote":
                try:
                    y_prob = self.predict_proba(X_test)
                except Exception:
                    pass
            return classification_metrics(y_test, y_pred, y_prob)
        else:
            return regression_metrics(y_test, y_pred)

    def get_info(self) -> dict:
        return {
            "method":   self.method,
            "top_k":    self.top_k,
            "members":  list(self.members.keys()),
            "weights":  self.weights,
        }
