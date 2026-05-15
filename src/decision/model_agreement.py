"""
EMMDS Model Agreement Engine
Measures how much all trained models agree on predictions.

High agreement = more reliable output.
Low agreement  = models are uncertain; treat predictions with caution.

Agreement is used as one component of the enhanced Trust Score.
"""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelAgreementEngine:
    """
    Computes pairwise and global model agreement on test predictions.

    Metrics:
      - global_agreement:  fraction of test samples where ALL models agree
      - mean_pairwise:     average pairwise agreement across all model pairs
      - per_sample:        per-sample agreement fraction
      - entropy:           mean prediction entropy (lower = more confident)
    """

    def __init__(self):
        self.result: dict = {}

    def compute(
        self,
        trained_models: dict,
        X_test: np.ndarray,
        task: str = "classification",
    ) -> dict:
        """
        Compute agreement across all models on X_test.

        Args:
            trained_models: {name: fitted_model}
            X_test:         Test feature matrix
            task:           "classification" or "regression"

        Returns:
            {
                "global_agreement":   float,   # all models same output
                "mean_pairwise":      float,   # avg pairwise match
                "agreement_score":    float,   # 0-1 composite for Trust
                "per_model_agreement": {...},  # each vs majority vote
                "predictions":        {...},   # raw predictions per model
            }
        """
        logger.info(f"Computing model agreement on {X_test.shape[0]} samples...")

        # Collect all predictions
        preds: dict = {}
        for name, model in trained_models.items():
            try:
                preds[name] = model.predict(X_test)
            except Exception as e:
                logger.warning(f"  Model {name} predict failed: {e}")

        if len(preds) < 2:
            logger.warning("Fewer than 2 models — agreement is undefined, returning 1.0")
            return {"global_agreement": 1.0, "mean_pairwise": 1.0, "agreement_score": 1.0}

        model_names = list(preds.keys())
        pred_matrix  = np.stack([preds[n] for n in model_names], axis=0)  # (n_models, n_samples)
        n_models, n_samples = pred_matrix.shape

        if task == "classification":
            global_agreement   = self._global_agreement(pred_matrix)
            mean_pairwise      = self._mean_pairwise(pred_matrix)
            per_model          = self._per_model_vs_majority(pred_matrix, model_names)
            entropy_score      = self._prediction_entropy(pred_matrix)
        else:
            # Regression: use coefficient of variation of predictions
            global_agreement   = self._regression_agreement(pred_matrix)
            mean_pairwise      = global_agreement
            per_model          = {}
            entropy_score      = global_agreement

        # Composite agreement score
        agreement_score = round(
            0.5 * global_agreement
            + 0.3 * mean_pairwise
            + 0.2 * entropy_score,
            4,
        )

        self.result = {
            "global_agreement":    round(float(global_agreement),  4),
            "mean_pairwise":       round(float(mean_pairwise),     4),
            "entropy_score":       round(float(entropy_score),     4),
            "agreement_score":     agreement_score,
            "per_model_agreement": per_model,
            "n_models":            n_models,
            "n_samples":           n_samples,
            "predictions":         {n: preds[n].tolist() for n in model_names},
        }

        logger.info(
            f"Agreement → global={global_agreement:.4f} | "
            f"pairwise={mean_pairwise:.4f} | composite={agreement_score:.4f}"
        )
        return self.result

    # ─────────────────────────────────────────────────────────────

    def _global_agreement(self, pred_matrix: np.ndarray) -> float:
        """Fraction of samples where ALL models predict the same class."""
        all_same = np.all(pred_matrix == pred_matrix[0, :], axis=0)
        return float(all_same.mean())

    def _mean_pairwise(self, pred_matrix: np.ndarray) -> float:
        """Mean pairwise model agreement (fraction of matching predictions)."""
        n_models = pred_matrix.shape[0]
        agreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                match = (pred_matrix[i] == pred_matrix[j]).mean()
                agreements.append(float(match))
        return float(np.mean(agreements)) if agreements else 1.0

    def _per_model_vs_majority(
        self, pred_matrix: np.ndarray, model_names: list
    ) -> dict:
        """
        Per-model agreement with the majority vote prediction.
        """
        # Majority vote per sample
        from scipy.stats import mode
        majority, _ = mode(pred_matrix, axis=0, keepdims=True)
        result = {}
        for i, name in enumerate(model_names):
            agree = (pred_matrix[i] == majority[0]).mean()
            result[name] = round(float(agree), 4)
        return result

    def _prediction_entropy(self, pred_matrix: np.ndarray) -> float:
        """
        Mean normalised entropy of per-sample class vote distributions.
        Entropy = 0 → full agreement; Entropy = 1 → maximum disagreement.
        Returns 1 - entropy so higher = better agreement.
        """
        n_models, n_samples = pred_matrix.shape
        classes = np.unique(pred_matrix)
        n_classes = len(classes)
        if n_classes == 1:
            return 1.0

        entropies = []
        for s in range(n_samples):
            votes = pred_matrix[:, s]
            counts = np.array([(votes == c).sum() for c in classes])
            probs = counts / counts.sum()
            entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
            max_e = np.log(n_classes)
            entropies.append(entropy / max_e if max_e > 0 else 0.0)

        return float(1.0 - np.mean(entropies))

    def _regression_agreement(self, pred_matrix: np.ndarray) -> float:
        """
        For regression: 1 - mean CoV of predictions across models.
        Low spread → high agreement.
        """
        mean_preds = pred_matrix.mean(axis=0)
        std_preds  = pred_matrix.std(axis=0)
        # Avoid division by zero
        cov = np.where(np.abs(mean_preds) > 1e-8, std_preds / np.abs(mean_preds), 0.0)
        return float(np.clip(1.0 - np.mean(cov), 0.0, 1.0))

    def get_result(self) -> dict:
        return self.result

    def get_agreement_score(self) -> float:
        return self.result.get("agreement_score", 0.5)
