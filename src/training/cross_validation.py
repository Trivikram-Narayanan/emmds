"""
EMMDS Cross Validator
K-Fold cross-validation for all trained models.
Produces mean/std scores for robust model comparison.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from typing import Optional
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger(__name__)


class CrossValidator:
    """
    Runs stratified K-Fold CV on all trained models.
    Returns per-model mean ± std for accuracy, f1, precision, recall.
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        self.cv_results: dict = {}

    def run(
        self,
        trained_models: dict,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: Optional[int] = None,
    ) -> dict:
        """
        Run cross-validation for all models.

        Args:
            trained_models: {name: fitted_model}
            X:              Full feature array
            y:              Full label array
            n_splits:       Number of CV folds (default from config)

        Returns:
            {model_name: {metric: {mean, std}}}
        """
        n_splits = n_splits or get("training.cv_folds", 5)
        random_state = get("training.random_state", 42)

        if self.task == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        logger.info(f"Running {n_splits}-fold CV on {len(trained_models)} model(s)...")

        for name, model in trained_models.items():
            try:
                scores = cross_validate(
                    model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
                )
                self.cv_results[name] = self._summarize(scores, scoring)
                logger.info(f"  ✅ {name:25s} CV complete")
            except Exception as e:
                logger.error(f"  ❌ {name:25s} CV failed: {e}")
                self.cv_results[name] = {}

        return self.cv_results

    def _summarize(self, scores: dict, scoring: list) -> dict:
        """Compute mean ± std for each metric."""
        summary = {}
        for metric in scoring:
            key = f"test_{metric}"
            if key in scores:
                vals = scores[key]
                # neg metrics → flip sign
                if metric.startswith("neg_"):
                    vals = -vals
                    clean_name = metric.replace("neg_", "")
                else:
                    clean_name = metric
                summary[clean_name] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "values": [round(float(v), 4) for v in vals],
                }
        return summary

    def get_stability_score(self, model_name: str) -> float:
        """
        Stability = 1 - (std / mean) of primary metric.
        Higher is more stable. Clipped to [0, 1].
        """
        result = self.cv_results.get(model_name, {})
        primary = "f1_weighted" if self.task == "classification" else "r2"
        if primary not in result:
            return 0.0
        mean = result[primary]["mean"]
        std = result[primary]["std"]
        if mean == 0:
            return 0.0
        stability = 1.0 - (std / abs(mean))
        return float(np.clip(stability, 0.0, 1.0))

    def get_cv_results(self) -> dict:
        return self.cv_results
