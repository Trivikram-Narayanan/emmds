"""
EMMDS Model Ranker
Sorts models by performance metrics.
Primary: F1 (classification) / R2 (regression)
Secondary: Accuracy / MAE
"""

from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRanker:
    """
    Ranks trained models based on evaluation results.
    Produces a sorted leaderboard.
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        self.ranked: list = []

    def rank(
        self,
        eval_results: dict,
        cv_results: Optional[dict] = None,
    ) -> list:
        """
        Rank all models.

        Args:
            eval_results: From ModelEvaluator.evaluate_all()
            cv_results:   From CrossValidator.run() (optional)

        Returns:
            List of dicts sorted best → worst:
            [{name, rank, primary_metric, secondary_metric, ...}]
        """
        if self.task == "classification":
            primary_key = "f1"
            secondary_key = "accuracy"
        else:
            primary_key = "r2"
            secondary_key = "mae"

        rows = []
        for name, metrics in eval_results.items():
            if not metrics:
                continue

            primary = metrics.get(primary_key, 0.0) or 0.0
            secondary = metrics.get(secondary_key, 0.0) or 0.0

            # CV stability bonus info
            cv_mean = None
            cv_std = None
            if cv_results and name in cv_results:
                cv_data = cv_results[name]
                cv_primary_key = "f1_weighted" if self.task == "classification" else "r2"
                if cv_primary_key in cv_data:
                    cv_mean = cv_data[cv_primary_key]["mean"]
                    cv_std = cv_data[cv_primary_key]["std"]

            rows.append({
                "model": name,
                primary_key: round(primary, 4),
                secondary_key: round(secondary, 4),
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "auc_roc": metrics.get("auc_roc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
            })

        # Sort: primary DESC, then secondary DESC
        self.ranked = sorted(
            rows,
            key=lambda x: (x[primary_key], x[secondary_key]),
            reverse=True,
        )

        # Assign rank numbers
        for i, row in enumerate(self.ranked):
            row["rank"] = i + 1

        logger.info(
            f"Ranking complete | Best: {self.ranked[0]['model']} "
            f"({primary_key}={self.ranked[0][primary_key]})"
            if self.ranked else "No models to rank."
        )
        return self.ranked

    def get_best_model_name(self) -> Optional[str]:
        """Return the name of the top-ranked model."""
        if not self.ranked:
            return None
        return self.ranked[0]["model"]

    def get_leaderboard(self) -> list:
        return self.ranked
