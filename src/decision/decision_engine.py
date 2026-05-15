"""
EMMDS Decision Engine
The final intelligence layer.
Combines ranking, trust, and explainability into a unified output.
"""

import numpy as np
from src.decision.trust_score import TrustScoreEngine
from src.decision.model_selector import ModelSelector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    """
    Orchestrates the final decision:
    1. Compute trust scores for all models
    2. Select best model via combined ranking + trust
    3. Produce final structured output
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        self.trust_engine = TrustScoreEngine()
        self.selector = ModelSelector()
        self.final_output: dict = {}

    def decide(
        self,
        trained_models: dict,
        eval_results: dict,
        calibration_scores: dict,
        cv_results: dict,
        leaderboard: list,
        shap_global: dict,
        analysis_report: dict,
    ) -> dict:
        """
        Run full decision logic.

        Returns:
            The complete EMMDS output: best model, trust, top features, summary.
        """
        logger.info("Running Decision Engine...")

        # 1. Trust scores
        trust_scores = self.trust_engine.compute_all(
            eval_results=eval_results,
            calibration_scores=calibration_scores,
            cv_results=cv_results,
            task=self.task,
        )

        # 2. Model selection
        selection = self.selector.select(
            leaderboard=leaderboard,
            trust_scores=trust_scores,
            trained_models=trained_models,
        )

        best_name = selection.get("best_model_name", "N/A")
        best_metrics = eval_results.get(best_name, {})
        trust_breakdown = self.trust_engine.get_breakdown().get(best_name, {})

        # 3. Top features from SHAP
        top_features = []
        if shap_global and "ranking" in shap_global:
            top_features = [
                f"{i+1}. {r['feature']} (importance: {r['importance']:.4f})"
                for i, r in enumerate(shap_global["ranking"][:5])
            ]

        # 4. Build final output
        primary_metric = "f1" if self.task == "classification" else "r2"
        primary_value = best_metrics.get(primary_metric, "N/A")
        accuracy = best_metrics.get("accuracy", "N/A")

        trust_score = trust_scores.get(best_name, 0.0)
        trust_label = self.trust_engine.get_trust_label(trust_score)

        self.final_output = {
            "best_model": best_name,
            "task": self.task,
            "primary_metric": primary_metric,
            "primary_score": primary_value,
            "accuracy": accuracy,
            "trust_score": trust_score,
            "trust_label": trust_label,
            "trust_breakdown": trust_breakdown,
            "top_features": top_features,
            "all_trust_scores": trust_scores,
            "leaderboard": leaderboard,
            "selection_breakdown": selection.get("selection_breakdown", []),
            "dataset_info": {
                "rows": analysis_report.get("rows"),
                "features": analysis_report.get("feature_count"),
                "task": analysis_report.get("task"),
                "imbalance_ratio": analysis_report.get("imbalance_ratio"),
            },
        }

        logger.info(
            f"\n{'='*50}\n"
            f"  EMMDS DECISION\n"
            f"  Best Model:    {best_name}\n"
            f"  {primary_metric.upper()}:         {primary_value}\n"
            f"  Accuracy:      {accuracy}\n"
            f"  Trust Score:   {trust_score} ({trust_label})\n"
            f"{'='*50}"
        )
        return self.final_output

    def get_output(self) -> dict:
        return self.final_output
