"""
EMMDS Model Selector
Picks the best model using a combined ranking + trust strategy.
"""

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelSelector:
    """
    Selects the final recommended model using a combined score.

    Strategy:
      combined_score = 0.6 × rank_score + 0.4 × trust_score
    
    rank_score is normalized from the leaderboard position.
    This ensures the best-performing AND most trusted model wins.
    """

    def __init__(self):
        self.selection_result: dict = {}

    def select(
        self,
        leaderboard: list,
        trust_scores: dict,
        trained_models: dict,
    ) -> dict:
        """
        Select the best model.

        Args:
            leaderboard:    Ranked list from ModelRanker
            trust_scores:   {model_name: float} from TrustScoreEngine
            trained_models: {model_name: fitted_model}

        Returns:
            {
                "best_model_name": str,
                "best_model": fitted_model,
                "combined_score": float,
                "selection_breakdown": [...]
            }
        """
        if not leaderboard:
            logger.error("Leaderboard is empty — cannot select model.")
            return {}

        n = len(leaderboard)
        rows = []

        for entry in leaderboard:
            name = entry["model"]
            rank = entry["rank"]

            # Rank score: 1st place = 1.0, last = 1/n
            rank_score = (n - rank + 1) / n

            # Trust score
            trust = trust_scores.get(name, 0.5)

            # Combined
            combined = round(0.6 * rank_score + 0.4 * trust, 4)

            rows.append({
                "model": name,
                "rank": rank,
                "rank_score": round(rank_score, 4),
                "trust_score": round(trust, 4),
                "combined_score": combined,
            })

        # Sort by combined score
        rows.sort(key=lambda x: x["combined_score"], reverse=True)
        best = rows[0]
        best_name = best["model"]

        self.selection_result = {
            "best_model_name": best_name,
            "best_model": trained_models.get(best_name),
            "combined_score": best["combined_score"],
            "trust_score": best["trust_score"],
            "rank_score": best["rank_score"],
            "selection_breakdown": rows,
        }

        logger.info(
            f"Selected model: {best_name} | "
            f"combined={best['combined_score']:.4f} | "
            f"trust={best['trust_score']:.4f}"
        )
        return self.selection_result

    def get_result(self) -> dict:
        return self.selection_result
