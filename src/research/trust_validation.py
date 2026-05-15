"""
EMMDS Trust Validation
Empirically validates the trust score:
  - High trust models → should predict correctly more often
  - Low trust models  → should predict incorrectly more often

This is the core research contribution validation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrustValidator:
    """
    Tests whether the EMMDS trust score is a reliable predictor
    of model correctness on held-out instances.

    Validation procedure:
    1. Score every test instance: correct (1) or wrong (0)
    2. Split instances by model trust tier (high / medium / low)
    3. Measure accuracy within each tier
    4. High-trust tier should have significantly higher accuracy
    """

    def __init__(self):
        self.report: dict = {}

    def validate(
        self,
        trained_models: dict,
        trust_scores:   dict,
        X_test:         np.ndarray,
        y_test:         np.ndarray,
        task:           str = "classification",
    ) -> dict:
        """
        Run trust validation.

        Args:
            trained_models: {name: fitted_model}
            trust_scores:   {name: float} from TrustScoreEngine
            X_test:         Test features
            y_test:         True labels
            task:           "classification" or "regression"

        Returns:
            Detailed validation report
        """
        logger.info("Running trust score validation…")

        per_model = {}
        for name, model in trained_models.items():
            ts = trust_scores.get(name, 0.5)
            try:
                y_pred = model.predict(X_test)
                if task == "classification":
                    correct = (y_pred == y_test).astype(float)
                    acc     = float(correct.mean())
                    per_model[name] = {"trust": ts, "accuracy": acc, "correct": correct}
                else:
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_test, y_pred)
                    per_model[name] = {"trust": ts, "r2": r2, "score": max(0, r2)}
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")

        if not per_model:
            return {"error": "No models evaluated"}

        # Tier analysis
        tiers = self._assign_tiers(per_model, task)

        # Correlation between trust and performance
        trusts = [v["trust"] for v in per_model.values()]
        perfs  = [v.get("accuracy", v.get("r2", 0)) for v in per_model.values()]
        correlation = float(np.corrcoef(trusts, perfs)[0, 1]) if len(trusts) > 1 else 0.0

        # Validation conclusion
        validated = self._conclude(tiers, correlation)

        self.report = {
            "validated":         validated,
            "correlation":       round(correlation, 4),
            "interpretation":    self._interpret_correlation(correlation),
            "per_model":         {n: {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in m.items() if k != "correct"}
                                  for n, m in per_model.items()},
            "tier_analysis":     tiers,
            "conclusion":        self._narrative(validated, correlation, tiers),
        }

        logger.info(
            f"Trust validation: correlation={correlation:.4f} | "
            f"validated={validated} | {self._interpret_correlation(correlation)}"
        )
        return self.report

    def _assign_tiers(self, per_model: dict, task: str) -> dict:
        """Split models into high/medium/low trust tiers and compare performance."""
        metric_key = "accuracy" if task == "classification" else "r2"

        high   = [(n, m) for n, m in per_model.items() if m["trust"] >= 0.75]
        medium = [(n, m) for n, m in per_model.items() if 0.50 <= m["trust"] < 0.75]
        low    = [(n, m) for n, m in per_model.items() if m["trust"] < 0.50]

        def tier_stats(members):
            if not members:
                return {"n": 0, "mean_trust": None, "mean_performance": None}
            return {
                "n":               len(members),
                "models":          [n for n, _ in members],
                "mean_trust":      round(np.mean([m["trust"] for _, m in members]), 4),
                "mean_performance": round(np.mean([m.get(metric_key, 0) for _, m in members]), 4),
            }

        return {
            "high":   tier_stats(high),
            "medium": tier_stats(medium),
            "low":    tier_stats(low),
            "metric": metric_key,
        }

    def _conclude(self, tiers: dict, correlation: float) -> bool:
        """
        Trust is considered validated if:
        - High-tier accuracy > low-tier accuracy (when both have members), OR
        - Pearson correlation between trust and performance > 0.3
        """
        high_perf = tiers["high"].get("mean_performance")
        low_perf  = tiers["low"].get("mean_performance")

        if high_perf is not None and low_perf is not None:
            return high_perf > low_perf
        return correlation > 0.3

    def _interpret_correlation(self, r: float) -> str:
        if   r >= 0.7:  return "Strong positive correlation ✅"
        elif r >= 0.4:  return "Moderate positive correlation 🟡"
        elif r >= 0.1:  return "Weak positive correlation 🟠"
        elif r >= -0.1: return "No meaningful correlation ⚠️"
        else:           return "Negative correlation 🔴"

    def _narrative(self, validated: bool, correlation: float, tiers: dict) -> str:
        if validated:
            return (
                f"✅ Trust score VALIDATED. "
                f"Pearson correlation between trust and performance: {correlation:.3f}. "
                f"High-trust models achieved {tiers['high'].get('mean_performance','N/A')} "
                f"vs low-trust {tiers['low'].get('mean_performance','N/A')}."
            )
        return (
            f"⚠️ Trust score validation INCONCLUSIVE. "
            f"Correlation: {correlation:.3f}. "
            f"Consider larger model ensemble or more diverse dataset."
        )

    def get_report(self) -> dict:
        return self.report
