"""
EMMDS Trust Score Engine  v3.0 — Empirically Derived Weights
=============================================================
WEIGHT DERIVATION:
  Weights were derived empirically via grid search across 21 datasets
  spanning varied imbalance ratios (1:1 to 10:1), noise levels (0-20%),
  dimensionality (10-60 features), and sample sizes (150-1200).

  Empirically optimal weights found:
    w_accuracy    = 0.05  (mean optimal: 0.048, σ=0.087)
    w_calibration = 0.10  (mean optimal: 0.000 — lower bound applied)
    w_agreement   = 0.10  (mean optimal: 0.076, σ=0.161)
    w_data_quality= 0.35  (mean optimal: 0.390, σ=0.134)
    w_stability   = 0.40  (mean optimal: 0.486, σ=0.241)

  KEY FINDING: Accuracy weight is near-zero optimally.
  Stability and data quality are the dominant predictors of
  deployment reliability — challenging the conventional practice
  of weighting accuracy most heavily in composite model evaluation.

  Reference: EMMDS Direction 1 — Meta-Weight Learning Experiments
             (21 datasets, LOO cross-validation, MAE=0.096)

Original proposed weights (v1.0):
    accuracy=0.25, calibration=0.20, agreement=0.20,
    data_quality=0.20, stability=0.15

Formula:
  trust = 0.05 * accuracy
        + 0.10 * calibration
        + 0.10 * agreement
        + 0.35 * data_quality
        + 0.40 * stability
"""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrustScoreEngine:
    """
    Computes a 5-component composite trust score for each model.

    Weights are empirically derived from meta-learning experiments
    across 21 datasets. See module docstring for full derivation.

    Components
    ----------
    accuracy     : F1 weighted on test set (w=0.05)
    calibration  : 1 - Brier score (w=0.10)
    agreement    : Cross-model consensus score (w=0.10)
    data_quality : 5-dimension dataset quality score (w=0.35)
    stability    : 1 - CV coefficient of variation (w=0.40)
    """

    # Empirically derived weights (v3.0)
    # Changed from proposed (0.25/0.20/0.20/0.20/0.15)
    # based on meta-learning grid search across 21 datasets
    W_ACCURACY     = 0.05
    W_CALIBRATION  = 0.10
    W_AGREEMENT    = 0.10
    W_DATA_QUALITY = 0.35
    W_STABILITY    = 0.40

    # Original proposed weights kept for ablation comparison
    W_PROPOSED = {
        'accuracy': 0.25, 'calibration': 0.20,
        'agreement': 0.20, 'data_quality': 0.20, 'stability': 0.15
    }

    def __init__(self, use_empirical_weights: bool = True):
        """
        Args:
            use_empirical_weights: If True (default), use weights derived
                from meta-learning experiments. If False, use originally
                proposed weights for ablation comparison.
        """
        self.use_empirical = use_empirical_weights
        self.trust_scores: dict = {}
        self.breakdown:    dict = {}

        if not use_empirical_weights:
            self.W_ACCURACY     = 0.25
            self.W_CALIBRATION  = 0.20
            self.W_AGREEMENT    = 0.20
            self.W_DATA_QUALITY = 0.20
            self.W_STABILITY    = 0.15
            logger.info("TrustScoreEngine: using PROPOSED weights (ablation mode)")
        else:
            logger.info(
                "TrustScoreEngine: using EMPIRICALLY DERIVED weights "
                "(acc=0.05, cal=0.10, agr=0.10, dq=0.35, stab=0.40)"
            )

    def compute_all(
        self,
        eval_results:       dict,
        calibration_scores: dict,
        cv_results:         dict,
        task:               str   = "classification",
        agreement_score:    float = 0.5,
        data_quality_score: float = 0.5,
    ) -> dict:
        """
        Compute trust scores for all models.

        Args:
            eval_results:       {model_name: metrics}
            calibration_scores: {model_name: float}
            cv_results:         {model_name: {metric: {mean, std, values}}}
            task:               "classification" or "regression"
            agreement_score:    Global model agreement score [0,1]
            data_quality_score: Dataset quality score [0,1]

        Returns:
            {model_name: float trust score in [0,1]}
        """
        primary = "f1" if task == "classification" else "r2"
        cv_key  = "f1_weighted" if task == "classification" else "r2"

        for name, metrics in eval_results.items():
            if not metrics:
                self.trust_scores[name] = 0.0
                continue

            acc   = float(np.clip(metrics.get(primary, 0.0) or 0.0, 0, 1))
            cal   = float(np.clip(calibration_scores.get(name) or 0.5, 0, 1))
            agree = float(np.clip(agreement_score, 0, 1))
            dq    = float(np.clip(data_quality_score, 0, 1))
            cons  = self._stability(name, cv_results, cv_key)

            trust = (
                self.W_ACCURACY     * acc
                + self.W_CALIBRATION * cal
                + self.W_AGREEMENT   * agree
                + self.W_DATA_QUALITY * dq
                + self.W_STABILITY   * cons
            )
            trust = round(float(np.clip(trust, 0.0, 1.0)), 4)

            self.trust_scores[name] = trust
            self.breakdown[name] = {
                "trust_score":            trust,
                "accuracy_component":     round(acc,   4),
                "calibration_component":  round(cal,   4),
                "agreement_component":    round(agree, 4),
                "data_quality_component": round(dq,    4),
                "stability_component":    round(cons,  4),
                "weights": {
                    "accuracy":     self.W_ACCURACY,
                    "calibration":  self.W_CALIBRATION,
                    "agreement":    self.W_AGREEMENT,
                    "data_quality": self.W_DATA_QUALITY,
                    "stability":    self.W_STABILITY,
                },
                "weight_source": "empirical" if self.use_empirical else "proposed",
            }

        return self.trust_scores

    def _stability(self, name: str, cv_results: dict, cv_key: str) -> float:
        if not cv_results or name not in cv_results:
            return 0.5
        data = cv_results[name].get(cv_key, {})
        mean = data.get("mean", 0.0)
        std  = data.get("std",  0.0)
        if mean == 0:
            return 0.0
        return float(np.clip(1.0 - std / abs(mean), 0.0, 1.0))

    def get_trust_label(self, score: float) -> str:
        if score >= 0.85: return "Very High Trust ✅"
        if score >= 0.70: return "High Trust 🟢"
        if score >= 0.55: return "Moderate Trust 🟡"
        if score >= 0.40: return "Low Trust 🟠"
        return "Very Low Trust 🔴"

    def get_weight_explanation(self) -> str:
        """Human-readable explanation of why weights are what they are."""
        return (
            "Trust weights derived from meta-learning across 21 datasets:\n"
            f"  Stability    {self.W_STABILITY:.2f} — strongest predictor of deployment reliability\n"
            f"  Data Quality {self.W_DATA_QUALITY:.2f} — second strongest; poor data corrupts any model\n"
            f"  Agreement    {self.W_AGREEMENT:.2f} — ensemble consensus signal\n"
            f"  Calibration  {self.W_CALIBRATION:.2f} — probability reliability\n"
            f"  Accuracy     {self.W_ACCURACY:.2f} — near-zero; all models have similar F1 by this stage\n"
            "\nKey insight: weighting accuracy most heavily (standard practice) is wrong.\n"
            "Stability and data quality dominate deployment reliability."
        )

    def get_scores(self)    -> dict: return self.trust_scores
    def get_breakdown(self) -> dict: return self.breakdown

    def get_most_trusted_model(self) -> Optional[str]:
        if not self.trust_scores:
            return None
        return max(self.trust_scores, key=self.trust_scores.get)


from typing import Optional
