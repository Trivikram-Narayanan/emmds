"""
EMMDS Data Quality Scorer
Produces a quantitative quality score [0, 1] for a dataset.
Used as one component of the Trust Score.

Quality dimensions:
  completeness   — how much data is present (vs missing)
  uniqueness     — absence of duplicates
  consistency    — absence of extreme outliers / inf values
  balance        — class balance (classification only)
  noise          — feature signal-to-noise estimate
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataQualityScorer:
    """
    Computes a composite data quality score in [0, 1].
    Higher = cleaner data. Used by the Trust Score Engine.
    """

    # Component weights (must sum to 1.0)
    W_COMPLETENESS  = 0.30
    W_UNIQUENESS    = 0.20
    W_CONSISTENCY   = 0.20
    W_BALANCE       = 0.15
    W_NOISE         = 0.15

    def __init__(self):
        self.score: Optional[float] = None
        self.breakdown: dict = {}

    def score_dataset(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: str = "classification",
    ) -> float:
        """
        Compute composite data quality score.

        Returns:
            float in [0, 1] — higher is better
        """
        logger.info(f"Scoring data quality: {df.shape}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        completeness  = self._completeness(X)
        uniqueness    = self._uniqueness(df)
        consistency   = self._consistency(X)
        balance       = self._balance(y) if task == "classification" else 1.0
        noise         = self._noise(X)

        composite = (
            self.W_COMPLETENESS  * completeness
            + self.W_UNIQUENESS  * uniqueness
            + self.W_CONSISTENCY * consistency
            + self.W_BALANCE     * balance
            + self.W_NOISE       * noise
        )
        self.score = round(float(np.clip(composite, 0.0, 1.0)), 4)

        self.breakdown = {
            "quality_score":  self.score,
            "completeness":   round(completeness,  4),
            "uniqueness":     round(uniqueness,    4),
            "consistency":    round(consistency,   4),
            "balance":        round(balance,       4),
            "noise_score":    round(noise,         4),
            "label":          self._label(self.score),
        }

        logger.info(
            f"Data quality score: {self.score} ({self.breakdown['label']}) | "
            f"comp={completeness:.3f} uniq={uniqueness:.3f} "
            f"cons={consistency:.3f} bal={balance:.3f} noise={noise:.3f}"
        )
        return self.score

    # ─────────────────────────────────────────────────────────────

    def _completeness(self, X: pd.DataFrame) -> float:
        """1 - overall missing rate."""
        return float(1.0 - X.isnull().mean().mean())

    def _uniqueness(self, df: pd.DataFrame) -> float:
        """1 - duplicate row rate."""
        return float(1.0 - df.duplicated().mean())

    def _consistency(self, X: pd.DataFrame) -> float:
        """
        Penalise infinite values and extreme outliers.
        Score = fraction of (numeric) cells that are finite and not extreme outliers.
        """
        num = X.select_dtypes(include=[np.number])
        if num.empty:
            return 1.0

        total = num.size
        inf_count = np.isinf(num.values).sum()

        # IQR-based outlier count
        outlier_count = 0
        for col in num.columns:
            s = num[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_count += int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())

        bad = inf_count + outlier_count
        return float(max(0.0, 1.0 - bad / max(total, 1)))

    def _balance(self, y: pd.Series) -> float:
        """
        Class balance score. Perfect balance = 1.0.
        Uses normalised entropy: H(y) / log(n_classes).
        """
        counts = y.value_counts().values
        if len(counts) <= 1:
            return 0.0
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(len(counts))
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0

    def _noise(self, X: pd.DataFrame) -> float:
        """
        Noise score based on coefficient of variation.
        Low CoV → consistent features → higher score.
        Score = 1 / (1 + mean_CoV), bounded to [0, 1].
        """
        num = X.select_dtypes(include=[np.number])
        if num.empty:
            return 1.0
        cv_vals = []
        for col in num.columns:
            s = num[col].dropna()
            if s.mean() != 0 and len(s) > 1:
                cv_vals.append(abs(s.std() / s.mean()))
        if not cv_vals:
            return 1.0
        mean_cv = np.mean(cv_vals)
        return float(1.0 / (1.0 + mean_cv))

    def _label(self, score: float) -> str:
        if score >= 0.85: return "Excellent 🟢"
        if score >= 0.70: return "Good 🟡"
        if score >= 0.50: return "Fair 🟠"
        return "Poor 🔴"

    def get_breakdown(self) -> dict:
        return self.breakdown
