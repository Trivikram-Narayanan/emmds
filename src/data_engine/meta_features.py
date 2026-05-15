"""
EMMDS Meta-Feature Extractor
Extracts dataset-level statistics used by the model recommender
to suggest which models are likely to perform best.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetaFeatureExtractor:
    """
    Computes dataset meta-features — numeric descriptors that
    characterise the problem and guide model selection.

    Meta-features extracted:
      n_samples, n_features, imbalance_ratio, missing_ratio,
      avg_correlation, feature_noise_estimate, dimensionality_ratio,
      numeric_ratio, has_high_cardinality_cats, skewness_score
    """

    def __init__(self):
        self.meta: dict = {}

    def extract(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Extract meta-features from a raw dataframe.

        Args:
            df:         Full dataframe including target
            target_col: Name of the target column

        Returns:
            dict of meta-feature name → value
        """
        logger.info(f"Extracting meta-features from dataset ({df.shape})")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        n, p = X.shape

        num_cols  = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols  = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

        self.meta = {
            # ── Size ────────────────────────────────────────────
            "n_samples":           int(n),
            "n_features":          int(p),
            "dimensionality_ratio": round(p / n, 6),   # p/n > 0.1 → curse of dimensionality risk

            # ── Feature types ────────────────────────────────────
            "n_numerical":         len(num_cols),
            "n_categorical":       len(cat_cols),
            "numeric_ratio":       round(len(num_cols) / max(p, 1), 4),

            # ── Missing values ───────────────────────────────────
            "missing_ratio":       round(float(X.isnull().mean().mean()), 4),
            "has_missing":         bool(X.isnull().any().any()),

            # ── Target / class balance ───────────────────────────
            "n_classes":           int(y.nunique()),
            "imbalance_ratio":     self._imbalance(y),

            # ── Correlation / redundancy ─────────────────────────
            "avg_abs_correlation": self._avg_correlation(X[num_cols]) if len(num_cols) > 1 else 0.0,
            "max_correlation":     self._max_correlation(X[num_cols]) if len(num_cols) > 1 else 0.0,

            # ── Categorical cardinality ──────────────────────────
            "has_high_cardinality": self._high_cardinality(X[cat_cols]) if cat_cols else False,
            "max_cat_cardinality":  self._max_cardinality(X[cat_cols]) if cat_cols else 0,

            # ── Skewness ─────────────────────────────────────────
            "mean_skewness":       self._mean_skewness(X[num_cols]) if num_cols else 0.0,
            "skewed_feature_ratio": self._skewed_ratio(X[num_cols]) if num_cols else 0.0,

            # ── Noise estimate ───────────────────────────────────
            "noise_estimate":      self._noise_estimate(X[num_cols]) if len(num_cols) > 1 else 0.0,
        }

        logger.info(
            f"Meta-features: n={n}, p={p}, imbalance={self.meta['imbalance_ratio']}, "
            f"missing={self.meta['missing_ratio']}, avg_corr={self.meta['avg_abs_correlation']}"
        )
        return self.meta

    # ─────────────────────────────────────────────────────────────

    def _imbalance(self, y: pd.Series) -> Optional[float]:
        """Majority / minority class ratio. 1.0 = perfectly balanced."""
        vc = y.value_counts()
        if len(vc) < 2:
            return None
        return round(float(vc.iloc[0]) / float(vc.iloc[-1]), 3)

    def _avg_correlation(self, X: pd.DataFrame) -> float:
        """Mean absolute pairwise Pearson correlation (upper triangle)."""
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        vals = upper.stack().values
        return round(float(np.mean(vals)), 4) if len(vals) > 0 else 0.0

    def _max_correlation(self, X: pd.DataFrame) -> float:
        """Maximum pairwise absolute correlation."""
        corr = X.corr().abs()
        arr = corr.to_numpy().copy()
        np.fill_diagonal(arr, 0)
        return round(float(arr.max()), 4)

    def _high_cardinality(self, X: pd.DataFrame, threshold: int = 20) -> bool:
        """True if any categorical column has > threshold unique values."""
        return any(X[col].nunique() > threshold for col in X.columns)

    def _max_cardinality(self, X: pd.DataFrame) -> int:
        if X.empty:
            return 0
        return int(max(X[col].nunique() for col in X.columns))

    def _mean_skewness(self, X: pd.DataFrame) -> float:
        return round(float(X.skew().abs().mean()), 4)

    def _skewed_ratio(self, X: pd.DataFrame, threshold: float = 1.0) -> float:
        """Fraction of numerical features with |skew| > threshold."""
        skews = X.skew().abs()
        return round(float((skews > threshold).mean()), 4)

    def _noise_estimate(self, X: pd.DataFrame) -> float:
        """
        Crude noise estimate: mean coefficient of variation across features.
        High CoV suggests noisy / wide-range features.
        """
        cv_vals = []
        for col in X.columns:
            s = X[col].dropna()
            if s.mean() != 0:
                cv_vals.append(abs(s.std() / s.mean()))
        return round(float(np.mean(cv_vals)), 4) if cv_vals else 0.0

    def get_meta(self) -> dict:
        if not self.meta:
            raise RuntimeError("No meta-features computed yet. Call .extract() first.")
        return self.meta
