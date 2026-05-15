"""
EMMDS Data Profiler
Deep statistical profiling: distributions, skewness, correlations, outliers.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProfiler:
    """
    Produces a deep statistical profile for each feature.
    Detects skewness, outliers, correlations, and zero-variance columns.
    """

    def __init__(self):
        self.profile: dict = {}

    def profile_dataframe(self, df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
        """
        Full statistical profile of the dataset.

        Args:
            df:         Full dataframe
            target_col: Optional — excluded from feature profiling if provided

        Returns:
            Nested dict with per-feature stats and global summaries
        """
        logger.info(f"Profiling dataset: {df.shape}")

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]

        numerical = list(X.select_dtypes(include=[np.number]).columns)
        categorical = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

        self.profile = {
            "numerical_features": self._profile_numerical(X[numerical]) if numerical else {},
            "categorical_features": self._profile_categorical(X[categorical]) if categorical else {},
            "correlation_matrix": self._compute_correlation(X[numerical]) if len(numerical) > 1 else {},
            "high_correlation_pairs": self._find_high_correlations(X[numerical]) if len(numerical) > 1 else [],
            "zero_variance_columns": self._zero_variance_cols(X),
            "skewed_features": self._find_skewed_features(X[numerical]) if numerical else [],
        }

        logger.info(
            f"Profiling complete | "
            f"Numerical: {len(numerical)} | Categorical: {len(categorical)} | "
            f"Skewed: {len(self.profile['skewed_features'])} | "
            f"High-corr pairs: {len(self.profile['high_correlation_pairs'])}"
        )
        return self.profile

    # ──────────────────────────────────────────────────────────────

    def _profile_numerical(self, X: pd.DataFrame) -> dict:
        """Per-column stats for numerical features."""
        result = {}
        for col in X.columns:
            s = X[col].dropna()
            skewness = float(s.skew()) if len(s) > 2 else 0.0
            try:
                _, ks_p = stats.kstest(s, "norm", args=(s.mean(), s.std()))
                normality_p = round(float(ks_p), 4)
            except Exception:
                normality_p = None

            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

            result[col] = {
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "skewness": round(skewness, 4),
                "is_skewed": abs(skewness) > 1.0,
                "outlier_count": outlier_count,
                "outlier_percent": round(outlier_count / len(s) * 100, 2) if len(s) > 0 else 0.0,
                "normality_p_value": normality_p,
                "missing_count": int(X[col].isnull().sum()),
            }
        return result

    def _profile_categorical(self, X: pd.DataFrame) -> dict:
        """Per-column stats for categorical features."""
        result = {}
        for col in X.columns:
            vc = X[col].value_counts()
            result[col] = {
                "unique_values": int(X[col].nunique()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_value_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_value_percent": round(float(vc.iloc[0]) / len(X) * 100, 2) if len(vc) > 0 else 0.0,
                "value_counts": {str(k): int(v) for k, v in vc.head(10).items()},
                "missing_count": int(X[col].isnull().sum()),
            }
        return result

    def _compute_correlation(self, X: pd.DataFrame) -> dict:
        """Pearson correlation matrix as nested dict."""
        corr = X.corr().round(4)
        return corr.to_dict()

    def _find_high_correlations(self, X: pd.DataFrame, threshold: float = 0.85) -> list:
        """Find feature pairs with |correlation| > threshold."""
        corr = X.corr().abs()
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if val > threshold:
                    pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": round(float(corr.iloc[i, j]), 4),
                    })
        return sorted(pairs, key=lambda x: x["correlation"], reverse=True)

    def _zero_variance_cols(self, X: pd.DataFrame) -> list:
        """Columns with zero variance (constant values)."""
        num = X.select_dtypes(include=[np.number])
        return [col for col in num.columns if num[col].std() == 0]

    def _find_skewed_features(self, X: pd.DataFrame, threshold: float = 1.0) -> list:
        """Return feature names with |skewness| > threshold."""
        skewed = []
        for col in X.columns:
            sk = abs(X[col].dropna().skew())
            if sk > threshold:
                skewed.append({"feature": col, "skewness": round(float(sk), 4)})
        return sorted(skewed, key=lambda x: x["skewness"], reverse=True)

    def get_profile(self) -> dict:
        if not self.profile:
            raise RuntimeError("No profile computed yet. Call .profile_dataframe() first.")
        return self.profile
