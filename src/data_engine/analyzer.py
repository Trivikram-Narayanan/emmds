"""
EMMDS Data Analyzer
Detects: task type, feature types, dataset size, class imbalance.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataAnalyzer:
    """
    Analyzes a dataset and produces a structured profile
    used by the rest of the pipeline.
    """

    def __init__(self):
        self.report: dict = {}

    def analyze(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Main entry point. Returns full analysis report.

        Args:
            df:         The full dataframe (features + target)
            target_col: Name of the target column

        Returns:
            dict with task, shape, feature types, missing info, imbalance
        """
        logger.info(f"Analyzing dataset: {df.shape}, target='{target_col}'")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.report = {
            "target_column": target_col,
            "task": self._detect_task(y),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_count": int(X.shape[1]),
            "feature_names": list(X.columns),
            "feature_types": self._classify_features(X),
            "missing": self._missing_summary(df),
            "target_info": self._target_summary(y),
            "imbalance_ratio": self._imbalance_ratio(y),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "duplicates": int(df.duplicated().sum()),
        }

        logger.info(
            f"Task detected: {self.report['task']} | "
            f"Rows: {self.report['rows']} | "
            f"Features: {self.report['feature_count']} | "
            f"Missing: {self.report['missing']['has_missing']}"
        )
        return self.report

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _detect_task(self, y: pd.Series) -> str:
        """Heuristic: few unique values → classification, else regression."""
        n_unique = y.nunique()
        dtype = y.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return "classification"
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            return "classification"
        if n_unique <= 20 and pd.api.types.is_integer_dtype(dtype):
            return "classification"
        if n_unique / len(y) < 0.05:
            return "classification"
        return "regression"

    def _classify_features(self, X: pd.DataFrame) -> dict:
        """Split features into numerical vs categorical."""
        numerical = list(X.select_dtypes(include=[np.number]).columns)
        categorical = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
        return {
            "numerical": numerical,
            "categorical": categorical,
            "numerical_count": len(numerical),
            "categorical_count": len(categorical),
        }

    def _missing_summary(self, df: pd.DataFrame) -> dict:
        """Per-column missing value stats."""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        cols_with_missing = missing_counts[missing_counts > 0]

        return {
            "has_missing": bool(cols_with_missing.any()),
            "total_missing_cells": int(missing_counts.sum()),
            "missing_percent_overall": round(missing_counts.sum() / df.size * 100, 2),
            "columns_with_missing": {
                col: {"count": int(cnt), "percent": float(missing_pct[col])}
                for col, cnt in cols_with_missing.items()
            },
        }

    def _target_summary(self, y: pd.Series) -> dict:
        """Summary stats for the target variable."""
        task = self._detect_task(y)
        base = {
            "dtype": str(y.dtype),
            "unique_values": int(y.nunique()),
            "null_count": int(y.isnull().sum()),
        }
        if task == "classification":
            vc = y.value_counts()
            base["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            base["num_classes"] = int(y.nunique())
        else:
            base["mean"] = float(y.mean())
            base["std"] = float(y.std())
            base["min"] = float(y.min())
            base["max"] = float(y.max())
        return base

    def _imbalance_ratio(self, y: pd.Series) -> Optional[float]:
        """
        For classification: ratio of majority to minority class.
        A ratio > 1.5 is considered imbalanced.
        Returns None for regression tasks.
        """
        if self._detect_task(y) != "classification":
            return None
        vc = y.value_counts()
        if len(vc) < 2:
            return None
        return round(float(vc.iloc[0]) / float(vc.iloc[-1]), 3)

    def get_report(self) -> dict:
        """Return the last computed report."""
        if not self.report:
            raise RuntimeError("No analysis run yet. Call .analyze() first.")
        return self.report
