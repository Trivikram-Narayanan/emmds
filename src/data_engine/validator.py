"""
EMMDS Data Validator
Checks: duplicates, invalid values, inconsistent formats, target integrity.
"""

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates data quality before training.
    Issues are classified as errors (blockers) or warnings (non-blockers).
    """

    def __init__(self):
        self.errors: list = []
        self.warnings: list = []
        self.passed: bool = False

    def validate(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Run all validation checks.

        Returns:
            {
                "passed": bool,
                "errors": [...],
                "warnings": [...],
                "summary": str
            }
        """
        logger.info(f"Validating dataset: {df.shape}")
        self.errors = []
        self.warnings = []

        self._check_empty(df)
        self._check_target_exists(df, target_col)
        self._check_target_null(df, target_col)
        self._check_duplicates(df)
        self._check_constant_columns(df, target_col)
        self._check_high_missing(df, target_col)
        self._check_single_class(df, target_col)
        self._check_inf_values(df, target_col)

        self.passed = len(self.errors) == 0

        result = {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self._summary(),
        }

        if self.passed:
            logger.info(f"Validation PASSED | Warnings: {len(self.warnings)}")
        else:
            logger.error(f"Validation FAILED | Errors: {len(self.errors)}")

        return result

    # ──────────────────────────────────────────────────────────────

    def _check_empty(self, df: pd.DataFrame):
        if df.empty:
            self.errors.append("Dataset is empty (0 rows or 0 columns).")
        elif len(df) < 10:
            self.warnings.append(f"Dataset is very small ({len(df)} rows). Results may be unreliable.")

    def _check_target_exists(self, df: pd.DataFrame, target_col: str):
        if target_col not in df.columns:
            self.errors.append(f"Target column '{target_col}' not found in dataset.")

    def _check_target_null(self, df: pd.DataFrame, target_col: str):
        if target_col not in df.columns:
            return
        null_count = df[target_col].isnull().sum()
        if null_count > 0:
            pct = round(null_count / len(df) * 100, 2)
            self.errors.append(
                f"Target column '{target_col}' has {null_count} null values ({pct}%). Must be 0."
            )

    def _check_duplicates(self, df: pd.DataFrame):
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = round(dup_count / len(df) * 100, 2)
            self.warnings.append(f"Dataset contains {dup_count} duplicate rows ({pct}%).")

    def _check_constant_columns(self, df: pd.DataFrame, target_col: str):
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].nunique() <= 1:
                self.warnings.append(f"Column '{col}' is constant (only 1 unique value) — consider removing.")

    def _check_high_missing(self, df: pd.DataFrame, target_col: str, threshold: float = 0.5):
        for col in df.columns:
            if col == target_col:
                continue
            pct = df[col].isnull().mean()
            if pct > threshold:
                self.warnings.append(
                    f"Column '{col}' has {round(pct*100, 1)}% missing values (>{threshold*100}%)."
                )

    def _check_single_class(self, df: pd.DataFrame, target_col: str):
        if target_col not in df.columns:
            return
        y = df[target_col]
        if y.nunique() == 1:
            self.errors.append(
                f"Target column has only 1 unique value ({y.iloc[0]}). Classification requires ≥2 classes."
            )

    def _check_inf_values(self, df: pd.DataFrame, target_col: str):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col == target_col:
                continue
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                self.warnings.append(f"Column '{col}' contains {inf_count} infinite values.")

    def _summary(self) -> str:
        if self.passed and len(self.warnings) == 0:
            return "✅ Dataset passed all validation checks."
        elif self.passed:
            return f"⚠️  Dataset passed with {len(self.warnings)} warning(s). Review before proceeding."
        else:
            return f"❌ Dataset failed validation with {len(self.errors)} error(s) and {len(self.warnings)} warning(s)."
