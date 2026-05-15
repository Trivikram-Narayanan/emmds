"""
EMMDS Error Handler
Centralised error classification, logging, and recovery logic.

Error types:
  DataError      — problems with the input dataset
  TrainingError  — model failed to train
  PipelineError  — orchestration / stage failure
  ConfigError    — bad configuration values
"""

import traceback
from enum import Enum
from typing import Optional, Callable, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Error taxonomy ────────────────────────────────────────────────────

class ErrorCode(str, Enum):
    # Data errors
    EMPTY_DATASET         = "EMPTY_DATASET"
    MISSING_TARGET        = "MISSING_TARGET"
    ALL_TARGET_NULL       = "ALL_TARGET_NULL"
    SINGLE_CLASS          = "SINGLE_CLASS"
    INSUFFICIENT_SAMPLES  = "INSUFFICIENT_SAMPLES"
    BAD_FEATURE_TYPES     = "BAD_FEATURE_TYPES"

    # Training errors
    MODEL_FIT_FAILED      = "MODEL_FIT_FAILED"
    ALL_MODELS_FAILED     = "ALL_MODELS_FAILED"
    CV_FAILED             = "CV_FAILED"
    CALIBRATION_FAILED    = "CALIBRATION_FAILED"

    # Pipeline errors
    PIPELINE_STAGE_FAILED = "PIPELINE_STAGE_FAILED"
    PREPROCESSING_FAILED  = "PREPROCESSING_FAILED"
    EXPLAINABILITY_FAILED = "EXPLAINABILITY_FAILED"

    # Config errors
    INVALID_CONFIG        = "INVALID_CONFIG"
    MISSING_CONFIG_KEY    = "MISSING_CONFIG_KEY"

    UNKNOWN               = "UNKNOWN"


class EMMDSError(Exception):
    """Base exception for all EMMDS-specific errors."""
    def __init__(self, code: ErrorCode, message: str, context: dict = None):
        super().__init__(message)
        self.code    = code
        self.message = message
        self.context = context or {}

    def to_dict(self) -> dict:
        return {
            "error":   True,
            "code":    self.code.value,
            "message": self.message,
            "context": self.context,
        }


class DataError(EMMDSError):
    pass

class TrainingError(EMMDSError):
    pass

class PipelineError(EMMDSError):
    pass

class ConfigError(EMMDSError):
    pass


# ── Handlers ─────────────────────────────────────────────────────────

class ErrorHandler:
    """
    Wraps pipeline stages with graceful error handling.
    Classifies exceptions, logs them, and returns structured error dicts.
    """

    def __init__(self, reraise: bool = False):
        """
        Args:
            reraise: If True, re-raise after logging. If False, return error dict.
        """
        self.reraise = reraise
        self.errors: list = []

    def handle(
        self,
        fn: Callable,
        *args,
        stage_name: str = "unknown",
        fallback: Any = None,
        **kwargs,
    ) -> Any:
        """
        Execute fn(*args, **kwargs) safely.

        On success: returns the function result.
        On failure: logs the error, appends to self.errors, returns fallback.

        Usage:
            result = handler.handle(
                trainer.train_all, X_train, y_train,
                stage_name="Training",
                fallback={},
            )
        """
        try:
            return fn(*args, **kwargs)
        except EMMDSError as e:
            self._record(stage_name, e.code.value, str(e), e.context)
            if self.reraise:
                raise
            return fallback
        except Exception as e:
            code = self._classify(e)
            tb   = traceback.format_exc()
            self._record(stage_name, code, str(e), {"traceback": tb})
            if self.reraise:
                raise PipelineError(ErrorCode.PIPELINE_STAGE_FAILED, str(e), {"stage": stage_name}) from e
            return fallback

    def _classify(self, exc: Exception) -> str:
        """Map common exception types to EMMDS error codes."""
        msg = str(exc).lower()
        if "memory" in msg:
            return ErrorCode.MODEL_FIT_FAILED.value
        if "convergence" in msg or "did not converge" in msg:
            return ErrorCode.MODEL_FIT_FAILED.value
        if "feature" in msg and ("name" in msg or "shape" in msg):
            return ErrorCode.BAD_FEATURE_TYPES.value
        return ErrorCode.UNKNOWN.value

    def _record(self, stage: str, code: str, message: str, context: dict):
        entry = {"stage": stage, "code": code, "message": message, "context": context}
        self.errors.append(entry)
        logger.error(f"[{stage}] {code}: {message}")

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_errors(self) -> list:
        return self.errors

    def clear(self):
        self.errors = []


# ── Convenience validators ────────────────────────────────────────────

def assert_not_empty(df, context: str = ""):
    import pandas as pd
    if df is None or (hasattr(df, "empty") and df.empty):
        raise DataError(
            ErrorCode.EMPTY_DATASET,
            f"Dataset is empty. {context}",
        )


def assert_target_exists(df, target_col: str):
    if target_col not in df.columns:
        raise DataError(
            ErrorCode.MISSING_TARGET,
            f"Target column '{target_col}' not found. Available: {list(df.columns)}",
            {"target_col": target_col, "columns": list(df.columns)},
        )


def assert_min_samples(df, min_n: int = 10):
    if len(df) < min_n:
        raise DataError(
            ErrorCode.INSUFFICIENT_SAMPLES,
            f"Dataset has only {len(df)} rows. Minimum required: {min_n}.",
            {"n_rows": len(df), "min_required": min_n},
        )


def assert_enough_classes(y, min_classes: int = 2):
    import pandas as pd
    n_unique = pd.Series(y).nunique()
    if n_unique < min_classes:
        raise DataError(
            ErrorCode.SINGLE_CLASS,
            f"Target has only {n_unique} unique value(s). Classification requires ≥ {min_classes}.",
            {"n_unique": n_unique},
        )
