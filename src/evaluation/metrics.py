"""
EMMDS Metrics
Computes all evaluation metrics for classification and regression.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> dict:
    """
    Full classification metrics suite.

    Args:
        y_true:  Ground-truth labels
        y_pred:  Predicted labels
        y_prob:  Predicted probabilities (optional, for AUC)
        average: Averaging strategy for multi-class

    Returns:
        Dict of metric name → float value
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }

    # AUC-ROC (requires probability estimates)
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                )
            metrics["auc_roc"] = round(float(auc), 4)
        except Exception as e:
            logger.debug(f"AUC-ROC skipped: {e}")
            metrics["auc_roc"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Full regression metrics suite."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }
