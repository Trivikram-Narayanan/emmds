"""
Fairness evaluation metrics for EMMDS trust scoring.

Implements demographic parity gap and equalized odds (TPR + FPR parity)
aligned with EU AI Act Article 10 requirements for high-risk ML systems.

Reference: Hardt et al. (2016), "Equality of Opportunity in Supervised Learning"
"""
import numpy as np
from typing import Dict, Optional


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 1e-12 else default


# ---------------------------------------------------------------------------
# Core group metrics
# ---------------------------------------------------------------------------

def group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-group classification metrics.

    Args:
        y_true:    Binary ground truth labels (0/1).
        y_pred:    Binary predictions (0/1).
        sensitive: Categorical sensitive attribute (e.g., [0, 1, 0, ...]).

    Returns:
        Dict keyed by group label, each with keys:
            n, positive_rate, tpr, fpr, precision, accuracy
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    sensitive = np.asarray(sensitive)

    groups = np.unique(sensitive)
    result = {}
    for g in groups:
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        n = len(yt)
        pos = (yt == 1).sum()
        neg = n - pos
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        result[str(g)] = {
            "n": int(n),
            "positive_rate": float(_safe_divide(yp.sum(), n)),
            "tpr": float(_safe_divide(tp, pos)),
            "fpr": float(_safe_divide(fp, neg)),
            "precision": float(_safe_divide(tp, tp + fp)),
            "accuracy": float(_safe_divide(tp + tn, n)),
        }
    return result


# ---------------------------------------------------------------------------
# Gap metrics
# ---------------------------------------------------------------------------

def demographic_parity_gap(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """
    Max difference in positive prediction rates across groups.
    0 = perfect demographic parity.
    """
    sensitive = np.asarray(sensitive)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.unique(sensitive)
    rates = [y_pred[sensitive == g].mean() for g in groups]
    return float(max(rates) - min(rates))


def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, float]:
    """
    Max difference in TPR and FPR across groups.
    Both should be 0 for perfect equalized odds.
    """
    metrics = group_metrics(y_true, y_pred, sensitive)
    tprs = [v["tpr"] for v in metrics.values()]
    fprs = [v["fpr"] for v in metrics.values()]
    return {
        "tpr_gap": float(max(tprs) - min(tprs)),
        "fpr_gap": float(max(fprs) - min(fprs)),
    }


def fairness_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    dp_threshold: float = 0.10,
    eo_threshold: float = 0.10,
) -> dict:
    """
    Full fairness report for a single model's predictions.

    Returns a dict with:
        - demographic_parity_gap
        - equalized_odds_tpr_gap
        - equalized_odds_fpr_gap
        - dp_pass   (gap < dp_threshold)
        - eo_pass   (both gaps < eo_threshold)
        - per_group metrics
        - overall_fairness_score  [0, 1] — higher is fairer
    """
    dp = demographic_parity_gap(y_pred, sensitive)
    eo = equalized_odds_gap(y_true, y_pred, sensitive)
    per_group = group_metrics(y_true, y_pred, sensitive)

    # Composite fairness score: penalise each gap linearly, clipped to [0,1]
    dp_score = max(0.0, 1.0 - dp / dp_threshold)
    eo_score = max(0.0, 1.0 - max(eo["tpr_gap"], eo["fpr_gap"]) / eo_threshold)
    fairness_score = 0.5 * dp_score + 0.5 * eo_score

    return {
        "demographic_parity_gap": dp,
        "equalized_odds_tpr_gap": eo["tpr_gap"],
        "equalized_odds_fpr_gap": eo["fpr_gap"],
        "dp_pass": bool(dp < dp_threshold),
        "eo_pass": bool(eo["tpr_gap"] < eo_threshold and eo["fpr_gap"] < eo_threshold),
        "overall_fairness_score": round(fairness_score, 4),
        "per_group": per_group,
    }


# ---------------------------------------------------------------------------
# Integration with EMMDS trust score
# ---------------------------------------------------------------------------

def fairness_trust_component(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: Optional[np.ndarray],
    dp_threshold: float = 0.10,
    eo_threshold: float = 0.10,
) -> float:
    """
    Fairness component for inclusion in the EMMDS trust score.
    Returns a value in [0, 1]; 1 = perfectly fair.

    If sensitive is None (no protected attribute provided),
    returns 1.0 (neutral — no penalty).
    """
    if sensitive is None or len(np.unique(sensitive)) < 2:
        return 1.0
    report = fairness_summary(y_true, y_pred, sensitive, dp_threshold, eo_threshold)
    return report["overall_fairness_score"]


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_model_fairness(
    y_true: np.ndarray,
    sensitive: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """
    Run fairness_summary for each model and return a dict keyed by model name.
    Useful for comparing candidate models before deployment selection.
    """
    return {
        name: fairness_summary(y_true, preds, sensitive)
        for name, preds in model_predictions.items()
    }
