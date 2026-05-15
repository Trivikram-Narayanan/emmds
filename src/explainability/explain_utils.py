"""
EMMDS Explain Utils
Shared helpers for SHAP and LIME modules.
"""

import numpy as np
import pandas as pd
from typing import Optional


def top_n_features(
    importance_dict: dict,
    n: int = 10,
    absolute: bool = True,
) -> list:
    """
    Return top-N features sorted by importance.

    Args:
        importance_dict: {feature_name: importance_value}
        n:              Number of features to return
        absolute:       Sort by absolute value if True

    Returns:
        List of {feature, importance} dicts
    """
    items = [
        {"feature": k, "importance": float(v)}
        for k, v in importance_dict.items()
    ]
    items.sort(key=lambda x: abs(x["importance"]) if absolute else x["importance"], reverse=True)
    return items[:n]


def normalize_importances(importances: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    min_val = importances.min()
    max_val = importances.max()
    if max_val == min_val:
        return np.zeros_like(importances)
    return (importances - min_val) / (max_val - min_val)


def format_lime_explanation(lime_list: list) -> list:
    """
    Convert raw LIME output list to structured dicts.

    Args:
        lime_list: [(feature_label, weight), ...]

    Returns:
        [{feature, weight, direction}, ...]
    """
    result = []
    for feat_label, weight in lime_list:
        result.append({
            "feature": feat_label,
            "weight": round(float(weight), 6),
            "direction": "positive" if weight > 0 else "negative",
        })
    return sorted(result, key=lambda x: abs(x["weight"]), reverse=True)


def aggregate_shap_for_multiclass(shap_values: list) -> np.ndarray:
    """
    Average absolute SHAP values across all classes.

    Args:
        shap_values: List of arrays, one per class

    Returns:
        2D array of averaged |SHAP| values
    """
    return np.mean([np.abs(sv) for sv in shap_values], axis=0)
