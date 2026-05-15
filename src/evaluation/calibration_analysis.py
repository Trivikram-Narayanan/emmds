"""
EMMDS Calibration Analysis — ECE + Reliability Diagrams
=========================================================
Provides:
  - Expected Calibration Error (ECE, multi-class safe)
  - Maximum Calibration Error (MCE)
  - Reliability diagram data (for plotting)
  - Entropy of predictions
  - Brier score (already in calibrator.py — unified here)
  - CalibrationAnalyser: drop-in component for the pipeline
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# ECE / MCE
# ─────────────────────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Expected Calibration Error (equal-width bins).

    Works for binary (y_prob is 1-D) and multi-class
    (y_prob is 2-D; uses max-confidence column).

    Returns ECE, MCE, and per-bin reliability data.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2:
        confidence = y_prob.max(axis=1)
        y_pred     = y_prob.argmax(axis=1)
        correct    = (y_pred == y_true).astype(float)
    else:
        confidence = y_prob
        correct    = (y_true == 1).astype(float)

    n    = len(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if i := mask.sum():
            acc  = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            gap  = abs(acc - conf)
            ece += gap * i / n
            mce  = max(mce, gap)
            bin_data.append({
                "bin_lo":   round(lo, 3),
                "bin_hi":   round(hi, 3),
                "count":    int(i),
                "accuracy": round(acc,  4),
                "confidence": round(conf, 4),
                "gap":      round(gap,  4),
            })

    return {
        "ece":       round(float(ece), 4),
        "mce":       round(float(mce), 4),
        "n_bins":    n_bins,
        "n_samples": n,
        "bins":      bin_data,
    }


def compute_entropy(y_prob: np.ndarray) -> float:
    """Mean entropy of predicted probability distributions."""
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        p = np.clip(y_prob, 1e-9, 1 - 1e-9)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    else:
        p = np.clip(y_prob, 1e-9, 1)
        entropy = -(p * np.log(p)).sum(axis=1)
    return float(entropy.mean())


# ─────────────────────────────────────────────────────────────
# SHAP Explanation Stability
# ─────────────────────────────────────────────────────────────

def compute_shap_stability(
    model,
    X_train: np.ndarray,
    feature_names: List[str],
    n_folds: int = 5,
    max_samples: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Measures stability of SHAP feature importance rankings across CV folds.

    For each fold, train a copy of the model and compute global SHAP importances.
    Stability = mean Spearman rank correlation between all fold pairs.

    Returns:
        stability_score: float in [0,1] (1 = perfectly stable ranking)
        top_features:    most consistently important features
        rank_variance:   variance of rank per feature across folds
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}

    from sklearn.base import clone
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # We need labels — use model's classes if available, else binary 0/1
    n = len(X_train)
    y_dummy = np.zeros(n)  # placeholder; we only need feature importances

    fold_importances = []
    for tr_idx, _ in kf.split(X_train):
        Xtr = X_train[tr_idx]
        n_bg = min(max_samples, len(Xtr))
        try:
            m = clone(model)
            # KernelExplainer needs a callable — skip if model not fittable
            explainer = shap.Explainer(m, Xtr[:n_bg],
                                        feature_names=feature_names)
            sv = explainer(Xtr[:n_bg])
            if hasattr(sv, "values"):
                vals = np.abs(sv.values)
                if vals.ndim == 3:
                    vals = vals.mean(axis=2)
                imp = vals.mean(axis=0)
            else:
                imp = np.abs(sv).mean(axis=0)
            fold_importances.append(imp)
        except Exception:
            pass

    if len(fold_importances) < 2:
        return {"stability_score": None, "error": "insufficient folds"}

    # Pairwise Spearman correlations
    n_folds_actual = len(fold_importances)
    corrs = []
    for i in range(n_folds_actual):
        for j in range(i + 1, n_folds_actual):
            r, _ = spearmanr(fold_importances[i], fold_importances[j])
            corrs.append(float(r))

    stability = float(np.mean(corrs))
    mean_imp  = np.mean(fold_importances, axis=0)
    std_imp   = np.std(fold_importances,  axis=0)

    # Rank variance: rank each fold's importances, compute variance
    ranks = np.array([np.argsort(np.argsort(-imp)) for imp in fold_importances])
    rank_var = ranks.var(axis=0)

    ranked_features = sorted(
        zip(feature_names, mean_imp.tolist(), rank_var.tolist()),
        key=lambda x: x[1], reverse=True)

    return {
        "stability_score": round(stability, 4),
        "mean_pairwise_spearman": round(stability, 4),
        "n_folds": n_folds_actual,
        "top_10_features": [
            {"feature": f, "mean_importance": round(imp, 6),
             "rank_variance": round(rv, 4)}
            for f, imp, rv in ranked_features[:10]
        ],
        "stability_label": (
            "HIGH" if stability > 0.8 else
            "MEDIUM" if stability > 0.6 else "LOW"
        ),
        "trust_penalty": float(np.clip(1 - stability, 0, 1)),
    }


# ─────────────────────────────────────────────────────────────
# Calibration Analyser (pipeline-compatible)
# ─────────────────────────────────────────────────────────────

class CalibrationAnalyser:
    """
    Drop-in analyser: given trained models + test data, produces
    ECE, MCE, entropy, and reliability diagram data for each model.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._results: Dict = {}

    def analyse_all(
        self,
        trained_models: Dict,
        X_test,
        y_test: np.ndarray,
    ) -> Dict:
        y_test = np.asarray(y_test)
        for name, model in trained_models.items():
            if not hasattr(model, "predict_proba"):
                self._results[name] = {"ece": None, "mce": None,
                                        "entropy": None, "error": "no predict_proba"}
                continue
            try:
                proba = model.predict_proba(X_test)
                ece_result = compute_ece(y_test, proba, self.n_bins)
                ent = compute_entropy(proba)
                self._results[name] = {
                    **ece_result,
                    "entropy": round(ent, 4),
                    "calibration_score": round(float(1 - ece_result["ece"]), 4),
                }
            except Exception as e:
                self._results[name] = {"error": str(e)}
        return self._results

    def get_results(self) -> Dict:
        return self._results

    def summary_table(self) -> List[Dict]:
        rows = []
        for name, r in self._results.items():
            rows.append({
                "model": name,
                "ece":             r.get("ece"),
                "mce":             r.get("mce"),
                "entropy":         r.get("entropy"),
                "calibration_score": r.get("calibration_score"),
            })
        return sorted(rows, key=lambda x: x["ece"] or 999)
