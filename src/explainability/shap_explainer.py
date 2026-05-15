"""
EMMDS SHAP Explainer
Global + local explanations using SHAP.
Supports tree-based and kernel explainers with auto-detection.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    Produces SHAP-based feature importance explanations.
    - Global: mean |SHAP| per feature
    - Local:  SHAP values for a single instance
    """

    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self._explainer = None
        self._shap_values = None
        self._feature_names: list = []

    def fit(
        self,
        model,
        X_background: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> None:
        """
        Fit SHAP explainer to a trained model.

        Args:
            model:          Any fitted sklearn-compatible model
            X_background:   Background dataset for explainer
            feature_names:  Column names for display
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            return

        self._feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]

        # Subsample background for speed
        n = min(self.max_samples, len(X_background))
        background = shap.sample(X_background, n) if len(X_background) > n else X_background

        # Auto-select best explainer
        model_type = type(model).__name__.lower()
        try:
            if any(k in model_type for k in ["forest", "tree", "boost", "xgb", "lgbm"]):
                self._explainer = shap.TreeExplainer(model)
                logger.info("Using TreeExplainer")
            elif "linear" in model_type or "logistic" in model_type:
                self._explainer = shap.LinearExplainer(model, background)
                logger.info("Using LinearExplainer")
            else:
                # Use predict_proba for classifiers, predict for regressors
                predict_fn = (model.predict_proba
                              if hasattr(model, "predict_proba")
                              else model.predict)
                self._explainer = shap.KernelExplainer(predict_fn, background)
                logger.info("Using KernelExplainer (slow for large datasets)")
        except Exception as e:
            logger.warning(f"Primary SHAP explainer failed ({e}), falling back to KernelExplainer")
            try:
                predict_fn = (model.predict_proba
                              if hasattr(model, "predict_proba")
                              else model.predict)
                self._explainer = shap.KernelExplainer(predict_fn, background)
            except Exception as e2:
                logger.error(f"KernelExplainer also failed: {e2}")
                self._explainer = None

    def explain_global(self, X: np.ndarray) -> dict:
        """
        Global feature importance: mean |SHAP value| per feature.

        Returns:
            {
                "feature_names": [...],
                "mean_abs_shap": [...],
                "ranking": [{feature, importance}, ...]
            }
        """
        if self._explainer is None:
            return {"error": "SHAP explainer not fitted."}

        try:
            n = min(self.max_samples, len(X))
            X_sample = X[:n]
            shap_values = self._explainer.shap_values(X_sample)

            # For multi-class, average across classes
            if isinstance(shap_values, list):
                shap_matrix = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_matrix = np.abs(shap_values)

            mean_abs = np.mean(shap_matrix, axis=0)
            self._shap_values = shap_matrix

            feature_importance = [
                {"feature": name, "importance": round(float(imp), 6)}
                for name, imp in zip(self._feature_names, mean_abs)
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "feature_names": self._feature_names,
                "mean_abs_shap": mean_abs.tolist(),
                "ranking": feature_importance,
                "top_features": [r["feature"] for r in feature_importance[:10]],
            }

        except Exception as e:
            logger.error(f"SHAP global explanation failed: {e}")
            return {"error": str(e)}

    def explain_instance(self, instance: np.ndarray) -> dict:
        """
        Local SHAP explanation for a single prediction instance.

        Args:
            instance: 1D or 2D array (single sample)

        Returns:
            {feature_name: shap_value, ...}
        """
        if self._explainer is None:
            return {"error": "SHAP explainer not fitted."}

        try:
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)

            shap_values = self._explainer.shap_values(instance)

            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                sv = shap_values[0]

            result = {
                name: round(float(val), 6)
                for name, val in zip(self._feature_names, sv)
            }
            # Sort by absolute impact
            sorted_result = dict(
                sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            return sorted_result

        except Exception as e:
            logger.error(f"SHAP instance explanation failed: {e}")
            return {"error": str(e)}
