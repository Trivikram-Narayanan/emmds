"""
EMMDS LIME Explainer
Instance-level explanations using LIME.
Explains individual predictions in human-readable form.
"""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger(__name__)


class LIMEExplainer:
    """
    Produces LIME-based local explanations for individual predictions.
    Best for communicating "why did the model make THIS decision?".
    """

    def __init__(self):
        self._explainer = None
        self._feature_names: list = []
        self._class_names: list = []
        self._task: str = "classification"

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
        task: str = "classification",
    ) -> None:
        """
        Initialize the LIME explainer with training data context.

        Args:
            X_train:       Training data (for LIME's perturbation reference)
            feature_names: Column names
            class_names:   Target class names
            task:          "classification" or "regression"
        """
        try:
            from lime import lime_tabular
        except ImportError:
            logger.warning("LIME not installed. Run: pip install lime")
            return

        self._feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self._class_names = class_names or ["Class 0", "Class 1"]
        self._task = task

        mode = "classification" if task == "classification" else "regression"

        self._explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self._feature_names,
            class_names=self._class_names,
            mode=mode,
            random_state=get("training.random_state", 42),
        )
        logger.info(f"LIME explainer fitted (mode={mode})")

    def explain_instance(
        self,
        instance: np.ndarray,
        model,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> dict:
        """
        Explain a single prediction.

        Args:
            instance:     1D feature array for the instance
            model:        Fitted model with predict_proba
            num_features: Number of features to show
            num_samples:  Number of perturbation samples

        Returns:
            {
                "predicted_class": ...,
                "prediction_probability": ...,
                "feature_contributions": [{feature, weight, direction}, ...]
            }
        """
        if self._explainer is None:
            return {"error": "LIME explainer not fitted. Call .fit() first."}

        num_features = num_features or get("explainability.lime_num_features", 10)
        num_samples = num_samples or get("explainability.lime_num_samples", 500)

        if instance.ndim != 1:
            instance = instance.flatten()

        try:
            predict_fn = (
                model.predict_proba
                if self._task == "classification" and hasattr(model, "predict_proba")
                else model.predict
            )

            explanation = self._explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            # Parse contributions
            contributions = []
            for feat_label, weight in explanation.as_list():
                contributions.append({
                    "feature": feat_label,
                    "weight": round(float(weight), 6),
                    "direction": "positive" if weight > 0 else "negative",
                })

            # Predicted class
            predicted_class = None
            predicted_prob = None
            if self._task == "classification":
                probs = model.predict_proba(instance.reshape(1, -1))[0]
                predicted_class = int(np.argmax(probs))
                predicted_prob = round(float(np.max(probs)), 4)

            return {
                "predicted_class": predicted_class,
                "predicted_class_name": (
                    self._class_names[predicted_class]
                    if predicted_class is not None and predicted_class < len(self._class_names)
                    else str(predicted_class)
                ),
                "prediction_probability": predicted_prob,
                "feature_contributions": contributions,
                "intercept": round(float(explanation.intercept.get(predicted_class, 0.0)), 6),
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"error": str(e)}
