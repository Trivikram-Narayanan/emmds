"""
EMMDS Calibrator
Classification: CalibratedClassifierCV (isotonic/sigmoid), score = 1 - Brier.
Regression:     No probability calibration; score = reliability of predictions
                measured as max(0, 1 - RMSE/std(y)) — a model that predicts as
                well as the mean gets score ≈ 0; a near-perfect predictor gets ≈ 1.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, mean_squared_error
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCalibrator:
    """
    Wraps trained models in probability calibration (classification) or
    computes a residual-reliability score (regression).

    Calibration scores are in [0, 1] — higher is better calibrated.
    Passed directly into the TrustScoreEngine calibration component.
    """

    def __init__(self, method: str = "isotonic", cv: int = 3, task: str = "classification"):
        self.method = method
        self.cv = cv
        self.task = task
        self.calibrated_models: dict = {}
        self.calibration_scores: dict = {}

    def calibrate_all(
        self,
        trained_models: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Calibrate all models. Returns {model_name: (calibrated) model}.
        For regression, models are returned unchanged; only scores differ.
        """
        logger.info(f"Calibrating {len(trained_models)} model(s) [{self.task}]...")

        for name, model in trained_models.items():
            if self.task == "regression":
                score = self._regression_calibration_score(model, X_test, y_test)
                self.calibrated_models[name] = model   # unchanged
                self.calibration_scores[name] = score
                logger.info(f"  ✅ {name:30s} | residual reliability: {score:.4f}")
                continue

            # Classification path
            if not hasattr(model, "predict_proba"):
                logger.debug(f"  ⏭️  {name}: no predict_proba — skipping calibration")
                self.calibrated_models[name] = model
                self.calibration_scores[name] = None
                continue

            try:
                from sklearn.base import clone as _clone
                try:
                    calibrated = CalibratedClassifierCV(
                        estimator=model, method=self.method, cv="prefit"
                    )
                    calibrated.fit(X_train, y_train)
                except TypeError:
                    calibrated = CalibratedClassifierCV(
                        estimator=_clone(model), method=self.method, cv=self.cv
                    )
                    calibrated.fit(X_train, y_train)

                score = self._classification_calibration_score(calibrated, X_test, y_test)
                self.calibrated_models[name] = calibrated
                self.calibration_scores[name] = score
                logger.info(f"  ✅ {name:30s} | calibration score: {score:.4f}")

            except Exception as e:
                logger.warning(f"  ⚠️  {name}: calibration failed ({e}) — using original")
                self.calibrated_models[name] = model
                self.calibration_scores[name] = 0.5

        return self.calibrated_models

    # ── Scoring ────────────────────────────────────────────────────────

    def _classification_calibration_score(self, model, X_test, y_test) -> float:
        """1 - Brier score (binary or macro-averaged multi-class)."""
        try:
            proba = model.predict_proba(X_test)
            classes = np.unique(y_test)
            if len(classes) == 2:
                brier = brier_score_loss(y_test, proba[:, 1], pos_label=classes[1])
            else:
                brier = float(np.mean([
                    brier_score_loss((y_test == cls).astype(int), proba[:, i])
                    for i, cls in enumerate(classes)
                ]))
            return round(float(np.clip(1.0 - brier, 0, 1)), 4)
        except Exception:
            return 0.5

    def _regression_calibration_score(self, model, X_test, y_test) -> float:
        """
        Residual reliability: max(0, 1 - RMSE / std(y)).
        A model predicting the mean has RMSE ≈ std(y) → score ≈ 0.
        A near-perfect predictor has RMSE ≈ 0 → score ≈ 1.
        """
        try:
            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            std_y = float(np.std(y_test))
            if std_y < 1e-9:
                return 1.0 if rmse < 1e-9 else 0.0
            score = float(np.clip(1.0 - rmse / std_y, 0.0, 1.0))
            return round(score, 4)
        except Exception:
            return 0.5

    def get_calibration_scores(self) -> dict:
        return self.calibration_scores

    def get_calibrated_model(self, name: str):
        return self.calibrated_models.get(name)
