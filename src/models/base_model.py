"""
EMMDS Base Model
Abstract interface that all model wrappers must implement.
Provides a consistent API: fit, predict, predict_proba, score.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEMMDSModel(ABC):
    """
    Abstract base class for EMMDS model wrappers.
    All concrete models implement this interface.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEMMDSModel":
        """Train on (X, y). Returns self for chaining."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted labels."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.
        Override in models that support it.
        Default raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba.")

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the primary score metric (accuracy for classification, R² for regression)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


class SKLearnModelWrapper(BaseEMMDSModel):
    """
    Thin wrapper around any scikit-learn estimator.
    Makes sklearn models compatible with the BaseEMMDSModel interface.
    """

    def __init__(self, estimator, name: str = None):
        self._estimator = estimator
        self._name = name or type(estimator).__name__

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKLearnModelWrapper":
        self._estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)
        raise NotImplementedError(f"{self._name} does not support predict_proba.")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self._estimator.score(X, y))

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"SKLearnModelWrapper({self._name})"
