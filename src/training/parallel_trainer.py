"""
EMMDS Parallel Trainer
Trains all models simultaneously using joblib.Parallel.
Dramatically faster than sequential training for large model sets.

Each model is cloned and trained in its own subprocess/thread.
Failures in one job do not affect others.
"""

import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _train_one(name: str, model, X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Worker function: clone and train a single model.
    Returns (name, fitted_model, elapsed_seconds) or (name, None, elapsed).
    """
    t0 = time.time()
    try:
        fitted = clone(model)
        fitted.fit(X, y)
        elapsed = round(time.time() - t0, 3)
        return name, fitted, elapsed, None
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return name, None, elapsed, str(e)


class ParallelTrainer:
    """
    Trains all models in parallel using joblib.

    Usage:
        trainer = ParallelTrainer(n_jobs=-1)
        trained = trainer.train_all(models_dict, X_train, y_train)
    """

    def __init__(self, n_jobs: int = -1, backend: str = "loky", verbose: int = 0):
        """
        Args:
            n_jobs:   Number of parallel jobs (-1 = all CPU cores)
            backend:  joblib backend — "loky" (default), "threading", "multiprocessing"
            verbose:  joblib verbosity level (0 = silent)
        """
        self.n_jobs   = n_jobs
        self.backend  = backend
        self.verbose  = verbose

        self.trained_models:  dict = {}
        self.training_times:  dict = {}
        self.failed_models:   list = []

    def train_all(
        self,
        models: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        enabled_models: Optional[list] = None,
    ) -> dict:
        """
        Train all models in parallel.

        Args:
            models:         {name: unfitted_model}
            X_train:        Training features
            y_train:        Training labels
            enabled_models: Optional filter list

        Returns:
            {name: fitted_model}  (excludes failed models)
        """
        if enabled_models:
            models = {k: v for k, v in models.items() if k in enabled_models}

        n = len(models)
        logger.info(
            f"ParallelTrainer: {n} models | n_jobs={self.n_jobs} | backend={self.backend}"
        )
        wall_start = time.time()

        results = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose)(
            delayed(_train_one)(name, model, X_train, y_train)
            for name, model in models.items()
        )

        self.trained_models  = {}
        self.training_times  = {}
        self.failed_models   = []

        for name, fitted, elapsed, err in results:
            if fitted is not None:
                self.trained_models[name] = fitted
                self.training_times[name] = elapsed
                logger.info(f"  ✅ {name:25s} trained in {elapsed:.3f}s")
            else:
                self.failed_models.append(name)
                logger.error(f"  ❌ {name:25s} failed ({elapsed:.3f}s): {err}")

        total_wall = round(time.time() - wall_start, 3)
        sum_serial = round(sum(self.training_times.values()), 3)
        speedup    = round(sum_serial / total_wall, 2) if total_wall > 0 else 1.0

        logger.info(
            f"Parallel training done | "
            f"Success: {len(self.trained_models)} | Failed: {len(self.failed_models)} | "
            f"Wall: {total_wall}s (serial would be ≈{sum_serial}s, speedup ≈{speedup}x)"
        )
        return self.trained_models

    def get_summary(self) -> dict:
        return {
            "trained":        list(self.trained_models.keys()),
            "failed":         self.failed_models,
            "training_times": self.training_times,
            "total_time":     round(sum(self.training_times.values()), 3),
            "n_jobs_used":    self.n_jobs,
        }
