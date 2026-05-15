"""
EMMDS Data Split
Stratified train/test/validation splits with leakage prevention.
Single source of truth for all splitting logic in the system.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from typing import Optional, Tuple
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger(__name__)


class DataSplitter:
    """
    Responsible for ALL data splitting in EMMDS.
    Enforces stratification for classification and prevents data leakage
    by ensuring no transformation is fitted before splitting.

    Split contract:
      - Split happens BEFORE any preprocessing
      - Preprocessing is fitted on train only, applied to test
      - CV uses the training set only
    """

    def __init__(
        self,
        task: str = "classification",
        test_size: float = None,
        val_size: Optional[float] = None,
        random_state: int = None,
    ):
        self.task = task
        self.test_size = test_size or get("training.test_size", 0.2)
        self.val_size = val_size  # Optional hold-out validation set
        self.random_state = random_state or get("training.random_state", 42)

        # Record of the split for reproducibility
        self.split_info: dict = {}

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple:
        """
        Split features and labels into train / (val) / test.

        Returns:
            Without val_size:  (X_train, X_test, y_train, y_test)
            With    val_size:  (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify = y if self.task == "classification" else None
        n_classes = y.nunique() if self.task == "classification" else None

        # Guard: can't stratify if a class has < 2 members
        if stratify is not None:
            min_class_count = y.value_counts().min()
            if min_class_count < 2:
                logger.warning(
                    f"Class imbalance too extreme for stratification "
                    f"(min class has {min_class_count} sample(s)). Falling back to random split."
                )
                stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        self.split_info = {
            "strategy": "stratified" if stratify is not None else "random",
            "test_size": self.test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_classes": int(n_classes) if n_classes else None,
        }

        if self.val_size is not None:
            # Carve validation out of the training set
            val_ratio_of_train = self.val_size / (1.0 - self.test_size)
            stratify_train = y_train if stratify is not None else None

            if stratify_train is not None:
                min_train_class = y_train.value_counts().min()
                if min_train_class < 2:
                    stratify_train = None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_ratio_of_train,
                random_state=self.random_state,
                stratify=stratify_train,
            )
            self.split_info["val_samples"] = len(X_val)

            logger.info(
                f"Split ({self.split_info['strategy']}) → "
                f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

        logger.info(
            f"Split ({self.split_info['strategy']}) → "
            f"Train: {len(X_train)} | Test: {len(X_test)}"
        )
        return X_train, X_test, y_train, y_test

    def get_cv_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = None,
    ):
        """
        Return a fitted CV splitter object for use in cross_validate().
        Always uses training data only — never the test set.
        """
        n_splits = n_splits or get("training.cv_folds", 5)

        if self.task == "classification":
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def get_split_info(self) -> dict:
        return self.split_info
