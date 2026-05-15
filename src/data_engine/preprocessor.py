"""
EMMDS Data Preprocessor
Handles: missing values, encoding, feature scaling, train/test split.
Fully stateful — fit on train, transform on test.

Supports any dataset type:
- Numerical, categorical, boolean, datetime columns
- OneHot encoding for low-cardinality categoricals (≤ MAX_OHE_CARDINALITY unique values)
- Ordinal encoding for high-cardinality categoricals
- Datetime columns extracted into year/month/day/hour/weekday/is_weekend
- Text columns detected and dropped with warning (use text_modality for NLP tasks)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler,
    OrdinalEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger(__name__)

MAX_OHE_CARDINALITY = 20   # columns with <= 20 unique values get OneHotEncoded
MAX_TEXT_AVG_LEN    = 50   # columns with avg token length > this are treated as text


class DataPreprocessor:
    """
    Stateful preprocessor. Call fit_transform() on training data,
    then transform() on any new data using the same fitted state.

    Handles:
        - Numeric columns  → median imputation → scaling
        - Low-card cats    → mode imputation → OneHotEncoder
        - High-card cats   → mode imputation → OrdinalEncoder
        - Boolean cols     → cast to int (0/1)
        - Datetime cols    → extract year/month/day/hour/weekday/is_weekend
        - Text cols        → dropped (warning emitted)
    """

    def __init__(self, task: str = "classification", scaler: str = "standard"):
        self.task = task
        self.scaler_type = scaler  # "standard" | "minmax" | "none"

        # Fitted state
        self._numerical_imputer = None
        self._categorical_imputer = None
        self._scaler = None
        self._label_encoder = None

        # Column categories (determined at fit time)
        self._numerical_cols: List[str] = []
        self._bool_cols: List[str] = []
        self._datetime_cols: List[str] = []
        self._ohe_cols: List[str] = []        # low cardinality
        self._ordinal_cols: List[str] = []    # high cardinality
        self._text_cols: List[str] = []       # dropped
        self._ohe_feature_names: List[str] = []

        # Encoders keyed by column name
        self._ohe_encoders: Dict[str, OneHotEncoder] = {}
        self._ordinal_encoders: Dict[str, OrdinalEncoder] = {}

        self.is_fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and return train/test splits.
        Returns: X_train, X_test, y_train, y_test (all numpy arrays)
        """
        test_size    = test_size    or get("training.test_size",    0.2)
        random_state = random_state or get("training.random_state", 42)

        logger.info(f"Preprocessing: task={self.task}, test_size={test_size}")

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Detect column types on the full feature set before splitting
        self._detect_column_types(X)

        # Extract datetime features before split (non-leaking: structure only)
        X = self._extract_datetime_features(X)

        # Drop text columns
        if self._text_cols:
            logger.warning(
                f"Text columns detected and dropped: {self._text_cols}. "
                "Use src/modality/text_modality.py for NLP tasks."
            )
            X = X.drop(columns=[c for c in self._text_cols if c in X.columns])

        # Split FIRST to prevent leakage
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if self.task == "classification" else None,
            )
        except ValueError:
            # stratify fails on continuous or single-class targets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
            )

        X_train_proc = self._fit_and_transform_X(X_train)
        X_test_proc  = self._transform_X(X_test)
        y_train_proc, y_test_proc = self._process_target(y_train, y_test)

        self.is_fitted = True
        logger.info(
            f"Preprocessing complete | "
            f"Train: {X_train_proc.shape} | Test: {X_test_proc.shape}"
        )
        return X_train_proc, X_test_proc, y_train_proc, y_test_proc

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted state (no fitting)."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform() first.")
        X = self._extract_datetime_features(X)
        X = X.drop(columns=[c for c in self._text_cols if c in X.columns], errors="ignore")
        return self._transform_X(X)

    def get_feature_names(self) -> List[str]:
        """Return output feature names in the same order as transform()."""
        names = list(self._numerical_cols)
        names += [f"{c}_bool" for c in self._bool_cols]
        names += self._ohe_feature_names
        names += list(self._ordinal_cols)
        return names

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        if self._label_encoder:
            return self._label_encoder.inverse_transform(y)
        return y

    # ── Column type detection ─────────────────────────────────────────

    def _detect_column_types(self, X: pd.DataFrame) -> None:
        self._bool_cols     = []
        self._datetime_cols = []
        self._numerical_cols = []
        self._ohe_cols      = []
        self._ordinal_cols  = []
        self._text_cols     = []

        for col in X.columns:
            dtype = X[col].dtype

            # Boolean
            if dtype == bool or str(dtype) == "bool":
                self._bool_cols.append(col)
                continue

            # Datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                self._datetime_cols.append(col)
                continue

            # Try parsing string columns as datetime
            if dtype == object:
                sample = X[col].dropna().head(20)
                if _looks_like_datetime(sample):
                    self._datetime_cols.append(col)
                    continue

                # Text heuristic: avg word count > threshold
                avg_words = sample.astype(str).str.split().str.len().mean()
                if avg_words > 6:
                    self._text_cols.append(col)
                    continue

                # Categorical: choose OHE vs ordinal by cardinality
                n_unique = X[col].nunique()
                if n_unique <= MAX_OHE_CARDINALITY:
                    self._ohe_cols.append(col)
                else:
                    self._ordinal_cols.append(col)
                continue

            # Numeric (float, int)
            if pd.api.types.is_numeric_dtype(dtype):
                self._numerical_cols.append(col)
                continue

            # Category dtype
            if hasattr(dtype, "categories"):
                n_unique = X[col].nunique()
                if n_unique <= MAX_OHE_CARDINALITY:
                    self._ohe_cols.append(col)
                else:
                    self._ordinal_cols.append(col)
                continue

            # Fallback: treat as ordinal
            self._ordinal_cols.append(col)

        logger.info(
            f"Column types — numeric:{len(self._numerical_cols)} "
            f"bool:{len(self._bool_cols)} "
            f"datetime:{len(self._datetime_cols)} "
            f"ohe_cat:{len(self._ohe_cols)} "
            f"ordinal_cat:{len(self._ordinal_cols)} "
            f"text(dropped):{len(self._text_cols)}"
        )

    # ── Datetime extraction ───────────────────────────────────────────

    def _extract_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace datetime columns with extracted numeric features."""
        if not self._datetime_cols:
            return X
        X = X.copy()
        drop_cols = []
        new_cols = {}
        for col in self._datetime_cols:
            if col not in X.columns:
                continue
            try:
                dt = pd.to_datetime(X[col], errors="coerce")
                new_cols[f"{col}_year"]      = dt.dt.year.fillna(0).astype(int)
                new_cols[f"{col}_month"]     = dt.dt.month.fillna(0).astype(int)
                new_cols[f"{col}_day"]       = dt.dt.day.fillna(0).astype(int)
                new_cols[f"{col}_hour"]      = dt.dt.hour.fillna(0).astype(int)
                new_cols[f"{col}_weekday"]   = dt.dt.weekday.fillna(0).astype(int)
                new_cols[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
                drop_cols.append(col)
            except Exception as e:
                logger.warning(f"Failed to parse datetime column '{col}': {e}")
        X = X.drop(columns=drop_cols)
        for k, v in new_cols.items():
            X[k] = v.values
            if k not in self._numerical_cols:
                self._numerical_cols.append(k)
        return X

    # ── Feature transformation ────────────────────────────────────────

    def _fit_and_transform_X(self, X: pd.DataFrame) -> np.ndarray:
        parts = []

        # Numeric
        if self._numerical_cols:
            num_cols = [c for c in self._numerical_cols if c in X.columns]
            self._numerical_imputer = SimpleImputer(strategy="median")
            parts.append(self._numerical_imputer.fit_transform(X[num_cols]))
        else:
            self._numerical_imputer = None

        # Boolean → int
        if self._bool_cols:
            bool_cols = [c for c in self._bool_cols if c in X.columns]
            parts.append(X[bool_cols].astype(float).values)

        # OHE categoricals
        if self._ohe_cols:
            ohe_cols = [c for c in self._ohe_cols if c in X.columns]
            cat_imp = SimpleImputer(strategy="most_frequent")
            X_ohe = pd.DataFrame(
                cat_imp.fit_transform(X[ohe_cols]),
                columns=ohe_cols,
            )
            ohe_block = []
            self._ohe_feature_names = []
            for col in ohe_cols:
                enc = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                )
                encoded = enc.fit_transform(X_ohe[[col]])
                self._ohe_encoders[col] = enc
                feat_names = [f"{col}_{c}" for c in enc.get_feature_names_out([col])]
                self._ohe_feature_names.extend(feat_names)
                ohe_block.append(encoded)
            if ohe_block:
                parts.append(np.hstack(ohe_block))
            self._categorical_imputer = cat_imp

        # Ordinal categoricals
        if self._ordinal_cols:
            ord_cols = [c for c in self._ordinal_cols if c in X.columns]
            if not hasattr(self, "_ordinal_imputer"):
                self._ordinal_imputer = SimpleImputer(strategy="most_frequent")
            X_ord = pd.DataFrame(
                self._ordinal_imputer.fit_transform(X[ord_cols].astype(str)),
                columns=ord_cols,
            )
            ord_block = []
            for col in ord_cols:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                ord_block.append(enc.fit_transform(X_ord[[col]]))
                self._ordinal_encoders[col] = enc
            if ord_block:
                parts.append(np.hstack(ord_block))

        X_combined = np.hstack(parts) if parts else np.zeros((len(X), 1))

        # Scaling
        if self.scaler_type == "standard":
            self._scaler = StandardScaler()
            X_combined = self._scaler.fit_transform(X_combined)
        elif self.scaler_type == "minmax":
            self._scaler = MinMaxScaler()
            X_combined = self._scaler.fit_transform(X_combined)

        return X_combined

    def _transform_X(self, X: pd.DataFrame) -> np.ndarray:
        parts = []

        if self._numerical_imputer is not None:
            num_cols = [c for c in self._numerical_cols if c in X.columns]
            missing = [c for c in self._numerical_cols if c not in X.columns]
            if missing:
                logger.warning(f"Missing numeric columns at transform time: {missing}")
            if num_cols:
                parts.append(self._numerical_imputer.transform(X[num_cols]))

        if self._bool_cols:
            bool_cols = [c for c in self._bool_cols if c in X.columns]
            if bool_cols:
                parts.append(X[bool_cols].astype(float).values)

        if self._ohe_cols and self._ohe_encoders:
            ohe_cols = [c for c in self._ohe_cols if c in X.columns]
            if ohe_cols and hasattr(self, "_categorical_imputer"):
                X_ohe = pd.DataFrame(
                    self._categorical_imputer.transform(X[ohe_cols]),
                    columns=ohe_cols,
                )
                ohe_block = []
                for col in ohe_cols:
                    if col in self._ohe_encoders:
                        ohe_block.append(self._ohe_encoders[col].transform(X_ohe[[col]]))
                if ohe_block:
                    parts.append(np.hstack(ohe_block))

        if self._ordinal_cols and self._ordinal_encoders:
            ord_cols = [c for c in self._ordinal_cols if c in X.columns]
            if ord_cols and hasattr(self, "_ordinal_imputer"):
                X_ord = pd.DataFrame(
                    self._ordinal_imputer.transform(X[ord_cols].astype(str)),
                    columns=ord_cols,
                )
                ord_block = []
                for col in ord_cols:
                    if col in self._ordinal_encoders:
                        ord_block.append(self._ordinal_encoders[col].transform(X_ord[[col]]))
                if ord_block:
                    parts.append(np.hstack(ord_block))

        X_combined = np.hstack(parts) if parts else np.zeros((len(X), 1))

        if self._scaler is not None:
            X_combined = self._scaler.transform(X_combined)

        return X_combined

    # ── Target processing ─────────────────────────────────────────────

    def _process_target(
        self, y_train: pd.Series, y_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode target for classification; leave as-is for regression."""
        if self.task == "classification":
            self._label_encoder = LabelEncoder()
            y_train_enc = self._label_encoder.fit_transform(y_train.astype(str))
            y_test_enc  = self._label_encoder.transform(y_test.astype(str))
            return y_train_enc, y_test_enc
        # Regression: ensure numeric
        return (
            pd.to_numeric(y_train, errors="coerce").fillna(0).to_numpy(),
            pd.to_numeric(y_test,  errors="coerce").fillna(0).to_numpy(),
        )


# ── Helpers ────────────────────────────────────────────────────────────

def _looks_like_datetime(sample: pd.Series) -> bool:
    """Heuristic: try parsing 5 values as datetime."""
    try:
        parsed = pd.to_datetime(sample.head(5), errors="coerce")
        return parsed.notna().mean() >= 0.6
    except Exception:
        return False
