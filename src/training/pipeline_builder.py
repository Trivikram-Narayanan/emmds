"""
EMMDS Pipeline Builder
Wraps every model in a sklearn Pipeline:
    preprocessor → model

Handles: numeric, low/high cardinality categorical, boolean, datetime.
Text columns are dropped with a warning.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_OHE_CARDINALITY = 20   # <= this → OneHotEncoder, else OrdinalEncoder


# ---------------------------------------------------------------------------
# Custom transformer: datetime → numeric features
# ---------------------------------------------------------------------------

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert one datetime column to year/month/day/hour/weekday/is_weekend."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(X[:, 0])
        dt = pd.to_datetime(col, errors="coerce")
        return np.column_stack([
            dt.dt.year.fillna(0).astype(int),
            dt.dt.month.fillna(0).astype(int),
            dt.dt.day.fillna(0).astype(int),
            dt.dt.hour.fillna(0).astype(int),
            dt.dt.weekday.fillna(0).astype(int),
            (dt.dt.weekday >= 5).astype(int),
        ])

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features else "dt"
        return [f"{prefix}_{s}" for s in
                ["year", "month", "day", "hour", "weekday", "is_weekend"]]


# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------

def _detect_columns(X: pd.DataFrame):
    """Classify columns into numeric, ohe_cat, ordinal_cat, bool, datetime, text."""
    num, ohe, ordinal, bool_cols, datetime_cols, text_cols = [], [], [], [], [], []

    for col in X.columns:
        dtype = X[col].dtype
        if dtype == bool or str(dtype) == "bool":
            bool_cols.append(col); continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col); continue
        if pd.api.types.is_numeric_dtype(dtype):
            num.append(col); continue
        # object / category
        sample = X[col].dropna().head(30)
        # datetime heuristic
        try:
            parsed = pd.to_datetime(sample.head(5), errors="coerce")
            if parsed.notna().mean() >= 0.6:
                datetime_cols.append(col); continue
        except Exception:
            pass
        # text heuristic
        avg_words = sample.astype(str).str.split().str.len().mean()
        if avg_words > 6:
            text_cols.append(col); continue
        # cardinality-based cat split
        n_unique = X[col].nunique()
        if n_unique <= MAX_OHE_CARDINALITY:
            ohe.append(col)
        else:
            ordinal.append(col)

    if text_cols:
        logger.warning(
            f"Text columns dropped (use text_modality for NLP): {text_cols}"
        )
    return num, ohe, ordinal, bool_cols, datetime_cols


# ---------------------------------------------------------------------------
# Build ColumnTransformer
# ---------------------------------------------------------------------------

def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scaler: str = "standard",
    X_sample: Optional[pd.DataFrame] = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer.

    If X_sample is provided, automatically splits categorical_cols into
    OHE (low-cardinality) and ordinal (high-cardinality) and detects
    datetime and boolean columns.

    Falls back to OrdinalEncoder for all categoricals when X_sample=None
    (backwards-compatible).
    """
    transformers = []

    # Numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    num_pipeline = Pipeline(num_steps)

    if X_sample is not None:
        # Smart detection
        num_auto, ohe_cols, ordinal_cols, bool_cols, datetime_cols = \
            _detect_columns(X_sample)

        # Merge explicitly passed numerical_cols with auto-detected
        all_num = list(dict.fromkeys(numerical_cols + num_auto))
        # Remove anything that ended up in other buckets
        all_num = [c for c in all_num if c not in ohe_cols + ordinal_cols
                   + bool_cols + datetime_cols]

        if all_num:
            transformers.append(("num", num_pipeline, all_num))

        # Boolean → passthrough as float
        if bool_cols:
            from sklearn.preprocessing import FunctionTransformer
            bool_pipe = Pipeline([
                ("to_float", FunctionTransformer(lambda x: x.astype(float)))
            ])
            transformers.append(("bool", bool_pipe, bool_cols))

        # Datetime → feature extraction
        for col in datetime_cols:
            dt_pipe = Pipeline([("dt", DatetimeFeatureExtractor())])
            transformers.append((f"dt_{col}", dt_pipe, [col]))

        # OHE categoricals
        if ohe_cols:
            ohe_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                )),
            ])
            transformers.append(("ohe", ohe_pipe, ohe_cols))

        # Ordinal categoricals — exclude bool/datetime already handled above
        all_ordinal = list(dict.fromkeys(
            [c for c in categorical_cols
             if c not in ohe_cols + bool_cols + datetime_cols] + ordinal_cols
        ))
        if all_ordinal:
            ord_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ])
            transformers.append(("ord", ord_pipe, all_ordinal))

    else:
        # Backwards-compatible: ordinal encode everything
        if numerical_cols:
            transformers.append(("num", num_pipeline, numerical_cols))
        if categorical_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ])
            transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=-1,
    )


def build_model_pipeline(
    model,
    numerical_cols: List[str],
    categorical_cols: List[str],
    scaler: str = "standard",
    X_sample: Optional[pd.DataFrame] = None,
) -> Pipeline:
    """Build a complete sklearn Pipeline: preprocessor → model."""
    preprocessor = build_preprocessor(
        numerical_cols, categorical_cols, scaler, X_sample=X_sample
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_all_pipelines(
    models: dict,
    numerical_cols: List[str],
    categorical_cols: List[str],
    scaler: str = "standard",
    X_sample: Optional[pd.DataFrame] = None,
) -> dict:
    """Build a Pipeline for every model in the dict."""
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = build_model_pipeline(
            model, numerical_cols, categorical_cols, scaler, X_sample=X_sample
        )
        logger.debug(f"Pipeline built: {name}")
    logger.info(f"Built {len(pipelines)} model pipeline(s)")
    return pipelines


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list:
    try:
        ct = pipeline.named_steps["preprocessor"]
        names = []
        for tname, transformer, cols in ct.transformers_:
            if tname != "remainder" and isinstance(cols, list):
                names.extend(cols)
        return names
    except Exception:
        return []
