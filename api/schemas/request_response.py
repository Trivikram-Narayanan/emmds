"""
EMMDS API Schemas
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


# ── Upload ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    message: str


# ── Analysis ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    target_col: str = Field(..., description="Name of the target column")
    task: Optional[str] = Field(None, description="'classification' or 'regression' (auto-detected if omitted)")


class AnalyzeResponse(BaseModel):
    status: str
    analysis: dict
    validation: dict
    profile: dict


# ── Training ────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    target_col: str
    task: Optional[str] = None
    scaler: str = Field("standard", description="'standard' | 'minmax' | 'none'")


class TrainResponse(BaseModel):
    status: str
    training_summary: dict
    leaderboard: list
    message: str


# ── Results ─────────────────────────────────────────────────────────

class ResultsResponse(BaseModel):
    status: str
    decision: dict
    leaderboard: list
    shap_global: dict
    calibration_scores: dict
    cv_results: dict


# ── Predict ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: list[float | int | str] = Field(
        ..., description="Feature values in the same order as training columns"
    )

class PredictResponse(BaseModel):
    status: str
    prediction: Any
    probabilities: Optional[list[float]] = None
    lime_explanation: Optional[dict] = None
