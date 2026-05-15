"""
EMMDS Analyze Route
POST /api/analyze
Runs DataAnalyzer, DataProfiler, DataValidator on loaded dataset.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.request_response import AnalyzeRequest, AnalyzeResponse
from api.state import state

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_dataset(request: AnalyzeRequest):
    """
    Analyze the loaded dataset.
    Returns task type, feature profile, missing values, validation report.
    """
    if not state.has_data():
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload first.")

    df = state.df
    target_col = request.target_col

    if target_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        )

    # Validation
    from src.data_engine.validator import DataValidator
    validator = DataValidator()
    validation = validator.validate(df, target_col)

    # Analysis
    from src.data_engine.analyzer import DataAnalyzer
    analyzer = DataAnalyzer()
    analysis = analyzer.analyze(df, target_col)

    # Profile
    from src.data_engine.profiler import DataProfiler
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(df, target_col)

    # Store in state
    state.target_col = target_col
    state.task = request.task or analysis["task"]

    # Strip non-serializable parts from profile (correlation matrix can be large)
    profile_safe = {
        k: v for k, v in profile.items()
        if k not in ("correlation_matrix",)
    }

    return AnalyzeResponse(
        status="success",
        analysis=analysis,
        validation=validation,
        profile=profile_safe,
    )
