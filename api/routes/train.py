"""
EMMDS Train Route
POST /api/train
Runs the full EMMDS pipeline: preprocess → train → evaluate → explain → decide.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.request_response import TrainRequest, TrainResponse
from api.state import state

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
def train_models(request: TrainRequest):
    """
    Run the full EMMDS pipeline on the loaded dataset.
    This is the main endpoint — it triggers the entire system.
    """
    if not state.has_data():
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload first.")

    df = state.df
    target_col = request.target_col or state.target_col

    if not target_col:
        raise HTTPException(status_code=422, detail="target_col is required.")

    if target_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{target_col}' not found."
        )

    try:
        from src.pipeline.pipeline import EMPipeline

        pipeline = EMPipeline()
        result = pipeline.run(
            df=df,
            target_col=target_col,
            task=request.task or state.task,
            scaler=request.scaler,
        )

        if result.get("status") == "error" or "error" in result:
            raise HTTPException(status_code=422, detail=result.get("error", "Pipeline failed."))

        # Store result in state for results endpoint
        state.pipeline_result = result
        state.target_col = target_col

        training_summary = result["steps"].get("training", {})
        leaderboard = result["steps"].get("leaderboard", [])

        # Remove non-serializable model objects
        leaderboard_clean = [
            {k: v for k, v in row.items() if k != "model_object"}
            for row in leaderboard
        ]

        return TrainResponse(
            status="success",
            training_summary=training_summary,
            leaderboard=leaderboard_clean,
            message=(
                f"Training complete. Best model: "
                f"{result['decision'].get('best_model', 'N/A')} | "
                f"Trust: {result['decision'].get('trust_score', 'N/A')}"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
