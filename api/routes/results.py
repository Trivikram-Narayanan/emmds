"""
EMMDS Results Routes
GET  /api/results         — Full pipeline results
GET  /api/results/summary — Compact summary card
POST /api/predict         — Single-instance prediction via best sklearn pipeline
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.request_response import ResultsResponse, PredictRequest, PredictResponse
from api.state import state

router = APIRouter()


@router.get("/results", response_model=ResultsResponse)
def get_results():
    if not state.has_result():
        raise HTTPException(
            status_code=404,
            detail="No results available. Run /api/train first.",
        )
    r = state.pipeline_result
    steps = r.get("steps", {})
    return ResultsResponse(
        status="success",
        decision=r.get("decision", {}),
        leaderboard=steps.get("leaderboard", []),
        shap_global=steps.get("shap_global", {}),
        calibration_scores=steps.get("calibration_scores", {}),
        cv_results=steps.get("cv_results", {}),
    )


@router.get("/results/summary")
def get_summary():
    if not state.has_result():
        raise HTTPException(status_code=404, detail="No results available.")
    decision    = state.pipeline_result.get("decision", {})
    leaderboard = state.pipeline_result["steps"].get("leaderboard", [])
    return {
        "best_model":       decision.get("best_model"),
        "task":             decision.get("task"),
        "trust_score":      decision.get("trust_score"),
        "trust_label":      decision.get("trust_label"),
        "primary_score":    decision.get("primary_score"),
        "top_features":     decision.get("top_features", []),
        "leaderboard_top3": leaderboard[:3],
    }


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict using the best sklearn pipeline stored from the last training run.
    The pipeline handles all preprocessing internally — pass raw feature values
    in the same order as feature_names from the training run.
    """
    if not state.has_result():
        raise HTTPException(
            status_code=404,
            detail="No trained model available. Run /api/train first.",
        )

    r              = state.pipeline_result
    best_name      = r["decision"].get("best_model")
    trained_models = r.get("_trained_models", {})
    feature_names  = r.get("_feature_names", [])
    le             = r.get("_label_encoder")
    X_train_raw    = r.get("_X_train_raw")

    if not trained_models or best_name not in trained_models:
        raise HTTPException(status_code=404, detail="Trained models not available.")

    model    = trained_models[best_name]
    features = request.features

    if len(features) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {len(feature_names)} features "
                f"({feature_names}), got {len(features)}."
            ),
        )

    try:
        # Build one-row DataFrame — sklearn pipeline handles preprocessing
        instance_df    = pd.DataFrame([features], columns=feature_names)
        prediction_raw = model.predict(instance_df)
        prediction     = prediction_raw[0]

        # Decode label for classification
        if le is not None:
            try:
                prediction = le.inverse_transform([int(prediction)])[0]
            except Exception:
                pass

        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(instance_df)[0].tolist()
            except Exception:
                pass

        # LIME local explanation (best-effort)
        lime_explanation = None
        try:
            if X_train_raw is not None:
                from src.explainability.lime_explainer import LIMEExplainer
                class_names = ([str(c) for c in le.classes_]
                               if le is not None else None)
                lime_exp = LIMEExplainer()
                lime_exp.fit(
                    X_train_raw.values,
                    feature_names=feature_names,
                    class_names=class_names,
                    task=r.get("task", "classification"),
                )
                lime_explanation = lime_exp.explain_instance(
                    np.array(features), model)
        except Exception:
            pass

        return PredictResponse(
            status="success",
            prediction=(
                prediction.item() if hasattr(prediction, "item") else prediction
            ),
            probabilities=probabilities,
            lime_explanation=lime_explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
