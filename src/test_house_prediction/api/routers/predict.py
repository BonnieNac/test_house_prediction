"""Prediction route: serves the pipeline saved by the training."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from test_house_prediction.api.schemas import PredictionRequest, PredictionResponse
from test_house_prediction.core.models.predict import load_model, predict

router = APIRouter(tags=["prediction"])


@router.post("/predict")
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Predict the target for the given rows."""
    try:
        model = load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        predictions = predict(pd.DataFrame(request.records), model)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"invalid input: {error}") from error
    return PredictionResponse(predictions=predictions.tolist())
