from typing import Any

# FastAPI turns ordinary Python functions into HTTP endpoints so other
# programs can request predictions without importing our code directly.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model_utils import (
    FeatureMismatchError,
    ModelArtifactError,
    get_prediction_service,
    predict_from_features,
)


class PredictionRequest(BaseModel):
    """
    Feature payload for one activity prediction.
    """

    features: dict[str, float] = Field(
        ...,
        description="Feature dictionary produced by app.feature_extraction.extract_features.",
    )


class PredictionResponse(BaseModel):
    """
    Activity prediction returned by the API.
    """

    activity: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    """
    Health status for the prediction API.
    """

    status: str
    model_loaded: bool
    detail: str


app = FastAPI(
    title="DE4W Activity Recognition API",
    description="A small API for activity recognition from accelerometer features.",
    version="0.3.0",
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """
    Root endpoint mirroring /health.
    """
    return health()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Report whether model artifacts can be loaded.
    """
    service = get_prediction_service()

    try:
        service.load()
    except ModelArtifactError as exc:
        return HealthResponse(
            status="error",
            model_loaded=False,
            detail=str(exc),
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        detail="Model artifacts are available.",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, Any]:
    """
    Predict an activity from extracted accelerometer features.
    """
    try:
        return predict_from_features(request.features)
    except FeatureMismatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
