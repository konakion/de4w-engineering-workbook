from fastapi import FastAPI

from app.model_utils import predict_from_features


app = FastAPI(
    title="DE4W Activity Recognition API",
    description="A small API for activity recognition from accelerometer features.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "message": "DE4W Activity Recognition API is running."
    }


@app.post("/predict")
def predict(features: dict[str, float]) -> dict:
    """
    Predict an activity from extracted accelerometer features.

    Parameters
    ----------
    features:
        Dictionary containing all model features.

    Returns
    -------
    dict
        Predicted activity, confidence and class probabilities.
    """
    return predict_from_features(features)