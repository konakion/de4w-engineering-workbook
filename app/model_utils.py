import joblib
import pandas as pd

from app.config import FEATURE_COLUMNS_PATH, MODEL_PATH


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


def predict_from_features(features: dict[str, float]) -> dict:
    """
    Predict the activity from a feature dictionary.

    Parameters
    ----------
    features:
        Dictionary containing exactly the features used during training.

    Returns
    -------
    dict
        Prediction result containing predicted activity, confidence,
        and class probabilities.
    """
    features_df = pd.DataFrame([features])
    features_df = features_df[feature_columns]

    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    return {
        "activity": prediction,
        "confidence": float(probabilities.max()),
        "probabilities": class_probabilities,
    }