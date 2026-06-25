from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import FEATURE_COLUMNS_PATH, MODEL_PATH


class ModelArtifactError(RuntimeError):
    """
    Raised when a required model artifact is missing or cannot be loaded.
    """


class FeatureMismatchError(ValueError):
    """
    Raised when prediction features do not match the trained model.
    """


class PredictionService:
    """
    Load model artifacts lazily and run activity predictions.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    ) -> None:
        self.model_path = model_path
        self.feature_columns_path = feature_columns_path
        self.model: Any | None = None
        self.feature_columns: list[str] | None = None

    def load(self) -> None:
        """
        Load the trained model and feature column order from disk.
        """
        if not self.model_path.exists():
            raise ModelArtifactError(
                f"Missing model artifact: {self.model_path}. "
                "Train/export the model before starting prediction."
            )

        if not self.feature_columns_path.exists():
            raise ModelArtifactError(
                f"Missing feature columns artifact: {self.feature_columns_path}. "
                "Export feature_columns.joblib together with the model."
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelArtifactError(
                f"Could not load model artifact {self.model_path}: {exc}"
            ) from exc

        try:
            self.feature_columns = list(joblib.load(self.feature_columns_path))
        except Exception as exc:
            raise ModelArtifactError(
                "Could not load feature column artifact "
                f"{self.feature_columns_path}: {exc}"
            ) from exc

    def _ensure_loaded(self) -> None:
        if self.model is None or self.feature_columns is None:
            self.load()

    def validate_features(self, features: dict[str, float]) -> None:
        """
        Ensure the incoming feature dictionary matches the trained model.
        """
        self._ensure_loaded()

        assert self.feature_columns is not None

        expected = set(self.feature_columns)
        received = set(features)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)

        if missing or unexpected:
            message_parts = []

            if missing:
                message_parts.append(f"missing features: {missing}")

            if unexpected:
                message_parts.append(f"unexpected features: {unexpected}")

            raise FeatureMismatchError(
                "Feature mismatch for activity model; "
                + "; ".join(message_parts)
            )

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Predict the activity from a feature dictionary.
        """
        self.validate_features(features)

        assert self.model is not None
        assert self.feature_columns is not None

        features_df = pd.DataFrame([features])
        features_df = features_df[self.feature_columns]

        prediction = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]

        class_probabilities = {
            str(class_name): float(probability)
            for class_name, probability in zip(self.model.classes_, probabilities)
        }

        return {
            "activity": str(prediction),
            "confidence": float(probabilities.max()),
            "probabilities": class_probabilities,
        }


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """
    Return the shared lazy prediction service used by the API and dashboard.
    """
    return PredictionService()


def predict_from_features(features: dict[str, float]) -> dict[str, Any]:
    """
    Predict an activity with the shared prediction service.
    """
    return get_prediction_service().predict(features)
