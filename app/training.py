import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.config import (
    ACTIVITY_LABELS_PATH,
    DATA_DIR,
    FEATURE_COLUMNS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    WINDOW_SIZE,
)


def load_feature_table(
    features_path: Path = DATA_DIR / "features_df.csv",
) -> pd.DataFrame:
    """
    Load the saved feature table used for model training.

    Parameters
    ----------
    features_path:
        Path to the CSV file created by the feature engineering pipeline.

    Returns
    -------
    pandas.DataFrame
        Feature table containing feature columns, activity and subject_id.
    """
    return pd.read_csv(features_path)


def split_features_and_labels(
    features_df: pd.DataFrame,
    label_column: str = "activity",
    group_column: str = "subject_id",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Split a feature table into model inputs, labels, groups and feature names.

    Parameters
    ----------
    features_df:
        Feature table containing numeric features, labels and subject IDs.

    label_column:
        Name of the target label column.

    group_column:
        Name of the subject/group identifier column.

    Returns
    -------
    tuple
        X, y, groups and the ordered feature column names.
    """
    feature_columns = [
        column
        for column in features_df.columns
        if column not in [label_column, group_column]
    ]

    X = features_df[feature_columns]
    y = features_df[label_column]
    groups = features_df[group_column]

    return X, y, groups, feature_columns


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier for activity recognition.

    Parameters
    ----------
    X:
        Feature matrix.

    y:
        Activity labels.

    n_estimators:
        Number of trees in the forest.

    random_state:
        Seed for reproducible training.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X, y)

    return model


def save_model_artifacts(
    model: RandomForestClassifier,
    feature_columns: list[str],
    output_dir: Path = MODEL_PATH.parent,
    metadata: dict[str, Any] | None = None,
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    """
    Save the trained model, feature order, class labels and metadata.

    Parameters
    ----------
    model:
        Trained Random Forest classifier.

    feature_columns:
        Ordered feature columns expected by the model.

    output_dir:
        Directory where model artifacts are written.

    metadata:
        Additional JSON-serializable metadata, such as evaluation results.

    window_size:
        Window size used to create the feature table.

    Returns
    -------
    dict
        Metadata written to disk.
    """
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / MODEL_PATH.name
    feature_columns_path = output_dir / FEATURE_COLUMNS_PATH.name
    activity_labels_path = output_dir / ACTIVITY_LABELS_PATH.name
    metadata_path = output_dir / MODEL_METADATA_PATH.name

    class_labels = [str(class_label) for class_label in model.classes_]

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, feature_columns_path)
    joblib.dump(class_labels, activity_labels_path)

    model_metadata: dict[str, Any] = {
        "model_type": type(model).__name__,
        "window_size": window_size,
        "feature_columns": feature_columns,
        "class_labels": class_labels,
        "n_estimators": int(model.n_estimators),
        "random_state": model.random_state,
    }

    if metadata is not None:
        model_metadata.update(metadata)

    metadata_path.write_text(
        json.dumps(model_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return model_metadata


def load_model_artifacts(
    model_path: Path = MODEL_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    activity_labels_path: Path = ACTIVITY_LABELS_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[RandomForestClassifier, list[str], list[str], dict[str, Any]]:
    """
    Load the trained model, feature order, class labels and metadata.

    Parameters
    ----------
    model_path:
        Path to the saved model artifact.

    feature_columns_path:
        Path to the saved feature column order.

    activity_labels_path:
        Path to the saved class labels.

    metadata_path:
        Path to the saved metadata JSON file.

    Returns
    -------
    tuple
        Model, feature columns, activity labels and metadata.
    """
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_columns_path)
    activity_labels = joblib.load(activity_labels_path)

    metadata: dict[str, Any] = {}

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return model, feature_columns, activity_labels, metadata
