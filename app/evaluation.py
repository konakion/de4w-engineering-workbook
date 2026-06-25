from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from app.training import train_random_forest


def classification_report_dict(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Return a classification report as a JSON-serializable dictionary.
    """
    return sklearn_classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """
    Return a confusion matrix as a labeled DataFrame.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """
    Compute accuracy, classification report and confusion matrix.
    """
    matrix = confusion_matrix_table(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report_dict(y_true, y_pred),
        "confusion_matrix": matrix.values.tolist(),
        "labels": labels,
    }


def evaluate_random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """
    Train and evaluate a Random Forest with a random train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    y_pred = model.predict(X_test)
    labels = [str(label) for label in model.classes_]
    metrics = evaluate_predictions(y_test, y_pred, labels=labels)

    return {
        "strategy": "random_split",
        "model": model,
        "metrics": metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }


def evaluate_subject_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """
    Train and evaluate with different subjects in train and test sets.
    """
    subjects = pd.Series(groups.unique()).sort_values()

    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=random_state,
    )

    train_mask = groups.isin(train_subjects)
    test_mask = groups.isin(test_subjects)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    y_pred = model.predict(X_test)
    labels = [str(label) for label in model.classes_]
    metrics = evaluate_predictions(y_test, y_pred, labels=labels)

    return {
        "strategy": "subject_split",
        "model": model,
        "metrics": metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_subjects": [int(subject) for subject in train_subjects],
        "test_subjects": [int(subject) for subject in test_subjects],
    }


def leave_one_subject_out_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    base_model: RandomForestClassifier | None = None,
    max_subjects: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate generalization by testing on one held-out subject at a time.

    Parameters
    ----------
    X:
        Feature matrix.

    y:
        Activity labels.

    groups:
        Subject identifiers.

    base_model:
        Optional estimator to clone for each fold. If omitted, a 200-tree
        Random Forest with random_state=42 is used.

    max_subjects:
        Optional cap for teaching/demo runs.

    Returns
    -------
    pandas.DataFrame
        One row per held-out subject with accuracy and test size.
    """
    if base_model is None:
        base_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )

    subjects = sorted(groups.unique())

    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    rows: list[dict[str, float | int]] = []

    for subject in subjects:
        test_mask = groups == subject
        train_mask = ~test_mask

        model = clone(base_model)
        model.fit(X.loc[train_mask], y.loc[train_mask])

        y_pred = model.predict(X.loc[test_mask])
        rows.append(
            {
                "subject_id": int(subject),
                "accuracy": float(accuracy_score(y.loc[test_mask], y_pred)),
                "test_size": int(test_mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def metrics_for_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """
    Strip model objects from an evaluation result before JSON export.
    """
    return {
        key: value
        for key, value in result.items()
        if key != "model"
    }
