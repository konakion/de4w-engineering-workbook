from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATA_DIR, WINDOW_SIZE
from app.data_loader import load_wisdm_phone_accel
from app.feature_extraction import extract_features
from app.windowing import create_grouped_windows


def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the acceleration magnitude column to a sensor DataFrame.

    Parameters
    ----------
    df:
        Sensor DataFrame containing x, y and z acceleration columns.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with an added magnitude column.
    """
    result = df.copy()

    result["magnitude"] = np.sqrt(
        result["x"] ** 2 + result["y"] ** 2 + result["z"] ** 2
    )

    return result


def windows_to_features(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Convert sensor data into one feature row per subject/activity window.

    Windows are created separately per subject and activity so that no window
    mixes labels or participants.

    Parameters
    ----------
    df:
        Sensor DataFrame containing subject_id, activity, timestamp, x, y, z
        and magnitude columns.

    window_size:
        Number of samples per window.

    Returns
    -------
    pandas.DataFrame
        Feature table with one row per window.
    """
    if "magnitude" not in df.columns:
        df = add_magnitude(df)

    rows: list[dict[str, float | int | str]] = []

    for window in create_grouped_windows(df, window_size=window_size):
        feature_row: dict[str, float | int | str] = dict(extract_features(window))
        feature_row["activity"] = str(window["activity"].iloc[0])
        feature_row["subject_id"] = int(window["subject_id"].iloc[0])
        rows.append(feature_row)

    return pd.DataFrame(rows)


def build_feature_table(
    data_dir: Path = DATA_DIR,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Build the complete feature table from raw WISDM phone accelerometer files.

    Parameters
    ----------
    data_dir:
        Directory containing data_*_accel_phone.txt files.

    window_size:
        Number of samples per window.

    Returns
    -------
    pandas.DataFrame
        Feature table ready for machine learning.
    """
    df = load_wisdm_phone_accel(data_dir)
    df = add_magnitude(df)

    return windows_to_features(df, window_size=window_size)
