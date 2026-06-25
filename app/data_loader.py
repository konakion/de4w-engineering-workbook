from collections.abc import Sequence
from pathlib import Path

import pandas as pd


ACTIVITY_MAP: dict[str, str] = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "brushing_teeth",
    "H": "eating_soup",
    "I": "eating_chips",
    "J": "eating_pasta",
    "K": "drinking",
    "L": "eating_sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding_clothes",
}

TARGET_ACTIVITIES: list[str] = ["walking", "sitting", "standing"]


def load_single_wisdm_file(file_path: Path) -> pd.DataFrame:
    """
    Load one WISDM phone accelerometer file.

    Each row has the raw format:
    subject_id, activity_code, timestamp, x, y, z;

    Parameters
    ----------
    file_path:
        Path to one WISDM raw text file.

    Returns
    -------
    pandas.DataFrame
        Loaded sensor data with readable activity names.
    """
    df = pd.read_csv(
        file_path,
        header=None,
        names=["subject_id", "activity_code", "timestamp", "x", "y", "z"],
        sep=",",
    )

    df["z"] = (
        df["z"]
        .astype(str)
        .str.replace(";", "", regex=False)
        .astype(float)
    )

    df["activity"] = df["activity_code"].map(ACTIVITY_MAP)

    return df


def load_wisdm_phone_accel(
    data_dir: Path,
    target_activities: Sequence[str] | None = TARGET_ACTIVITIES,
) -> pd.DataFrame:
    """
    Load all WISDM phone accelerometer files from a directory.

    Parameters
    ----------
    data_dir:
        Directory containing data_*_accel_phone.txt files.

    target_activities:
        Activity names to keep. If None, all activities are kept.

    Returns
    -------
    pandas.DataFrame
        Combined sensor data from all participant files.
    """
    files = sorted(data_dir.glob("data_*_accel_phone.txt"))

    if not files:
        raise FileNotFoundError(
            f"No WISDM phone accelerometer files found in {data_dir}"
        )

    dataframes = []

    for file_path in files:
        df_file = load_single_wisdm_file(file_path)

        if target_activities is not None:
            df_file = df_file[df_file["activity"].isin(target_activities)]

        dataframes.append(df_file)

    return pd.concat(dataframes, ignore_index=True)
