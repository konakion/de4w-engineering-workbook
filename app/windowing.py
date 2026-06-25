import pandas as pd

from app.config import WINDOW_SIZE


def create_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> list[pd.DataFrame]:
    """
    Split a sensor DataFrame into non-overlapping windows.

    Parameters
    ----------
    df:
        Sensor DataFrame sorted by timestamp.

    window_size:
        Number of samples per window.

    Returns
    -------
    list[pandas.DataFrame]
        List of sensor windows.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    windows: list[pd.DataFrame] = []

    for start in range(0, len(df) - window_size + 1, window_size):
        end = start + window_size
        windows.append(df.iloc[start:end])

    return windows


def create_grouped_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> list[pd.DataFrame]:
    """
    Create windows separately for each subject and activity.

    This avoids windows that accidentally mix subjects or activities.

    Parameters
    ----------
    df:
        Sensor DataFrame containing subject_id, activity and timestamp.

    window_size:
        Number of samples per window.

    Returns
    -------
    list[pandas.DataFrame]
        List of windows created per subject and activity.
    """
    all_windows: list[pd.DataFrame] = []

    for activity in sorted(df["activity"].unique()):
        activity_df = df[df["activity"] == activity]

        for subject_id in sorted(activity_df["subject_id"].unique()):
            subject_df = (
                activity_df[activity_df["subject_id"] == subject_id]
                .sort_values("timestamp")
            )

            windows = create_windows(
                subject_df,
                window_size=window_size,
            )

            all_windows.extend(windows)

    return all_windows
