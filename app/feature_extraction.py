import numpy as np
import pandas as pd


FEATURE_AXES: tuple[str, ...] = ("x", "y", "z", "magnitude")


def extract_features(window: pd.DataFrame) -> dict[str, float]:
    """
    Extract statistical features from one accelerometer window.

    The input window must contain the columns:
    x, y, z, magnitude.

    Parameters
    ----------
    window:
        Sensor window containing accelerometer values.

    Returns
    -------
    dict
        Feature dictionary used by the trained model.
    """
    features: dict[str, float] = {}

    for axis in FEATURE_AXES:
        signal = window[axis]

        features[f"mean_{axis}"] = float(signal.mean())
        features[f"std_{axis}"] = float(signal.std())
        features[f"min_{axis}"] = float(signal.min())
        features[f"max_{axis}"] = float(signal.max())
        features[f"energy_{axis}"] = float(np.sum(signal**2))

    return features
