from collections import deque

import numpy as np
import pandas as pd

from app.config import WINDOW_SIZE


class LiveBuffer:
    """
    Fixed-size buffer for live accelerometer samples.

    The buffer stores the most recent x, y, z values and returns them
    as a pandas DataFrame once enough samples are available.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = window_size

        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        self.z_buffer = deque(maxlen=window_size)

    def add_sample(self, x: float, y: float, z: float) -> None:
        """
        Add one accelerometer sample to the buffer.
        """
        self.x_buffer.append(float(x))
        self.y_buffer.append(float(y))
        self.z_buffer.append(float(z))

    def is_ready(self) -> bool:
        """
        Check whether the buffer contains enough samples for prediction.
        """
        return len(self.x_buffer) == self.window_size

    def get_window(self) -> pd.DataFrame:
        """
        Return the current buffer content as a sensor window.

        Returns
        -------
        pandas.DataFrame
            Window with x, y, z and magnitude columns.
        """
        window = pd.DataFrame(
            {
                "x": list(self.x_buffer),
                "y": list(self.y_buffer),
                "z": list(self.z_buffer),
            }
        )

        window["magnitude"] = np.sqrt(
            window["x"] ** 2 + window["y"] ** 2 + window["z"] ** 2
        )

        return window