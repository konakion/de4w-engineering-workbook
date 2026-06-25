import time

import pandas as pd
import streamlit as st

from app.config import (
    DEFAULT_PHYPHOX_URL,
    SAMPLE_INTERVAL_SECONDS,
    WINDOW_SIZE,
)
from app.feature_extraction import extract_features
from app.live_buffer import LiveBuffer
from app.model_utils import predict_from_features
from app.phyphox_client import PhyphoxClient


st.set_page_config(
    page_title="DE4W Live Activity Recognition",
    layout="wide",
)

st.title("DE4W Live Activity Recognition")

st.write(
    "This dashboard reads live accelerometer data from phyphox, "
    "builds a sensor window, extracts features, and predicts the current activity."
)

phone_url = st.text_input(
    "phyphox URL",
    value=DEFAULT_PHYPHOX_URL,
)

window_size = st.number_input(
    "Window size",
    min_value=20,
    max_value=500,
    value=WINDOW_SIZE,
    step=10,
)

duration = st.number_input(
    "Demo duration [s]",
    min_value=10,
    max_value=300,
    value=60,
    step=10,
)

start = st.button("Start live demo")

status_placeholder = st.empty()
prediction_placeholder = st.empty()
probability_placeholder = st.empty()
signal_placeholder = st.empty()

if start:
    client = PhyphoxClient(phone_url)
    buffer = LiveBuffer(window_size=window_size)

    start_time = time.time()

    while time.time() - start_time < duration:
        x, y, z = client.get_acceleration()
        buffer.add_sample(x, y, z)

        status_placeholder.write(
            f"Collected samples: {len(buffer.x_buffer)} / {window_size}"
        )

        if buffer.is_ready():
            window = buffer.get_window()
            features = extract_features(window)
            result = predict_from_features(features)

            prediction_placeholder.metric(
                "Current Activity",
                result["activity"],
                f'{result["confidence"]:.2f} confidence',
            )

            probabilities_df = pd.DataFrame(
                list(result["probabilities"].items()),
                columns=["activity", "probability"],
            )

            probability_placeholder.bar_chart(
                probabilities_df,
                x="activity",
                y="probability",
            )

            signal_placeholder.line_chart(
                window[["x", "y", "z", "magnitude"]]
            )

        time.sleep(SAMPLE_INTERVAL_SECONDS)

    status_placeholder.success("Demo finished.")