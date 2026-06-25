from datetime import datetime
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

from app.config import (
    DATA_DIR,
    DEFAULT_PHYPHOX_URL,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    SAMPLE_INTERVAL_SECONDS,
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
)
from app.data_loader import load_wisdm_phone_accel
from app.feature_extraction import extract_features
from app.live_buffer import LiveBuffer
from app.model_utils import FeatureMismatchError, ModelArtifactError, predict_from_features
from app.phyphox_client import PhyphoxClient, PhyphoxClientError
from app.pipeline import add_magnitude


ACTIVITY_ICONS = {
    "walking": "🚶",
    "sitting": "🪑",
    "standing": "🧍",
}

ACTIVITY_ORDER = ["walking", "standing", "sitting"]

NETWORK_HINT = (
    "Check the IP address, keep phyphox remote access enabled, and prefer a "
    "phone hotspot or local router over eduroam because eduroam may block "
    "device-to-device traffic."
)


st.set_page_config(
    page_title="DE4W Live Activity Recognition",
    page_icon="📱",
    layout="wide",
)


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "history": [],
        "logs": [],
        "latest_result": None,
        "latest_features": None,
        "latest_window": None,
        "latest_error": None,
        "samples_seen": 0,
        "replay_index": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(message: str, level: str = "info") -> None:
    st.session_state.logs.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        },
    )
    st.session_state.logs = st.session_state.logs[:30]


@st.cache_data(show_spinner=False)
def load_replay_samples() -> pd.DataFrame:
    try:
        replay_df = load_wisdm_phone_accel(DATA_DIR)
        replay_df = add_magnitude(replay_df)
        replay_df = replay_df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
        return replay_df[["x", "y", "z", "magnitude", "activity"]].head(6000)
    except Exception:
        t = np.linspace(0, 40, 4000)
        synthetic = pd.DataFrame(
            {
                "x": np.sin(t) * 2.0,
                "y": np.cos(t * 0.5) * 0.8,
                "z": 9.81 + np.sin(t * 1.5) * 1.2,
                "activity": "synthetic replay",
            }
        )
        synthetic["magnitude"] = np.sqrt(
            synthetic["x"] ** 2 + synthetic["y"] ** 2 + synthetic["z"] ** 2
        )
        return synthetic


def check_api(api_url: str) -> tuple[str, str]:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=1.0)
        if response.ok:
            payload = response.json()
            if payload.get("model_loaded"):
                return "success", "API connected"
            return "warning", "API reachable, model not loaded"
        return "error", f"API returned HTTP {response.status_code}"
    except requests.RequestException:
        return "warning", "API not connected"


def check_model_artifacts() -> tuple[str, str]:
    missing = [
        path.name
        for path in [MODEL_PATH, FEATURE_COLUMNS_PATH]
        if not path.exists()
    ]
    if missing:
        return "error", f"Missing: {', '.join(missing)}"
    return "success", "Model artifacts found"


def render_status_box(label: str, state: str, message: str) -> None:
    if state == "success":
        st.success(f"✅ {label}: {message}")
    elif state == "warning":
        st.warning(f"⚠️ {label}: {message}")
    else:
        st.error(f"❌ {label}: {message}")


def render_prediction_card(result: dict[str, Any] | None) -> None:
    if result is None:
        st.markdown(
            """
            <div class="prediction-card muted">
              <div class="prediction-icon">⏳</div>
              <div class="prediction-label">Waiting for Prediction</div>
              <div class="prediction-confidence">Collect a full Window first</div>
              <div class="prediction-help">The Prediction is the class with the highest Probability.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    activity = str(result["activity"])
    confidence = float(result["confidence"])
    icon = ACTIVITY_ICONS.get(activity, "❔")
    st.markdown(
        f"""
        <div class="prediction-card">
          <div class="prediction-icon">{icon}</div>
          <div class="prediction-label">{activity.title()}</div>
          <div class="prediction-confidence">{confidence:.0%} Confidence</div>
          <div class="prediction-help">The Prediction is the class with the highest Probability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probabilities(result: dict[str, Any] | None) -> None:
    if result is None:
        st.info("No probabilities yet.")
        return

    probabilities = result["probabilities"]
    ordered_labels = [
        label
        for label in ACTIVITY_ORDER
        if label in probabilities
    ] + [
        label
        for label in probabilities
        if label not in ACTIVITY_ORDER
    ]

    for label in ordered_labels:
        probability = float(probabilities[label])
        icon = ACTIVITY_ICONS.get(label, "•")
        st.progress(probability, text=f"{icon} {label.title()}: {probability:.0%}")


def compact_feature_table(features: dict[str, float]) -> pd.DataFrame:
    preferred = [
        "mean_x",
        "mean_y",
        "mean_z",
        "std_x",
        "std_y",
        "std_z",
        "min_magnitude",
        "max_magnitude",
        "energy_magnitude",
    ]
    rows = [
        {"feature": key, "value": features[key]}
        for key in preferred
        if key in features
    ]
    return pd.DataFrame(rows)


def append_prediction(result: dict[str, Any]) -> None:
    st.session_state.history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "prediction": result["activity"],
            "confidence": f"{float(result['confidence']):.0%}",
            "probabilities": ", ".join(
                f"{label}: {float(probability):.0%}"
                for label, probability in result["probabilities"].items()
            ),
        },
    )
    st.session_state.history = st.session_state.history[:20]


def get_replay_sample(replay_df: pd.DataFrame) -> tuple[float, float, float]:
    if replay_df.empty:
        raise RuntimeError("Replay data is empty.")

    index = st.session_state.replay_index % len(replay_df)
    row = replay_df.iloc[index]
    st.session_state.replay_index += 1
    return float(row["x"]), float(row["y"]), float(row["z"])


def render_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        .subtitle { color: #5b6472; font-size: 1.1rem; margin-top: -0.6rem; }
        .prediction-card {
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 1.4rem;
            text-align: center;
            background: #ffffff;
            box-shadow: 0 1px 8px rgba(20, 30, 50, 0.08);
        }
        .prediction-card.muted { background: #f7f8fb; }
        .prediction-icon { font-size: 3.2rem; line-height: 1; }
        .prediction-label { font-size: 2.2rem; font-weight: 700; margin-top: 0.4rem; }
        .prediction-confidence { font-size: 1.2rem; color: #2457a6; margin-top: 0.3rem; }
        .prediction-help { color: #687386; margin-top: 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_explanation() -> None:
    with st.expander("How does this work?"):
        st.markdown(
            """
            ```text
            Sensor Stream
            ↓
            Window
            ↓
            Feature Vector
            ↓
            Random Forest
            ↓
            Class Probabilities
            ↓
            Prediction
            ```

            The phone produces a Sensor Stream. The dashboard stores recent samples
            until a full Window is available. The Window becomes a Feature Vector.
            The Random Forest computes Class Probabilities, and the Prediction is
            the class with the highest Probability.
            """
        )


def run_demo(
    source_mode: str,
    phone_url: str,
    window_size: int,
    duration_seconds: int,
    refresh_interval: float,
) -> None:
    buffer = LiveBuffer(window_size=window_size)
    client = PhyphoxClient(phone_url) if source_mode == "Live phyphox" else None
    replay_df = load_replay_samples() if source_mode == "Replay Mode" else pd.DataFrame()
    start_time = time.time()

    status_area = st.empty()
    progress_area = st.empty()
    main_area = st.empty()

    if source_mode == "Replay Mode":
        status_area.warning("Replay mode active. Using stored WISDM/synthetic samples, not a live phone.")
        add_log("Replay mode started.", "warning")
    else:
        status_area.info("Collecting live samples from phyphox...")
        add_log("Live phyphox collection started.", "info")

    while time.time() - start_time < duration_seconds:
        try:
            if client is not None:
                x, y, z = client.get_acceleration()
            else:
                x, y, z = get_replay_sample(replay_df)

            buffer.add_sample(x, y, z)
            st.session_state.samples_seen += 1
            progress_area.progress(
                min(buffer.sample_count() / window_size, 1.0),
                text=f"Buffer: {buffer.sample_count()} / {window_size} samples",
            )

            if not buffer.is_ready():
                status_area.info("Collecting samples. Waiting for enough data to fill one Window.")
            else:
                status_area.info("Running inference on the current Window...")
                window = buffer.get_window()
                features = extract_features(window)
                result = predict_from_features(features)

                st.session_state.latest_window = window
                st.session_state.latest_features = features
                st.session_state.latest_result = result
                st.session_state.latest_error = None
                append_prediction(result)
                status_area.success("Prediction updated.")

            with main_area.container():
                render_main_dashboard(
                    show_probabilities=st.session_state.show_probabilities,
                    show_signal_plot=st.session_state.show_signal_plot,
                    show_feature_table=st.session_state.show_feature_table,
                    debug_mode=st.session_state.mode == "Debug Mode",
                )

            time.sleep(refresh_interval)

        except PhyphoxClientError as exc:
            st.session_state.latest_error = f"{exc} {NETWORK_HINT}"
            add_log(st.session_state.latest_error, "error")
            status_area.error(st.session_state.latest_error)
            break
        except (ModelArtifactError, FeatureMismatchError) as exc:
            st.session_state.latest_error = str(exc)
            add_log(st.session_state.latest_error, "error")
            status_area.error(st.session_state.latest_error)
            break
        except Exception as exc:
            st.session_state.latest_error = f"Unexpected dashboard error: {exc}"
            add_log(st.session_state.latest_error, "error")
            status_area.error(st.session_state.latest_error)
            break

    add_log("Demo finished.", "success")
    status_area.success("Finished demo.")


def render_main_dashboard(
    show_probabilities: bool,
    show_signal_plot: bool,
    show_feature_table: bool,
    debug_mode: bool,
) -> None:
    prediction_col, probability_col = st.columns([1.1, 1])

    with prediction_col:
        st.subheader("🧠 Current Prediction")
        render_prediction_card(st.session_state.latest_result)

    with probability_col:
        if show_probabilities:
            st.subheader("📊 Class Probabilities")
            render_probabilities(st.session_state.latest_result)

    if show_signal_plot:
        st.subheader("📈 Live Signal")
        window = st.session_state.latest_window
        if window is None:
            st.info("Signal plot appears after the first full Window.")
        else:
            st.line_chart(window[["x", "y", "z", "magnitude"]])

    if show_feature_table:
        st.subheader("Feature Vector Preview")
        features = st.session_state.latest_features
        if features is None:
            st.info("Feature table appears after the first Prediction.")
        elif debug_mode:
            st.dataframe(
                pd.DataFrame(
                    sorted(features.items()),
                    columns=["feature", "value"],
                ),
                use_container_width=True,
                height=300,
            )
        else:
            st.dataframe(
                compact_feature_table(features),
                use_container_width=True,
                height=260,
            )

    st.subheader("Prediction History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("No Predictions yet.")

    if debug_mode:
        st.subheader("Debug Details")
        debug_cols = st.columns(3)
        debug_cols[0].metric("Samples seen", st.session_state.samples_seen)
        latest_window = st.session_state.latest_window
        debug_cols[1].metric("Window rows", 0 if latest_window is None else len(latest_window))
        debug_cols[2].metric("History rows", len(st.session_state.history))

        if st.session_state.latest_result is not None:
            st.json(st.session_state.latest_result)

        if st.session_state.logs:
            st.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True)


initialize_state()
render_css()

st.title("DE4W Live Activity Recognition")
st.markdown('<div class="subtitle">End-to-End Wearable AI Demo</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Control Center")
    st.session_state.mode = st.radio(
        "Mode",
        ["Teaching Mode", "Debug Mode"],
        horizontal=False,
    )
    source_mode = st.radio(
        "Sensor source",
        ["Live phyphox", "Replay Mode"],
        horizontal=False,
    )
    phone_url = st.text_input("phyphox URL", value=DEFAULT_PHYPHOX_URL)
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
    sampling_rate = st.number_input(
        "Sampling rate [Hz]",
        min_value=1,
        max_value=100,
        value=SAMPLING_RATE_HZ,
        step=1,
    )
    window_size = st.number_input(
        "Window size [samples]",
        min_value=20,
        max_value=500,
        value=WINDOW_SIZE,
        step=10,
    )
    refresh_interval = st.number_input(
        "Refresh interval [s]",
        min_value=0.05,
        max_value=2.0,
        value=float(SAMPLE_INTERVAL_SECONDS),
        step=0.05,
    )
    duration = st.number_input(
        "Demo duration [s]",
        min_value=5,
        max_value=300,
        value=60,
        step=5,
    )
    st.session_state.show_probabilities = st.checkbox("Show probabilities", value=True)
    st.session_state.show_feature_table = st.checkbox(
        "Show feature table",
        value=st.session_state.mode == "Debug Mode",
    )
    st.session_state.show_signal_plot = st.checkbox("Show signal plot", value=True)
    st.session_state.show_debug_details = st.checkbox(
        "Show debug details",
        value=st.session_state.mode == "Debug Mode",
    )

    if st.button("Reset history"):
        st.session_state.history = []
        st.session_state.logs = []
        st.session_state.latest_result = None
        st.session_state.latest_features = None
        st.session_state.latest_window = None
        st.session_state.samples_seen = 0
        add_log("History reset.", "info")

    start = st.button("Start demo", type="primary", use_container_width=True)

api_state, api_message = check_api(api_url)
model_state, model_message = check_model_artifacts()
phyphox_state = "warning" if source_mode == "Live phyphox" else "success"
phyphox_message = "Ready to test live URL" if source_mode == "Live phyphox" else "Replay data selected"
buffer_state = "success" if st.session_state.latest_window is not None else "warning"
buffer_message = (
    f"Window ready ({len(st.session_state.latest_window)} samples)"
    if st.session_state.latest_window is not None
    else "Waiting for samples"
)

status_cols = st.columns(4)
with status_cols[0]:
    render_status_box("API", api_state, api_message)
with status_cols[1]:
    render_status_box("Sensor", phyphox_state, phyphox_message)
with status_cols[2]:
    render_status_box("Model", model_state, model_message)
with status_cols[3]:
    render_status_box("Buffer", buffer_state, buffer_message)

if source_mode == "Replay Mode":
    st.warning("Replay mode active. This demonstrates the pipeline without a live phone.")

if st.session_state.latest_error:
    st.error(st.session_state.latest_error)

render_main_dashboard(
    show_probabilities=st.session_state.show_probabilities,
    show_signal_plot=st.session_state.show_signal_plot,
    show_feature_table=st.session_state.show_feature_table,
    debug_mode=st.session_state.mode == "Debug Mode" or st.session_state.show_debug_details,
)

render_explanation()

if start:
    run_demo(
        source_mode=source_mode,
        phone_url=phone_url,
        window_size=int(window_size),
        duration_seconds=int(duration),
        refresh_interval=float(refresh_interval),
    )
