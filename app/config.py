from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "activity_model_random_forest.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"
ACTIVITY_LABELS_PATH = MODEL_DIR / "activity_labels.joblib"

WINDOW_SIZE = 100
SAMPLING_RATE_HZ = 20
SAMPLE_INTERVAL_SECONDS = 1 / SAMPLING_RATE_HZ

DEFAULT_PHYPHOX_URL = "http://172.16.79.231:8080"