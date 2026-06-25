from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
MODEL_DIR: Path = PROJECT_ROOT / "models"

MODEL_PATH: Path = MODEL_DIR / "activity_model_random_forest.joblib"
FEATURE_COLUMNS_PATH: Path = MODEL_DIR / "feature_columns.joblib"
ACTIVITY_LABELS_PATH: Path = MODEL_DIR / "activity_labels.joblib"

WINDOW_SIZE: int = 100
SAMPLING_RATE_HZ: int = 20
SAMPLE_INTERVAL_SECONDS: float = 1 / SAMPLING_RATE_HZ

DEFAULT_PHYPHOX_URL: str = "http://172.16.79.231:8080"
