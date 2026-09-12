import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


DELIVERY_MODEL_PATH = (
    PROCESSED_DATA_DIR / "delivery_risk_random_forest.joblib"
)

REVIEW_MODEL_PATH = (
    PROCESSED_DATA_DIR / "review_risk_logistic_regression.joblib"
)


DELIVERY_THRESHOLD = 0.15

REVIEW_THRESHOLD = 0.12


APP_ENV = os.getenv(
    "APP_ENV",
    "development",
)