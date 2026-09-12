import joblib

from .config import (
    DELIVERY_MODEL_PATH,
    REVIEW_MODEL_PATH,
)


# =========================================================
# 1. LOAD REVIEW MODEL
# =========================================================

if not REVIEW_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Review model not found:\n"
        f"{REVIEW_MODEL_PATH}"
    )


review_model = joblib.load(
    REVIEW_MODEL_PATH
)


# =========================================================
# 2. LOAD DELIVERY MODEL
# =========================================================

if not DELIVERY_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Delivery model not found:\n"
        f"{DELIVERY_MODEL_PATH}"
    )


delivery_model = joblib.load(
    DELIVERY_MODEL_PATH
)


# =========================================================
# 3. MODEL VALIDATION
# =========================================================

if not hasattr(
    review_model,
    "predict_proba",
):
    raise TypeError(
        "Review model does not support "
        "predict_proba()."
    )


if not hasattr(
    delivery_model,
    "predict_proba",
):
    raise TypeError(
        "Delivery model does not support "
        "predict_proba()."
    )