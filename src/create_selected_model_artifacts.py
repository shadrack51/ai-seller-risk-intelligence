from pathlib import Path
import json
import shutil

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# =========================================================
# 2. EXISTING SELECTED DELIVERY MODEL
# =========================================================
#
# IMPORTANT:
#
# This is the delivery Random Forest that was trained using
# the final 39-feature delivery schema.
#
# We do NOT retrain it here.
# =========================================================

SOURCE_DELIVERY_MODEL = (
    PROCESSED_DIR
    / "delivery_early_warning_model.joblib"
)


# =========================================================
# 3. REVIEW DATASET
# =========================================================

REVIEW_DATASET = (
    PROCESSED_DIR
    / "temporal_review_ml_dataset.csv"
)


# =========================================================
# 4. FINAL DEPLOYMENT ARTIFACTS
# =========================================================

DELIVERY_MODEL = (
    PROCESSED_DIR
    / "delivery_risk_random_forest.joblib"
)

REVIEW_MODEL = (
    PROCESSED_DIR
    / "review_risk_logistic_regression.joblib"
)

MODEL_METADATA = (
    PROCESSED_DIR
    / "selected_model_metadata.json"
)


# =========================================================
# 5. REVIEW TARGET
# =========================================================

REVIEW_TARGET = (
    "future_review_risk"
)


# =========================================================
# 6. REVIEW FEATURES
# =========================================================

REVIEW_FEATURES = [
    "recent_orders",
    "older_orders",
    "orders_change",
    "recent_units_sold",
    "older_units_sold",
    "units_change",
    "recent_revenue",
    "older_revenue",
    "revenue_change",
    "recent_average_item_price",
    "older_average_item_price",
    "recent_average_freight",
    "older_average_freight",
    "seller_history_orders",
    "seller_history_reviews",
    "recent_order_activity",
    "older_order_activity",
    "low_history_orders",
    "low_history_reviews",
    "sufficient_order_history",
    "sufficient_review_history",
    "recent_average_review_score",
    "older_average_review_score",
    "review_score_change",
    "recent_low_review_rate",
    "older_low_review_rate",
    "low_review_rate_change",
    "recent_review_count",
    "older_review_count",
    "review_count_change",
    "recent_low_review_count",
    "older_low_review_count",
    "recent_review_activity",
    "older_review_activity",
    "review_score_declining",
    "low_review_rate_increasing",
]


# =========================================================
# 7. ESTABLISHED TEMPORAL SPLIT
# =========================================================

TRAIN_END = pd.Timestamp(
    "2018-02-01"
)

VALIDATION_START = pd.Timestamp(
    "2018-03-01"
)

TEST_START = pd.Timestamp(
    "2018-06-01"
)

EXPECTED_TRAIN_ROWS = 6046
EXPECTED_VALIDATION_ROWS = 2272
EXPECTED_TEST_ROWS = 2348


# =========================================================
# 8. INITIAL VALIDATION
# =========================================================

if not SOURCE_DELIVERY_MODEL.exists():

    raise FileNotFoundError(
        f"Source delivery model not found:\n"
        f"{SOURCE_DELIVERY_MODEL}"
    )


if not REVIEW_DATASET.exists():

    raise FileNotFoundError(
        f"Review dataset not found:\n"
        f"{REVIEW_DATASET}"
    )


print("=" * 70)
print("CREATING SELECTED MODEL ARTIFACTS")
print("=" * 70)


# =========================================================
# 9. LOAD AND VERIFY SOURCE DELIVERY MODEL
# =========================================================

delivery_source_model = joblib.load(
    SOURCE_DELIVERY_MODEL
)


print(
    "\nSource delivery model:"
)

print(
    f"File: {SOURCE_DELIVERY_MODEL.name}"
)

print(
    f"Object type: "
    f"{type(delivery_source_model).__name__}"
)


if not hasattr(
    delivery_source_model,
    "predict_proba",
):

    raise TypeError(
        "Source delivery model does not "
        "support predict_proba()."
    )


# ---------------------------------------------------------
# Verify the trained delivery feature schema.
# ---------------------------------------------------------

if not hasattr(
    delivery_source_model,
    "feature_names_in_",
):

    raise AttributeError(
        "Source delivery model does not expose "
        "feature_names_in_."
    )


delivery_model_features = list(
    delivery_source_model.feature_names_in_
)


if len(
    delivery_model_features
) != 39:

    raise ValueError(
        "Expected 39 delivery model features, "
        f"found {len(delivery_model_features)}."
    )


print(
    f"Training feature count: "
    f"{len(delivery_model_features)}"
)

print(
    "Source delivery schema validation: PASS"
)


# =========================================================
# 10. SAVE THE CORRECT DELIVERY ARTIFACT
# =========================================================

shutil.copy2(
    SOURCE_DELIVERY_MODEL,
    DELIVERY_MODEL
)


print(
    "\nDelivery model saved to:"
)

print(
    DELIVERY_MODEL
)


# =========================================================
# 11. LOAD REVIEW DATASET
# =========================================================

review_df = pd.read_csv(
    REVIEW_DATASET
)


review_df["snapshot_date"] = pd.to_datetime(
    review_df["snapshot_date"]
)


print(
    "\nReview dataset loaded:"
)

print(
    f"Rows: {len(review_df):,}"
)

print(
    f"Columns: {len(review_df.columns):,}"
)


# =========================================================
# 12. VALIDATE REVIEW FEATURES
# =========================================================

missing_review_features = [
    feature
    for feature in REVIEW_FEATURES
    if feature not in review_df.columns
]


if missing_review_features:

    raise KeyError(
        "Missing review features:\n"
        f"{missing_review_features}"
    )


print(
    "Review feature validation: PASS"
)


# =========================================================
# 13. CREATE REVIEW TEMPORAL SPLIT
# =========================================================

train_df = review_df[
    review_df["snapshot_date"]
    <= TRAIN_END
].copy()


validation_df = review_df[
    (review_df["snapshot_date"]
     >= VALIDATION_START)
    &
    (review_df["snapshot_date"]
     <= pd.Timestamp("2018-05-01"))
].copy()


test_df = review_df[
    (review_df["snapshot_date"]
     >= TEST_START)
    &
    (review_df["snapshot_date"]
     <= pd.Timestamp("2018-08-01"))
].copy()


print(
    "\nReview temporal split"
)

print(
    "-" * 70
)

print(
    f"Train rows:       {len(train_df):,}"
)

print(
    f"Validation rows:  {len(validation_df):,}"
)

print(
    f"Test rows:        {len(test_df):,}"
)


# =========================================================
# 14. VALIDATE REVIEW SPLIT
# =========================================================

if len(train_df) != EXPECTED_TRAIN_ROWS:

    raise ValueError(
        f"Train split mismatch. "
        f"Expected {EXPECTED_TRAIN_ROWS:,}, "
        f"got {len(train_df):,}."
    )


if len(validation_df) != EXPECTED_VALIDATION_ROWS:

    raise ValueError(
        f"Validation split mismatch. "
        f"Expected {EXPECTED_VALIDATION_ROWS:,}, "
        f"got {len(validation_df):,}."
    )


if len(test_df) != EXPECTED_TEST_ROWS:

    raise ValueError(
        f"Test split mismatch. "
        f"Expected {EXPECTED_TEST_ROWS:,}, "
        f"got {len(test_df):,}."
    )


print(
    "Review split validation: PASS"
)


# =========================================================
# 15. PREPARE REVIEW TRAINING DATA
# =========================================================

X_train = train_df[
    REVIEW_FEATURES
]

y_train = train_df[
    REVIEW_TARGET
]


# =========================================================
# 16. BUILD REVIEW LOGISTIC REGRESSION
# =========================================================

review_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="liblinear",
                random_state=42,
            ),
        ),
    ]
)


# =========================================================
# 17. TRAIN REVIEW MODEL
# =========================================================

print(
    "\nTraining selected review Logistic Regression..."
)


review_model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# =========================================================
# 18. SAVE REVIEW MODEL
# =========================================================

joblib.dump(
    review_model,
    REVIEW_MODEL
)


print(
    "\nReview model saved to:"
)

print(
    REVIEW_MODEL
)


# =========================================================
# 19. MODEL METADATA
# =========================================================

metadata = {

    "delivery": {

        "target": (
            "future_delivery_risk"
        ),

        "model_type": (
            "RandomForestClassifier"
        ),

        "source_artifact": (
            SOURCE_DELIVERY_MODEL.name
        ),

        "deployment_artifact": (
            DELIVERY_MODEL.name
        ),

        "validation_threshold": 0.15,

        "test_roc_auc": 0.5075,

        "test_precision": 0.1325,

        "test_recall": 0.5780,

        "test_f1": 0.2156,

        "role": (
            "Monitoring and early-warning signal"
        ),

        "confidence": "Low",

        "human_review_required": True,
    },


    "review": {

        "target": (
            "future_review_risk"
        ),

        "model_type": (
            "LogisticRegression"
        ),

        "deployment_artifact": (
            REVIEW_MODEL.name
        ),

        "validation_threshold": 0.12,

        "test_roc_auc": 0.6555,

        "test_precision": 0.1225,

        "test_recall": 0.8348,

        "test_f1": 0.2137,

        "role": (
            "Screening and prioritization signal"
        ),

        "confidence": "Moderate-low",

        "human_review_required": True,
    },


    "temporal_evaluation": {

        "train_rows": 6046,

        "validation_rows": 2272,

        "test_rows": 2348,

        "test_used_for_training": False,

        "test_used_for_threshold_selection": False,
    },
}


# =========================================================
# 20. SAVE METADATA
# =========================================================

with open(
    MODEL_METADATA,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


print(
    "\nModel metadata saved to:"
)

print(
    MODEL_METADATA
)


# =========================================================
# 21. VERIFY SAVED DELIVERY ARTIFACT
# =========================================================

saved_delivery_model = joblib.load(
    DELIVERY_MODEL
)


if not hasattr(
    saved_delivery_model,
    "predict_proba",
):

    raise TypeError(
        "Saved delivery artifact does not "
        "support predict_proba()."
    )


saved_delivery_features = list(
    saved_delivery_model.feature_names_in_
)


if saved_delivery_features != delivery_model_features:

    raise ValueError(
        "Delivery artifact feature schema changed "
        "during saving."
    )


print(
    "\nDelivery artifact verification: PASS"
)


# =========================================================
# 22. VERIFY SAVED REVIEW ARTIFACT
# =========================================================

saved_review_model = joblib.load(
    REVIEW_MODEL
)


if not hasattr(
    saved_review_model,
    "predict_proba",
):

    raise TypeError(
        "Saved review artifact does not "
        "support predict_proba()."
    )


saved_review_features = list(
    saved_review_model.feature_names_in_
)


if saved_review_features != REVIEW_FEATURES:

    raise ValueError(
        "Review artifact feature schema does not "
        "match REVIEW_FEATURES."
    )


print(
    "Review artifact verification: PASS"
)


# =========================================================
# 23. FINAL SUMMARY
# =========================================================

print(
    "\n"
)

print(
    "=" * 70
)

print(
    "SELECTED MODEL ARTIFACTS COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nDeployment artifacts:"
)

print(
    f"Delivery: "
    f"{DELIVERY_MODEL.name}"
)

print(
    f"Review:   "
    f"{REVIEW_MODEL.name}"
)

print(
    f"Metadata: "
    f"{MODEL_METADATA.name}"
)

print(
    "\nCorrect delivery source:"
)

print(
    SOURCE_DELIVERY_MODEL.name
)

print(
    "\nThe delivery model was not retrained."
)

print(
    "The selected review model was trained using "
    "the established training population."
)

print(
    "Both saved artifacts passed schema verification."
)