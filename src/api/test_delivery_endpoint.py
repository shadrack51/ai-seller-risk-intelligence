import json
from collections import Counter
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen

import joblib
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

DELIVERY_DATASET = (
    PROCESSED_DIR
    / "temporal_delivery_ml_dataset.csv"
)

DELIVERY_MODEL = (
    PROCESSED_DIR
    / "delivery_risk_random_forest.joblib"
)


# =========================================================
# 2. API CONFIGURATION
# =========================================================

API_URL = (
    "http://127.0.0.1:8000/predict/delivery"
)


# =========================================================
# 3. DELIVERY MODEL FEATURES
# =========================================================

DELIVERY_FEATURES = [
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
    "recent_delivered_orders",
    "older_delivered_orders",
    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",
    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",
    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",
    "recent_late_orders",
    "older_late_orders",
    "recent_delivery_activity",
    "older_delivery_activity",
    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",
]


# =========================================================
# 4. VALIDATE FEATURE LIST
# =========================================================

feature_counts = Counter(
    DELIVERY_FEATURES
)

duplicate_features = [
    feature
    for feature, count
    in feature_counts.items()
    if count > 1
]


if duplicate_features:

    raise ValueError(
        "\nDuplicate features found in "
        "DELIVERY_FEATURES:\n"
        f"{duplicate_features}"
    )


if len(DELIVERY_FEATURES) != 39:

    raise ValueError(
        "\nExpected 39 delivery features, "
        f"but found {len(DELIVERY_FEATURES)}."
    )


print("=" * 70)
print("DELIVERY API END-TO-END TEST")
print("=" * 70)


print(
    f"\nDelivery feature count: "
    f"{len(DELIVERY_FEATURES)}"
)

print(
    "Feature-list uniqueness: PASS"
)


# =========================================================
# 5. CHECK REQUIRED FILES
# =========================================================

if not DELIVERY_DATASET.exists():

    raise FileNotFoundError(
        f"\nDelivery dataset not found:\n"
        f"{DELIVERY_DATASET}"
    )


if not DELIVERY_MODEL.exists():

    raise FileNotFoundError(
        f"\nDelivery model not found:\n"
        f"{DELIVERY_MODEL}"
    )


# =========================================================
# 6. LOAD DELIVERY DATASET
# =========================================================

df = pd.read_csv(
    DELIVERY_DATASET
)


print(
    "\nDataset loaded:"
)

print(
    f"Rows:    {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)


# =========================================================
# 7. CHECK DATASET COLUMN NAMES
# =========================================================

duplicate_columns = (
    df.columns[
        df.columns.duplicated()
    ]
    .tolist()
)


if duplicate_columns:

    raise ValueError(
        "\nThe dataset contains duplicate "
        "column names:\n"
        f"{duplicate_columns}"
    )


print(
    "Dataset column-name uniqueness: PASS"
)


# =========================================================
# 8. CHECK REQUIRED FEATURES
# =========================================================

missing_features = [
    feature
    for feature in DELIVERY_FEATURES
    if feature not in df.columns
]


if missing_features:

    raise KeyError(
        "\nRequired delivery features are missing:\n"
        f"{missing_features}"
    )


print(
    "Required feature presence: PASS"
)


# =========================================================
# 9. LOAD SAVED DELIVERY MODEL
# =========================================================

model = joblib.load(
    DELIVERY_MODEL
)


print(
    "\nModel loaded:"
)

print(
    f"Object type: "
    f"{type(model).__name__}"
)


if not hasattr(
    model,
    "predict_proba",
):

    raise TypeError(
        "Loaded delivery model does not support "
        "predict_proba()."
    )


print(
    "predict_proba(): PASS"
)


# =========================================================
# 10. SELECT A REAL DATASET OBSERVATION
# =========================================================

complete_rows = df.dropna(
    subset=DELIVERY_FEATURES
)


if complete_rows.empty:

    raise ValueError(
        "No complete delivery-risk observation "
        "was found."
    )


test_row = (
    complete_rows.iloc[0]
)


print(
    "\nSelected real dataset observation:"
)


if "seller_id" in test_row.index:

    print(
        f"Seller ID: "
        f"{test_row['seller_id']}"
    )


if "snapshot_date" in test_row.index:

    print(
        f"Snapshot date: "
        f"{test_row['snapshot_date']}"
    )


# =========================================================
# 11. BUILD MODEL INPUT
# =========================================================

input_values = []

for feature in DELIVERY_FEATURES:

    input_values.append(
        test_row[feature]
    )


input_df = pd.DataFrame(
    [input_values],
    columns=DELIVERY_FEATURES,
)


# =========================================================
# 12. VALIDATE INPUT SCHEMA
# =========================================================

if input_df.shape != (
    1,
    len(DELIVERY_FEATURES),
):

    raise ValueError(
        "\nUnexpected input shape.\n"
        f"Expected: "
        f"(1, {len(DELIVERY_FEATURES)})\n"
        f"Actual:   {input_df.shape}"
    )


input_duplicate_columns = (
    input_df.columns[
        input_df.columns.duplicated()
    ]
    .tolist()
)


if input_duplicate_columns:

    raise ValueError(
        "\nInput contains duplicate column names:\n"
        f"{input_duplicate_columns}"
    )


if list(input_df.columns) != DELIVERY_FEATURES:

    raise ValueError(
        "\nInput feature order does not match "
        "DELIVERY_FEATURES."
    )


print(
    "\nInput schema validation:"
)

print(
    "  Shape: PASS"
)

print(
    "  Unique columns: PASS"
)

print(
    "  Feature order: PASS"
)


# =========================================================
# 13. DIRECT MODEL PREDICTION
# =========================================================

direct_probability = float(
    model.predict_proba(
        input_df
    )[0, 1]
)


print(
    "\nDirect model prediction:"
)

print(
    f"Risk score: "
    f"{direct_probability:.10f}"
)


# =========================================================
# 14. CREATE JSON REQUEST
# =========================================================

request_payload = {
    feature: float(
        test_row[feature]
    )

    for feature in DELIVERY_FEATURES
}


request_body = json.dumps(
    request_payload
).encode(
    "utf-8"
)


# =========================================================
# 15. SEND REQUEST TO FASTAPI
# =========================================================

request = Request(
    API_URL,

    data=request_body,

    headers={
        "Content-Type": "application/json"
    },

    method="POST",
)


print(
    "\nSending request to:"
)

print(
    API_URL
)


try:

    with urlopen(
        request,
        timeout=10,
    ) as response:

        response_body = (
            response
            .read()
            .decode("utf-8")
        )

        api_result = json.loads(
            response_body
        )


except Exception as exc:

    raise RuntimeError(
        "\nCould not reach the FastAPI endpoint.\n\n"
        "Make sure the server is running with:\n"
        "python -m uvicorn "
        "src.api.main:app --reload\n\n"
        f"Original error: {exc}"
    ) from exc


# =========================================================
# 16. DISPLAY API RESPONSE
# =========================================================

print(
    "\nAPI response:"
)

print(
    json.dumps(
        api_result,
        indent=4,
    )
)


# =========================================================
# 17. VERIFY RESPONSE STRUCTURE
# =========================================================

required_response_fields = [
    "risk_score",
    "monitoring_threshold",
    "monitor_delivery",
    "model",
    "target",
]


missing_response_fields = [
    field
    for field in required_response_fields
    if field not in api_result
]


if missing_response_fields:

    raise AssertionError(
        "\nAPI response is missing required fields:\n"
        f"{missing_response_fields}"
    )


print(
    "\nResponse schema: PASS"
)


# =========================================================
# 18. COMPARE DIRECT MODEL AND API
# =========================================================

api_probability = float(
    api_result["risk_score"]
)


difference = abs(
    api_probability
    - round(
        direct_probability,
        4,
    )
)


print(
    "\nPrediction comparison:"
)

print(
    "-" * 50
)

print(
    f"Direct model score: "
    f"{direct_probability:.10f}"
)

print(
    f"API returned score: "
    f"{api_probability:.10f}"
)

print(
    f"Difference after API rounding: "
    f"{difference:.10f}"
)


# =========================================================
# 19. VERIFY PREDICTION CONSISTENCY
# =========================================================

if difference > 0.00005:

    raise AssertionError(
        "\nAPI prediction does not match "
        "the direct model prediction."
    )


print(
    "Prediction consistency: PASS"
)


# =========================================================
# 20. FINAL RESULT
# =========================================================

print(
    "\n"
)

print(
    "=" * 70
)

print(
    "DELIVERY API END-TO-END TEST PASSED"
)

print(
    "=" * 70
)

print(
    """
A real seller/snapshot observation was:

1. Loaded from the temporal delivery dataset.
2. Passed through the saved Random Forest model.
3. Sent through the FastAPI delivery endpoint.
4. Returned as a JSON prediction.
5. Compared against the direct model prediction.

The API and direct model predictions are consistent.
"""
)