"""
Seller Intelligence API

Purpose
-------
Expose historical temporal seller behavior for the frontend.

The temporal prediction grain is:

    seller_id + snapshot_date

This endpoint does NOT perform model prediction.

It provides historical seller intelligence that can be
visualized by the frontend alongside the existing seller
risk assessment.

Important
---------
The future target column is intentionally excluded from the
API response because it represents the observed outcome used
during model development rather than information available
to a business user at assessment time.
"""

from pathlib import Path

import pandas as pd

from fastapi import APIRouter, HTTPException


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TEMPORAL_FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_model_features.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_at_risk"


# ============================================================
# FEATURES EXPOSED TO THE BUSINESS FRONTEND
# ============================================================

INTELLIGENCE_FEATURES = [
    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    "recent_orders",
    "older_orders",
    "orders_change",

    "recent_units_sold",
    "older_units_sold",
    "units_change",

    "recent_order_activity",
    "older_order_activity",

    # --------------------------------------------------------
    # Commercial
    # --------------------------------------------------------

    "recent_revenue",
    "older_revenue",
    "revenue_change",

    "recent_average_item_price",
    "older_average_item_price",

    "recent_average_freight",
    "older_average_freight",

    # --------------------------------------------------------
    # Delivery
    # --------------------------------------------------------

    "recent_delivered_orders",
    "older_delivered_orders",

    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",

    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",

    "recent_late_orders",
    "older_late_orders",

    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",

    "recent_delivery_activity",
    "older_delivery_activity",

    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",

    # --------------------------------------------------------
    # Reviews
    # --------------------------------------------------------

    "recent_review_count",
    "older_review_count",
    "review_count_change",

    "recent_average_review_score",
    "older_average_review_score",
    "review_score_change",

    "recent_low_review_count",
    "older_low_review_count",

    "recent_low_review_rate",
    "older_low_review_rate",
    "low_review_rate_change",

    "recent_review_activity",
    "older_review_activity",

    "review_score_declining",
    "low_review_rate_increasing",

    # --------------------------------------------------------
    # Seller history
    # --------------------------------------------------------

    "seller_history_orders",
    "seller_history_reviews",

    "low_history_orders",
    "low_history_reviews",

    "sufficient_order_history",
    "sufficient_review_history",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_temporal_features():
    """
    Load and validate the curated temporal seller dataset.
    """

    if not TEMPORAL_FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Temporal model feature dataset not found:\n"
            f"{TEMPORAL_FEATURE_FILE}"
        )

    df = pd.read_csv(
        TEMPORAL_FEATURE_FILE
    )

    required_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
        *INTELLIGENCE_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Temporal feature dataset is missing "
            "required columns:\n"
            + "\n".join(missing_columns)
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    if df[DATE_COLUMN].isna().any():
        raise ValueError(
            "Invalid snapshot dates found in "
            "temporal feature dataset."
        )

    duplicate_keys = (
        df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_keys > 0:
        raise ValueError(
            "Duplicate seller/snapshot records found: "
            f"{duplicate_keys}"
        )

    return df


# ============================================================
# ROUTE
# ============================================================

@router.get(
    "/intelligence/seller/{seller_id}",
)
def get_seller_intelligence(
    seller_id: str,
):
    """
    Return historical temporal intelligence for one seller.
    """

    df = load_temporal_features()

    seller_df = df[
        df[SELLER_COLUMN]
        .astype(str)
        .eq(str(seller_id))
    ].copy()

    if seller_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Seller not found in the temporal "
                "feature dataset."
            ),
        )

    seller_df = seller_df.sort_values(
        DATE_COLUMN
    )

    response_rows = []

    for _, row in seller_df.iterrows():

        snapshot = {
            "snapshot_date": (
                row[DATE_COLUMN]
                .strftime("%Y-%m-%d")
            ),
        }

        for feature in INTELLIGENCE_FEATURES:

            value = row[feature]

            if pd.isna(value):
                snapshot[feature] = None
            else:
                snapshot[feature] = float(value)

        response_rows.append(
            snapshot
        )

    return {
        "seller_id": str(seller_id),
        "snapshot_count": len(response_rows),
        "snapshots": response_rows,
    }