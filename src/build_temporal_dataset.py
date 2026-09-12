"""
Build Temporal Seller Prediction Dataset

Purpose
-------
Create a forward-looking seller-level machine learning dataset.

Instead of predicting CURRENT seller risk, this dataset asks:

    "Using seller behavior from the previous 90 days,
     will the seller become at risk during the next 60 days?"

This creates a temporal / early-warning ML problem.

Observation window:
    90 days before the prediction cutoff.

Prediction window:
    60 days after the prediction cutoff.

Important:
    Features are calculated only from historical information
    available on or before the cutoff date.

The future window is used ONLY to construct the target.

Outputs
-------
data/processed/temporal_seller_dataset.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_seller_dataset.csv"
)


# ============================================================
# FILE PATHS
# ============================================================

ORDERS_FILE = (
    RAW_DATA_DIR
    / "olist_orders_dataset.csv"
)

ORDER_ITEMS_FILE = (
    RAW_DATA_DIR
    / "olist_order_items_dataset.csv"
)

REVIEWS_FILE = (
    RAW_DATA_DIR
    / "olist_order_reviews_dataset.csv"
)

SELLERS_FILE = (
    RAW_DATA_DIR
    / "olist_sellers_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

OBSERVATION_DAYS = 90

PREDICTION_DAYS = 60

# Minimum historical orders required to create a useful
# seller observation.

MIN_HISTORY_ORDERS = 2

# Minimum completed future orders required to evaluate
# future seller risk.

MIN_FUTURE_ORDERS = 2

# Business risk definitions.
#
# These are intentionally similar to our earlier risk logic,
# but they are now calculated ONLY from FUTURE outcomes.

FUTURE_LATE_RATE_THRESHOLD = 20.0

FUTURE_LOW_REVIEW_THRESHOLD = 3.5


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_mean(series):
    """Return mean or NaN when no values exist."""

    if series.empty:
        return np.nan

    return series.mean()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TEMPORAL SELLER PREDICTION DATASET")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section("LOADING RAW DATA")

    print("Loading orders...")

    orders = pd.read_csv(
        ORDERS_FILE
    )

    print(
        f"Orders: {len(orders):,}"
    )

    print("Loading order items...")

    order_items = pd.read_csv(
        ORDER_ITEMS_FILE
    )

    print(
        f"Order items: {len(order_items):,}"
    )

    print("Loading reviews...")

    reviews = pd.read_csv(
        REVIEWS_FILE
    )

    print(
        f"Reviews: {len(reviews):,}"
    )

    print("Loading sellers...")

    sellers = pd.read_csv(
        SELLERS_FILE
    )

    print(
        f"Sellers: {len(sellers):,}"
    )

    # ========================================================
    # 2. CONVERT DATE COLUMNS
    # ========================================================

    print_section(
        "PREPARING DATE COLUMNS"
    )

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:

        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce"
        )

    review_date_columns = [
        "review_creation_date",
        "review_answer_timestamp",
    ]

    for column in review_date_columns:

        reviews[column] = pd.to_datetime(
            reviews[column],
            errors="coerce"
        )

    print(
        "Order and review dates converted -> PASS"
    )

    # ========================================================
    # 3. DETERMINE DATA RANGE
    # ========================================================

    print_section(
        "DATA DATE RANGE"
    )

    min_date = (
        orders[
            "order_purchase_timestamp"
        ]
        .min()
    )

    max_date = (
        orders[
            "order_purchase_timestamp"
        ]
        .max()
    )

    print(
        f"Earliest purchase date: {min_date}"
    )

    print(
        f"Latest purchase date:   {max_date}"
    )

    # ========================================================
    # 4. PREPARE ORDER-SELLER DATA
    # ========================================================

    print_section(
        "BUILDING ORDER-SELLER HISTORY"
    )

    # Attach seller IDs to orders through order items.
    #
    # One order can contain multiple products/sellers.
    # We therefore create unique order-seller pairs.

    order_sellers = (
        order_items[
            [
                "order_id",
                "seller_id",
                "price",
                "freight_value",
            ]
        ]
        .drop_duplicates(
            subset=[
                "order_id",
                "seller_id",
            ]
        )
    )

    order_history = orders[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "order_status",
        ]
    ].merge(
        order_sellers,
        on="order_id",
        how="inner",
    )

    # ========================================================
    # 5. CALCULATE ORDER-LEVEL METRICS
    # ========================================================

    print(
        "Calculating order-level delivery metrics..."
    )

    order_history[
        "delivery_days"
    ] = (
        (
            order_history[
                "order_delivered_customer_date"
            ]
            - order_history[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_history[
        "estimated_delivery_days"
    ] = (
        (
            order_history[
                "order_estimated_delivery_date"
            ]
            - order_history[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_history[
        "delivery_delay_days"
    ] = (
        (
            order_history[
                "order_delivered_customer_date"
            ]
            - order_history[
                "order_estimated_delivery_date"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_history[
        "is_delivered"
    ] = (
        order_history[
            "order_delivered_customer_date"
        ]
        .notna()
    )

    order_history[
        "is_late"
    ] = (
        order_history[
            "delivery_delay_days"
        ]
        > 0
    )

    # ========================================================
    # 6. PREPARE REVIEW DATA
    # ========================================================

    print(
        "Preparing seller review history..."
    )

    reviews_clean = reviews[
        [
            "order_id",
            "review_score",
            "review_creation_date",
        ]
    ].copy()

    # Multiple reviews can exist for an order.
    #
    # We keep the available review records and aggregate
    # later by seller.

    review_orders = (
        order_history[
            [
                "order_id",
                "seller_id",
            ]
        ]
        .drop_duplicates(
            subset=[
                "order_id",
                "seller_id",
            ]
        )
        .merge(
            reviews_clean,
            on="order_id",
            how="left",
        )
    )

    # ========================================================
    # 7. CREATE SNAPSHOT DATES
    # ========================================================

    print_section(
        "CREATING TEMPORAL SNAPSHOTS"
    )

    # We create monthly prediction cutoffs.
    #
    # Each cutoff represents a point in time at which
    # the business could have made a prediction.

    first_snapshot = (
        min_date
        + pd.Timedelta(
            days=OBSERVATION_DAYS
        )
    )

    last_snapshot = (
        max_date
        - pd.Timedelta(
            days=PREDICTION_DAYS
        )
    )

    snapshot_dates = pd.date_range(
        start=first_snapshot.normalize(),
        end=last_snapshot.normalize(),
        freq="MS",
    )

    print(
        f"Observation window: "
        f"{OBSERVATION_DAYS} days"
    )

    print(
        f"Prediction window: "
        f"{PREDICTION_DAYS} days"
    )

    print(
        f"Snapshot dates: "
        f"{len(snapshot_dates)}"
    )

    print(
        f"First snapshot: "
        f"{snapshot_dates.min()}"
    )

    print(
        f"Last snapshot: "
        f"{snapshot_dates.max()}"
    )

    # ========================================================
    # 8. BUILD SNAPSHOTS
    # ========================================================

    print_section(
        "BUILDING SELLER SNAPSHOTS"
    )

    snapshot_rows = []

    for snapshot_date in snapshot_dates:

        observation_start = (
            snapshot_date
            - pd.Timedelta(
                days=OBSERVATION_DAYS
            )
        )

        observation_end = (
            snapshot_date
        )

        prediction_start = (
            snapshot_date
            + pd.Timedelta(
                days=1
            )
        )

        prediction_end = (
            snapshot_date
            + pd.Timedelta(
                days=PREDICTION_DAYS
            )
        )

        # ----------------------------------------------------
        # HISTORICAL OBSERVATION WINDOW
        # ----------------------------------------------------

        historical_orders = order_history[
            (
                order_history[
                    "order_purchase_timestamp"
                ]
                > observation_start
            )
            &
            (
                order_history[
                    "order_purchase_timestamp"
                ]
                <= observation_end
            )
        ].copy()

        # Require completed delivery outcomes to have actually
        # occurred before the prediction date.

        historical_delivered = (
            historical_orders[
                historical_orders[
                    "is_delivered"
                ]
                &
                (
                    historical_orders[
                        "order_delivered_customer_date"
                    ]
                    <= observation_end
                )
            ]
        )

        # ----------------------------------------------------
        # HISTORICAL REVIEWS
        # ----------------------------------------------------

        historical_reviews = (
            review_orders[
                (
                    review_orders[
                        "review_creation_date"
                    ]
                    > observation_start
                )
                &
                (
                    review_orders[
                        "review_creation_date"
                    ]
                    <= observation_end
                )
            ]
        )

        # ----------------------------------------------------
        # GROUP HISTORICAL FEATURES BY SELLER
        # ----------------------------------------------------

        if historical_orders.empty:

            continue

        history_grouped = (
            historical_orders
            .groupby("seller_id")
            .agg(
                total_orders=(
                    "order_id",
                    "nunique"
                ),
                total_revenue=(
                    "price",
                    "sum"
                ),
                average_item_price=(
                    "price",
                    "mean"
                ),
                average_freight=(
                    "freight_value",
                    "mean"
                ),
            )
            .reset_index()
        )

        history_grouped[
            "units_sold"
        ] = (
            historical_orders
            .groupby("seller_id")
            .size()
            .values
        )

        delivered_grouped = (
            historical_delivered
            .groupby("seller_id")
            .agg(
                delivered_orders=(
                    "order_id",
                    "nunique"
                ),
                average_delivery_days=(
                    "delivery_days",
                    "mean"
                ),
                average_estimated_delivery_days=(
                    "estimated_delivery_days",
                    "mean"
                ),
                average_delivery_delay_days=(
                    "delivery_delay_days",
                    "mean"
                ),
            )
            .reset_index()
        )

        history_grouped = (
            history_grouped
            .merge(
                delivered_grouped,
                on="seller_id",
                how="left",
            )
        )

        # ----------------------------------------------------
        # REVIEW FEATURES
        # ----------------------------------------------------

        if historical_reviews.empty:

            review_grouped = pd.DataFrame(
                columns=[
                    "seller_id",
                    "review_count",
                ]
            )

        else:

            review_grouped = (
                historical_reviews
                .groupby("seller_id")
                .agg(
                    review_count=(
                        "review_score",
                        "count"
                    ),
                )
                .reset_index()
            )

        history_grouped = (
            history_grouped
            .merge(
                review_grouped,
                on="seller_id",
                how="left",
            )
        )

        # ----------------------------------------------------
        # CREATE TARGET FROM FUTURE WINDOW
        # ----------------------------------------------------

        future_orders = order_history[
            (
                order_history[
                    "order_purchase_timestamp"
                ]
                > prediction_start
            )
            &
            (
                order_history[
                    "order_purchase_timestamp"
                ]
                <= prediction_end
            )
        ].copy()

        future_delivered = (
            future_orders[
                future_orders[
                    "is_delivered"
                ]
            ]
        )

        future_reviews = (
            review_orders[
                (
                    review_orders[
                        "review_creation_date"
                    ]
                    > prediction_start
                )
                &
                (
                    review_orders[
                        "review_creation_date"
                    ]
                    <= prediction_end
                )
            ]
        )

        if future_orders.empty:

            continue

        future_order_counts = (
            future_orders
            .groupby("seller_id")
            .agg(
                future_order_count=(
                    "order_id",
                    "nunique"
                )
            )
            .reset_index()
        )

        future_delivery = (
            future_delivered
            .groupby("seller_id")
            .agg(
                future_delivered_orders=(
                    "order_id",
                    "nunique"
                ),
                future_late_orders=(
                    "is_late",
                    "sum"
                ),
            )
            .reset_index()
        )

        future_data = (
            future_order_counts
            .merge(
                future_delivery,
                on="seller_id",
                how="left",
            )
        )

        future_data[
            "future_late_delivery_rate"
        ] = (
            future_data[
                "future_late_orders"
            ]
            /
            future_data[
                "future_delivered_orders"
            ]
            * 100
        )

        if future_reviews.empty:

            future_review_grouped = pd.DataFrame(
                columns=[
                    "seller_id",
                    "future_review_count",
                    "future_average_review_score",
                ]
            )

        else:

            future_review_grouped = (
                future_reviews
                .groupby("seller_id")
                .agg(
                    future_review_count=(
                        "review_score",
                        "count"
                    ),
                    future_average_review_score=(
                        "review_score",
                        "mean"
                    ),
                )
                .reset_index()
            )

        future_data = (
            future_data
            .merge(
                future_review_grouped,
                on="seller_id",
                how="left",
            )
        )

        # ----------------------------------------------------
        # FUTURE TARGET
        # ----------------------------------------------------

        # We require a minimum number of future orders so
        # that a seller with one exceptional order does not
        # automatically become a "future-risk" example.

        future_data = future_data[
            future_data[
                "future_order_count"
            ]
            >= MIN_FUTURE_ORDERS
        ].copy()

        # Risk is defined by FUTURE behavior.

        future_data[
            "future_late_risk"
        ] = (
            future_data[
                "future_late_delivery_rate"
            ]
            >= FUTURE_LATE_RATE_THRESHOLD
        )

        future_data[
            "future_review_risk"
        ] = (
            future_data[
                "future_average_review_score"
            ]
            < FUTURE_LOW_REVIEW_THRESHOLD
        )

        future_data[
            "future_at_risk"
        ] = (
            future_data[
                "future_late_risk"
            ]
            |
            future_data[
                "future_review_risk"
            ]
        ).astype(int)

        # ----------------------------------------------------
        # COMBINE HISTORICAL FEATURES WITH FUTURE TARGET
        # ----------------------------------------------------

        snapshot = (
            history_grouped
            .merge(
                future_data[
                    [
                        "seller_id",
                        "future_order_count",
                        "future_delivered_orders",
                        "future_late_delivery_rate",
                        "future_average_review_score",
                        "future_at_risk",
                    ]
                ],
                on="seller_id",
                how="inner",
            )
        )

        # ----------------------------------------------------
        # MINIMUM HISTORY REQUIREMENT
        # ----------------------------------------------------

        snapshot = snapshot[
            snapshot[
                "total_orders"
            ]
            >= MIN_HISTORY_ORDERS
        ].copy()

        if snapshot.empty:

            continue

        # ----------------------------------------------------
        # ADD TEMPORAL METADATA
        # ----------------------------------------------------

        snapshot[
            "snapshot_date"
        ] = snapshot_date

        snapshot[
            "observation_start"
        ] = observation_start

        snapshot[
            "prediction_start"
        ] = prediction_start

        snapshot[
            "prediction_end"
        ] = prediction_end

        snapshot_rows.append(
            snapshot
        )

        print(
            f"{snapshot_date.date()} | "
            f"historical sellers: "
            f"{len(history_grouped):,} | "
            f"labeled sellers: "
            f"{len(snapshot):,}"
        )

    # ========================================================
    # 9. COMBINE ALL SNAPSHOTS
    # ========================================================

    print_section(
        "COMBINING TEMPORAL SNAPSHOTS"
    )

    if not snapshot_rows:

        raise ValueError(
            "No temporal snapshots were created. "
            "Check the observation/prediction windows."
        )

    temporal_df = pd.concat(
        snapshot_rows,
        ignore_index=True
    )

    print(
        f"Temporal rows created: "
        f"{len(temporal_df):,}"
    )

    print(
        f"Unique sellers represented: "
        f"{temporal_df['seller_id'].nunique():,}"
    )

    print(
        f"Snapshots represented: "
        f"{temporal_df['snapshot_date'].nunique():,}"
    )

    # ========================================================
    # 10. HANDLE NUMERICAL CLEANUP
    # ========================================================

    print_section(
        "TEMPORAL DATA QUALITY"
    )

    numeric_columns = [
        "total_orders",
        "total_revenue",
        "average_item_price",
        "average_freight",
        "units_sold",
        "delivered_orders",
        "average_delivery_days",
        "average_estimated_delivery_days",
        "average_delivery_delay_days",
        "review_count",
    ]

    for column in numeric_columns:

        temporal_df[column] = pd.to_numeric(
            temporal_df[column],
            errors="coerce"
        )

    print(
        "Numeric feature conversion -> PASS"
    )

    # ========================================================
    # 11. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "FUTURE TARGET DISTRIBUTION"
    )

    target_counts = (
        temporal_df[
            "future_at_risk"
        ]
        .value_counts()
        .sort_index()
    )

    target_percentages = (
        temporal_df[
            "future_at_risk"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(
        "Future target counts:"
    )

    print(
        target_counts
    )

    print(
        "\nFuture target percentages:"
    )

    print(
        target_percentages
    )

    # ========================================================
    # 12. TEMPORAL LEAKAGE CHECK
    # ========================================================

    print_section(
        "TEMPORAL LEAKAGE CHECK"
    )

    leakage_checks = []

    leakage_checks.append(
        (
            temporal_df[
                "observation_start"
            ]
            <
            temporal_df[
                "snapshot_date"
            ]
        ).all()
    )

    leakage_checks.append(
        (
            temporal_df[
                "prediction_start"
            ]
            >
            temporal_df[
                "snapshot_date"
            ]
        ).all()
    )

    leakage_checks.append(
        (
            temporal_df[
                "prediction_end"
            ]
            >=
            temporal_df[
                "prediction_start"
            ]
        ).all()
    )

    if all(leakage_checks):

        print(
            "Temporal window ordering -> PASS"
        )

    else:

        raise ValueError(
            "Temporal window leakage detected."
        )

    # ========================================================
    # 13. SNAPSHOT DUPLICATE CHECK
    # ========================================================

    print_section(
        "TEMPORAL RECORD VALIDATION"
    )

    duplicate_snapshot_rows = (
        temporal_df[
            [
                "seller_id",
                "snapshot_date",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate seller/snapshot rows: "
        f"{duplicate_snapshot_rows}"
    )

    if duplicate_snapshot_rows > 0:

        raise ValueError(
            "Duplicate seller/snapshot records found."
        )

    print(
        "Seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 14. DISPLAY SAMPLE
    # ========================================================

    print_section(
        "TEMPORAL DATASET PREVIEW"
    )

    preview_columns = [
        "seller_id",
        "snapshot_date",
        "total_orders",
        "total_revenue",
        "delivered_orders",
        "review_count",
        "future_late_delivery_rate",
        "future_average_review_score",
        "future_at_risk",
    ]

    preview = temporal_df[
        preview_columns
    ].head(10).copy()

    print(
        preview.to_string(
            index=False
        )
    )

    # ========================================================
    # 15. SAVE DATASET
    # ========================================================

    print_section(
        "SAVING TEMPORAL DATASET"
    )

    temporal_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Temporal seller dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL DATASET COMPLETE"
    )

    print(
        f"Temporal observations: "
        f"{len(temporal_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{temporal_df['seller_id'].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{temporal_df['snapshot_date'].nunique():,}"
    )

    print(
        f"Observation window: "
        f"{OBSERVATION_DAYS} days"
    )

    print(
        f"Prediction window: "
        f"{PREDICTION_DAYS} days"
    )

    print(
        "\nThe dataset now represents a forward-looking "
        "seller risk prediction problem."
    )

    print(
        "\nImportant:"
    )

    print(
        "Historical features are constructed from the "
        "observation window."
    )

    print(
        "The target is constructed only from the future "
        "prediction window."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()