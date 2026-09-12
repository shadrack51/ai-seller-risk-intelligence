"""
Temporal Trend Feature Engineering

Purpose
-------
Create recency and trend-based seller features for the
forward-looking seller-risk prediction problem.

Instead of relying mainly on 90-day averages, this script
compares:

    Recent window: 30 days
    Older baseline: previous 60 days

The goal is to identify deterioration or improvement in:

    - order activity
    - revenue
    - delivery performance
    - freight
    - customer reviews

The future target is NOT used to create predictors.

The existing temporal target and metadata are retained so the
result can be used for later model training.

Outputs
-------
data/processed/temporal_trend_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

ORDERS_FILE = (
    RAW_DATA_DIR / "olist_orders_dataset.csv"
)

ORDER_ITEMS_FILE = (
    RAW_DATA_DIR / "olist_order_items_dataset.csv"
)

REVIEWS_FILE = (
    RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
)

TEMPORAL_DATASET_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_seller_dataset.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_trend_features.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATE_COLUMN = "snapshot_date"

SELLER_COLUMN = "seller_id"

TARGET_COLUMN = "future_at_risk"

RECENT_DAYS = 30

OLDER_DAYS = 60


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_percentage_change(
    recent,
    previous,
):
    """
    Calculate percentage change safely.

    If the previous value is zero, return NaN rather than
    producing an infinite value.
    """

    if pd.isna(previous):
        return np.nan

    if previous == 0:
        return np.nan

    return (
        (recent - previous)
        / abs(previous)
        * 100
    )


def safe_difference(
    recent,
    previous,
):
    """
    Calculate recent minus previous.
    """

    if pd.isna(recent) or pd.isna(previous):
        return np.nan

    return recent - previous


def build_order_seller_history(
    orders,
    order_items,
):
    """
    Build a seller-level order history table.

    One row represents one order-seller relationship.

    Price and freight are summed at seller/order level so that
    each seller receives the financial amount associated with
    that order.
    """

    order_items_agg = (
        order_items
        .groupby(
            [
                "order_id",
                "seller_id",
            ],
            as_index=False,
        )
        .agg(
            order_revenue=(
                "price",
                "sum",
            ),
            freight_value=(
                "freight_value",
                "sum",
            ),
            units_sold=(
                "product_id",
                "count",
            ),
        )
    )

    order_seller = (
        orders[
            [
                "order_id",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
        ]
        .merge(
            order_items_agg,
            on="order_id",
            how="inner",
        )
    )

    # --------------------------------------------------------
    # Order-level delivery metrics
    # --------------------------------------------------------

    order_seller[
        "delivery_days"
    ] = (
        (
            order_seller[
                "order_delivered_customer_date"
            ]
            - order_seller[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_seller[
        "estimated_delivery_days"
    ] = (
        (
            order_seller[
                "order_estimated_delivery_date"
            ]
            - order_seller[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_seller[
        "delivery_delay_days"
    ] = (
        (
            order_seller[
                "order_delivered_customer_date"
            ]
            - order_seller[
                "order_estimated_delivery_date"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_seller[
        "is_delivered"
    ] = (
        order_seller[
            "order_delivered_customer_date"
        ]
        .notna()
    )

    order_seller[
        "is_late"
    ] = (
        order_seller[
            "delivery_delay_days"
        ]
        > 0
    )

    return order_seller


def aggregate_order_window(
    order_seller,
    start_date,
    end_date,
):
    """
    Aggregate seller behavior within a date window.
    """

    window = order_seller[
        (
            order_seller[
                "order_purchase_timestamp"
            ]
            > start_date
        )
        &
        (
            order_seller[
                "order_purchase_timestamp"
            ]
            <= end_date
        )
    ].copy()

    if window.empty:
        return pd.DataFrame()

    grouped = (
        window
        .groupby(SELLER_COLUMN)
        .agg(
            orders=(
                "order_id",
                "nunique",
            ),
            units_sold=(
                "units_sold",
                "sum",
            ),
            revenue=(
                "order_revenue",
                "sum",
            ),
            average_item_price=(
                "order_revenue",
                "mean",
            ),
            average_freight=(
                "freight_value",
                "mean",
            ),
            delivered_orders=(
                "is_delivered",
                "sum",
            ),
            average_delivery_days=(
                "delivery_days",
                "mean",
            ),
            average_estimated_delivery_days=(
                "estimated_delivery_days",
                "mean",
            ),
            average_delivery_delay_days=(
                "delivery_delay_days",
                "mean",
            ),
            late_orders=(
                "is_late",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped[
        "late_delivery_rate"
    ] = np.where(
        grouped["delivered_orders"] > 0,
        (
            grouped["late_orders"]
            / grouped["delivered_orders"]
            * 100
        ),
        np.nan,
    )

    return grouped


def aggregate_review_window(
    review_orders,
    start_date,
    end_date,
):
    """
    Aggregate seller review behavior within a date window.

    Reviews are grouped using review_creation_date because that
    represents when customer feedback became available.
    """

    window = review_orders[
        (
            review_orders[
                "review_creation_date"
            ]
            > start_date
        )
        &
        (
            review_orders[
                "review_creation_date"
            ]
            <= end_date
        )
    ].copy()

    if window.empty:
        return pd.DataFrame()

    grouped = (
        window
        .groupby(SELLER_COLUMN)
        .agg(
            review_count=(
                "review_score",
                "count",
            ),
            average_review_score=(
                "review_score",
                "mean",
            ),
            low_review_count=(
                "review_score",
                lambda x: (
                    x < 3
                ).sum()
            ),
        )
        .reset_index()
    )

    grouped[
        "low_review_rate"
    ] = (
        grouped[
            "low_review_count"
        ]
        / grouped[
            "review_count"
        ]
        * 100
    )

    return grouped


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TEMPORAL TREND FEATURES")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING SOURCE DATA"
    )

    for path, label in [
        (
            ORDERS_FILE,
            "Orders dataset",
        ),
        (
            ORDER_ITEMS_FILE,
            "Order items dataset",
        ),
        (
            REVIEWS_FILE,
            "Reviews dataset",
        ),
        (
            TEMPORAL_DATASET_FILE,
            "Temporal seller dataset",
        ),
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

    orders = pd.read_csv(
        ORDERS_FILE
    )

    order_items = pd.read_csv(
        ORDER_ITEMS_FILE
    )

    reviews = pd.read_csv(
        REVIEWS_FILE
    )

    temporal_df = pd.read_csv(
        TEMPORAL_DATASET_FILE
    )

    print(
        f"Orders: {len(orders):,}"
    )

    print(
        f"Order items: {len(order_items):,}"
    )

    print(
        f"Reviews: {len(reviews):,}"
    )

    print(
        f"Temporal observations: "
        f"{len(temporal_df):,}"
    )

    # ========================================================
    # 2. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING DATE COLUMNS"
    )

    order_date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in order_date_columns:

        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce",
        )

    reviews[
        "review_creation_date"
    ] = pd.to_datetime(
        reviews[
            "review_creation_date"
        ],
        errors="coerce",
    )

    temporal_df[
        DATE_COLUMN
    ] = pd.to_datetime(
        temporal_df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    if temporal_df[
        DATE_COLUMN
    ].isna().any():

        raise ValueError(
            "Invalid snapshot dates found."
        )

    print(
        "Date conversion -> PASS"
    )

    # ========================================================
    # 3. BUILD ORDER-SELLER HISTORY
    # ========================================================

    print_section(
        "BUILDING ORDER-SELLER HISTORY"
    )

    order_seller = (
        build_order_seller_history(
            orders,
            order_items,
        )
    )

    print(
        f"Order-seller observations: "
        f"{len(order_seller):,}"
    )

    print(
        f"Unique sellers with order history: "
        f"{order_seller[SELLER_COLUMN].nunique():,}"
    )

    # ========================================================
    # 4. BUILD REVIEW-SELLER HISTORY
    # ========================================================

    print_section(
        "BUILDING REVIEW-SELLER HISTORY"
    )

    review_orders = (
        order_seller[
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
            reviews[
                [
                    "order_id",
                    "review_score",
                    "review_creation_date",
                ]
            ],
            on="order_id",
            how="left",
        )
    )

    print(
        f"Review-seller observations: "
        f"{len(review_orders):,}"
    )

    # ========================================================
    # 5. BUILD SNAPSHOT FEATURES
    # ========================================================

    print_section(
        "BUILDING RECENCY AND TREND FEATURES"
    )

    all_feature_rows = []

    snapshot_dates = sorted(
        temporal_df[
            DATE_COLUMN
        ]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Snapshots to process: "
        f"{len(snapshot_dates)}"
    )

    for snapshot_number, snapshot_date in enumerate(
        snapshot_dates,
        start=1,
    ):

        # ----------------------------------------------------
        # Time windows
        #
        # Older window:
        #     90 days before snapshot
        #     through 30 days before snapshot
        #
        # Recent window:
        #     30 days immediately before snapshot
        # ----------------------------------------------------

        recent_start = (
            snapshot_date
            - pd.Timedelta(
                days=RECENT_DAYS
            )
        )

        older_start = (
            snapshot_date
            - pd.Timedelta(
                days=RECENT_DAYS
                + OLDER_DAYS
            )
        )

        cutoff = snapshot_date

        # ----------------------------------------------------
        # Historical order features
        # ----------------------------------------------------

        recent_orders = (
            aggregate_order_window(
                order_seller,
                recent_start,
                cutoff,
            )
        )

        older_orders = (
            aggregate_order_window(
                order_seller,
                older_start,
                recent_start,
            )
        )

        # ----------------------------------------------------
        # Historical review features
        # ----------------------------------------------------

        recent_reviews = (
            aggregate_review_window(
                review_orders,
                recent_start,
                cutoff,
            )
        )

        older_reviews = (
            aggregate_review_window(
                review_orders,
                older_start,
                recent_start,
            )
        )

        # ----------------------------------------------------
        # Determine sellers present in either period
        # ----------------------------------------------------

        seller_ids = set()

        if not recent_orders.empty:

            seller_ids.update(
                recent_orders[
                    SELLER_COLUMN
                ]
                .tolist()
            )

        if not older_orders.empty:

            seller_ids.update(
                older_orders[
                    SELLER_COLUMN
                ]
                .tolist()
            )

        if not recent_reviews.empty:

            seller_ids.update(
                recent_reviews[
                    SELLER_COLUMN
                ]
                .tolist()
            )

        if not older_reviews.empty:

            seller_ids.update(
                older_reviews[
                    SELLER_COLUMN
                ]
                .tolist()
            )

        if not seller_ids:
            continue

        snapshot_features = pd.DataFrame(
            {
                SELLER_COLUMN: list(
                    seller_ids
                )
            }
        )

        # ----------------------------------------------------
        # Merge recent order features
        # ----------------------------------------------------

        if not recent_orders.empty:

            recent_orders = (
                recent_orders
                .add_prefix(
                    "recent_"
                )
                .rename(
                    columns={
                        "recent_seller_id":
                            SELLER_COLUMN
                    }
                )
            )

            snapshot_features = (
                snapshot_features
                .merge(
                    recent_orders,
                    on=SELLER_COLUMN,
                    how="left",
                )
            )

        # ----------------------------------------------------
        # Merge older order features
        # ----------------------------------------------------

        if not older_orders.empty:

            older_orders = (
                older_orders
                .add_prefix(
                    "older_"
                )
                .rename(
                    columns={
                        "older_seller_id":
                            SELLER_COLUMN
                    }
                )
            )

            snapshot_features = (
                snapshot_features
                .merge(
                    older_orders,
                    on=SELLER_COLUMN,
                    how="left",
                )
            )

        # ----------------------------------------------------
        # Merge recent review features
        # ----------------------------------------------------

        if not recent_reviews.empty:

            recent_reviews = (
                recent_reviews
                .add_prefix(
                    "recent_"
                )
                .rename(
                    columns={
                        "recent_seller_id":
                            SELLER_COLUMN
                    }
                )
            )

            snapshot_features = (
                snapshot_features
                .merge(
                    recent_reviews,
                    on=SELLER_COLUMN,
                    how="left",
                )
            )

        # ----------------------------------------------------
        # Merge older review features
        # ----------------------------------------------------

        if not older_reviews.empty:

            older_reviews = (
                older_reviews
                .add_prefix(
                    "older_"
                )
                .rename(
                    columns={
                        "older_seller_id":
                            SELLER_COLUMN
                    }
                )
            )

            snapshot_features = (
                snapshot_features
                .merge(
                    older_reviews,
                    on=SELLER_COLUMN,
                    how="left",
                )
            )

        # ----------------------------------------------------
        # Add snapshot date
        # ----------------------------------------------------

        snapshot_features[
            DATE_COLUMN
        ] = snapshot_date

        # ----------------------------------------------------
        # Fill genuine zero-activity measures
        #
        # If a seller had no orders/reviews in a window,
        # activity counts should be zero.
        #
        # We intentionally DO NOT fill delivery/review averages
        # with zero because zero days or zero review score would
        # have a misleading meaning.
        # ----------------------------------------------------

        zero_features = [
            "recent_orders",
            "older_orders",
            "recent_units_sold",
            "older_units_sold",
            "recent_revenue",
            "older_revenue",
            "recent_review_count",
            "older_review_count",
            "recent_low_review_count",
            "older_low_review_count",
            "recent_delivered_orders",
            "older_delivered_orders",
            "recent_late_orders",
            "older_late_orders",
        ]

        for feature in zero_features:

            if feature in snapshot_features.columns:

                snapshot_features[
                    feature
                ] = snapshot_features[
                    feature
                ].fillna(0)

        # ----------------------------------------------------
        # Activity changes
        # ----------------------------------------------------

        snapshot_features[
            "orders_change"
        ] = (
            snapshot_features["recent_orders"]
            -
            snapshot_features["older_orders"]
        )

        snapshot_features[
            "orders_change_percent"
        ] = snapshot_features.apply(
            lambda row: safe_percentage_change(
                row["recent_orders"],
                row["older_orders"],
            ),
            axis=1,
        )

        snapshot_features[
            "revenue_change"
        ] = (
            snapshot_features["recent_revenue"]
            -
            snapshot_features["older_revenue"]
        )

        snapshot_features[
            "revenue_change_percent"
        ] = snapshot_features.apply(
            lambda row: safe_percentage_change(
                row["recent_revenue"],
                row["older_revenue"],
            ),
            axis=1,
        )

        snapshot_features[
            "units_change"
        ] = (
            snapshot_features["recent_units_sold"]
            -
            snapshot_features["older_units_sold"]
        )

        snapshot_features[
            "units_change_percent"
        ] = snapshot_features.apply(
            lambda row: safe_percentage_change(
                row["recent_units_sold"],
                row["older_units_sold"],
            ),
            axis=1,
        )

        # ----------------------------------------------------
        # Delivery changes
        # ----------------------------------------------------

        snapshot_features[
            "delivery_days_change"
        ] = snapshot_features.apply(
            lambda row: safe_difference(
                row["recent_average_delivery_days"],
                row["older_average_delivery_days"],
            ),
            axis=1,
        )

        snapshot_features[
            "delivery_delay_change"
        ] = snapshot_features.apply(
            lambda row: safe_difference(
                row[
                    "recent_average_delivery_delay_days"
                ],
                row[
                    "older_average_delivery_delay_days"
                ],
            ),
            axis=1,
        )

        snapshot_features[
            "estimated_delivery_change"
        ] = snapshot_features.apply(
            lambda row: safe_difference(
                row[
                    "recent_average_estimated_delivery_days"
                ],
                row[
                    "older_average_estimated_delivery_days"
                ],
            ),
            axis=1,
        )

        snapshot_features[
            "late_delivery_rate_change"
        ] = snapshot_features.apply(
            lambda row: safe_difference(
                row[
                    "recent_late_delivery_rate"
                ],
                row[
                    "older_late_delivery_rate"
                ],
            ),
            axis=1,
        )

        # ----------------------------------------------------
        # Review changes
        # ----------------------------------------------------

        snapshot_features[
            "review_score_change"
        ] = snapshot_features.apply(
            lambda row: safe_difference(
                row[
                    "recent_average_review_score"
                ],
                row[
                    "older_average_review_score"
                ],
            ),
            axis=1,
        )

        snapshot_features[
            "low_review_rate_change"
        ] = snapshot_features.apply(
            lambda row: safe_difference(
                row[
                    "recent_low_review_rate"
                ],
                row[
                    "older_low_review_rate"
                ],
            ),
            axis=1,
        )

        snapshot_features[
            "review_count_change"
        ] = (
            snapshot_features[
                "recent_review_count"
            ]
            -
            snapshot_features[
                "older_review_count"
            ]
        )

        snapshot_features[
            "review_count_change_percent"
        ] = snapshot_features.apply(
            lambda row: safe_percentage_change(
                row[
                    "recent_review_count"
                ],
                row[
                    "older_review_count"
                ],
            ),
            axis=1,
        )

        # ----------------------------------------------------
        # Recent intensity / scale features
        # ----------------------------------------------------

        snapshot_features[
            "recent_orders_per_day"
        ] = (
            snapshot_features[
                "recent_orders"
            ]
            / RECENT_DAYS
        )

        snapshot_features[
            "older_orders_per_day"
        ] = (
            snapshot_features[
                "older_orders"
            ]
            / OLDER_DAYS
        )

        snapshot_features[
            "recent_revenue_per_day"
        ] = (
            snapshot_features[
                "recent_revenue"
            ]
            / RECENT_DAYS
        )

        snapshot_features[
            "older_revenue_per_day"
        ] = (
            snapshot_features[
                "older_revenue"
            ]
            / OLDER_DAYS
        )

        snapshot_features[
            "seller_history_orders"
        ] = (
            snapshot_features[
                "recent_orders"
            ]
            +
            snapshot_features[
                "older_orders"
            ]
        )

        snapshot_features[
            "seller_history_reviews"
        ] = (
            snapshot_features[
                "recent_review_count"
            ]
            +
            snapshot_features[
                "older_review_count"
            ]
        )

        # ----------------------------------------------------
        # Append snapshot
        # ----------------------------------------------------

        all_feature_rows.append(
            snapshot_features
        )

        print(
            f"{snapshot_date.date()} | "
            f"recent start: "
            f"{recent_start.date()} | "
            f"observations: "
            f"{len(snapshot_features):,}"
        )

    # ========================================================
    # 6. COMBINE SNAPSHOTS
    # ========================================================

    print_section(
        "COMBINING TREND FEATURES"
    )

    if not all_feature_rows:

        raise ValueError(
            "No trend-feature snapshots were created."
        )

    features_df = pd.concat(
        all_feature_rows,
        ignore_index=True,
    )

    print(
        f"Feature observations: "
        f"{len(features_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{features_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{features_df[DATE_COLUMN].nunique():,}"
    )

    # ========================================================
    # 7. MERGE FUTURE TARGET
    # ========================================================

    print_section(
        "ADDING FUTURE TARGET"
    )

    target_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        "future_at_risk",
    ]

    target_df = temporal_df[
        target_columns
    ].copy()

    target_df[DATE_COLUMN] = pd.to_datetime(
        target_df[DATE_COLUMN],
        errors="coerce",
    )

    features_df = features_df.merge(
        target_df,
        on=[
            SELLER_COLUMN,
            DATE_COLUMN,
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "Future target merged -> PASS"
    )

    # ========================================================
    # 8. VALIDATE TEMPORAL BOUNDARY
    # ========================================================

    print_section(
        "TEMPORAL FEATURE VALIDATION"
    )

    snapshot_set = set(
        temporal_df[
            DATE_COLUMN
        ]
        .dropna()
        .tolist()
    )

    trend_snapshot_set = set(
        features_df[
            DATE_COLUMN
        ]
        .dropna()
        .tolist()
    )

    unexpected_snapshots = (
        trend_snapshot_set
        - snapshot_set
    )

    if unexpected_snapshots:

        raise ValueError(
            "Trend features contain snapshots "
            "not found in the temporal dataset."
        )

    print(
        "Snapshot alignment -> PASS"
    )

    duplicate_keys = (
        features_df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate seller/snapshot rows: "
        f"{duplicate_keys}"
    )

    if duplicate_keys > 0:

        raise ValueError(
            "Duplicate seller/snapshot feature rows found."
        )

    print(
        "Seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 9. TARGET VALIDATION
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    missing_targets = (
        features_df[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing target values: "
        f"{missing_targets}"
    )

    if missing_targets > 0:

        raise ValueError(
            "Missing targets found after merge."
        )

    invalid_targets = (
        set(
            features_df[
                TARGET_COLUMN
            ]
            .unique()
        )
        - {0, 1}
    )

    print(
        f"Invalid target values: "
        f"{len(invalid_targets)}"
    )

    if invalid_targets:

        raise ValueError(
            f"Invalid target values: "
            f"{invalid_targets}"
        )

    print(
        "Target validation -> PASS"
    )

    # ========================================================
    # 10. FEATURE SUMMARY
    # ========================================================

    print_section(
        "TREND FEATURE SUMMARY"
    )

    excluded_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    feature_columns = [
        column
        for column in features_df.columns
        if column not in excluded_columns
    ]

    print(
        f"Number of trend features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nTrend features:"
    )

    for feature in feature_columns:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 11. MISSING VALUE ANALYSIS
    # ========================================================

    print_section(
        "TREND FEATURE MISSING VALUES"
    )

    missing_features = (
        features_df[
            feature_columns
        ]
        .isna()
        .sum()
    )

    missing_features = (
        missing_features[
            missing_features > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if missing_features.empty:

        print(
            "No missing trend-feature values."
        )

    else:

        print(
            missing_features.to_string()
        )

        print(
            "\nThese missing values are expected for "
            "some trend calculations when there is no "
            "historical activity in one of the comparison windows."
        )

    # ========================================================
    # 12. TREND STATISTICS
    # ========================================================

    print_section(
        "KEY TREND STATISTICS"
    )

    trend_columns = [
        "orders_change",
        "orders_change_percent",
        "revenue_change",
        "revenue_change_percent",
        "delivery_days_change",
        "delivery_delay_change",
        "late_delivery_rate_change",
        "review_score_change",
        "low_review_rate_change",
        "review_count_change",
        "review_count_change_percent",
    ]

    available_trend_columns = [
        column
        for column in trend_columns
        if column in features_df.columns
    ]

    trend_summary = (
        features_df[
            available_trend_columns
        ]
        .describe()
        .T
    )

    print(
        trend_summary.round(3).to_string()
    )

    # ========================================================
    # 13. FUTURE TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "FUTURE TARGET DISTRIBUTION"
    )

    print(
        features_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nPercentages:"
    )

    print(
        (
            features_df[
                TARGET_COLUMN
            ]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )

    # ========================================================
    # 14. SAMPLE TREND OBSERVATIONS
    # ========================================================

    print_section(
        "SAMPLE TREND OBSERVATIONS"
    )

    sample_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        "recent_orders",
        "older_orders",
        "orders_change_percent",
        "recent_revenue",
        "older_revenue",
        "revenue_change_percent",
        "recent_average_delivery_days",
        "older_average_delivery_days",
        "delivery_days_change",
        "recent_average_review_score",
        "older_average_review_score",
        "review_score_change",
        TARGET_COLUMN,
    ]

    sample_columns = [
        column
        for column in sample_columns
        if column in features_df.columns
    ]

    print(
        features_df[
            sample_columns
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 15. SAVE
    # ========================================================

    print_section(
        "SAVING TEMPORAL TREND FEATURES"
    )

    features_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Temporal trend feature dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL TREND FEATURE ENGINEERING COMPLETE"
    )

    print(
        f"Observations: "
        f"{len(features_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{features_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{features_df[DATE_COLUMN].nunique():,}"
    )

    print(
        f"Trend features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nThe new feature set captures recent behavior, "
        "historical baseline behavior, and changes between them."
    )

    print(
        "\nNo future outcome columns were used to construct "
        "the predictors."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()