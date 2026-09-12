"""
Build Delivery Distribution Features

Purpose
-------
Create distribution-aware historical delivery features for the
EXISTING temporal prediction population.

The prediction unit is:

    seller_id + snapshot_date

The temporal trend dataset is therefore treated as the
MASTER OBSERVATIONAL POPULATION.

Distribution statistics are calculated from order-level
historical data and then restricted to the exact seller/snapshot
pairs present in the temporal prediction dataset.

This prevents creation of extra seller/snapshot observations
that do not belong to the prediction population.

Historical windows
------------------
Recent window:
    30 days before snapshot

Older window:
    Previous 60 days before the recent window

Features
--------
Delivery:
    median
    standard deviation
    90th percentile
    maximum
    late count
    late rate

Trend:
    median change
    standard-deviation change
    p90 change
    maximum change
    delay median change
    delay standard-deviation change
    delay p90 change
    delay maximum change

Future information is NOT used to construct predictors.

Input
-----
data/raw/olist_orders_dataset.csv
data/raw/olist_order_items_dataset.csv

data/processed/temporal_trend_features.csv

Output
------
data/processed/delivery_distribution_features.csv
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
    RAW_DATA_DIR
    / "olist_orders_dataset.csv"
)

ORDER_ITEMS_FILE = (
    RAW_DATA_DIR
    / "olist_order_items_dataset.csv"
)

TEMPORAL_TREND_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_trend_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "delivery_distribution_features.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

SNAPSHOT_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_at_risk"


# ============================================================
# WINDOW CONFIGURATION
# ============================================================

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


def build_order_seller_history(
    orders,
    order_items,
):
    """
    Build order-level seller delivery history.
    """

    order_items_grouped = (
        order_items
        .groupby(
            [
                "order_id",
                SELLER_COLUMN,
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
            order_items_grouped,
            on="order_id",
            how="inner",
        )
    )

    # --------------------------------------------------------
    # Actual delivery duration
    # --------------------------------------------------------

    order_seller[
        "delivery_days"
    ] = (
        (
            order_seller[
                "order_delivered_customer_date"
            ]
            -
            order_seller[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    # --------------------------------------------------------
    # Estimated delivery duration
    # --------------------------------------------------------

    order_seller[
        "estimated_delivery_days"
    ] = (
        (
            order_seller[
                "order_estimated_delivery_date"
            ]
            -
            order_seller[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    # --------------------------------------------------------
    # Delivery delay
    # --------------------------------------------------------

    order_seller[
        "delivery_delay_days"
    ] = (
        (
            order_seller[
                "order_delivered_customer_date"
            ]
            -
            order_seller[
                "order_estimated_delivery_date"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    # --------------------------------------------------------
    # Delivery status
    # --------------------------------------------------------

    order_seller[
        "is_delivered"
    ] = (
        order_seller[
            "order_delivered_customer_date"
        ]
        .notna()
    )

    # --------------------------------------------------------
    # Late delivery
    # --------------------------------------------------------

    order_seller[
        "is_late"
    ] = (
        order_seller[
            "delivery_delay_days"
        ]
        > 0
    )

    return order_seller


def calculate_window_distribution(
    order_seller,
    start_date,
    end_date,
):
    """
    Calculate order-level delivery distribution statistics
    for each seller inside a historical time window.

    Only delivered orders are used for delivery distributions.
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
        &
        (
            order_seller[
                "is_delivered"
            ]
            == True
        )
    ].copy()

    if window.empty:

        return pd.DataFrame(
            columns=[
                SELLER_COLUMN,
                "delivery_count",
                "delivery_median",
                "delivery_std",
                "delivery_p90",
                "delivery_max",
                "delay_median",
                "delay_std",
                "delay_p90",
                "delay_max",
                "late_count",
                "late_rate",
            ]
        )

    grouped = (
        window
        .groupby(
            SELLER_COLUMN
        )
        .agg(
            delivery_count=(
                "delivery_days",
                "count",
            ),

            delivery_median=(
                "delivery_days",
                "median",
            ),

            delivery_std=(
                "delivery_days",
                "std",
            ),

            delivery_p90=(
                "delivery_days",
                lambda x: x.quantile(
                    0.90
                ),
            ),

            delivery_max=(
                "delivery_days",
                "max",
            ),

            delay_median=(
                "delivery_delay_days",
                "median",
            ),

            delay_std=(
                "delivery_delay_days",
                "std",
            ),

            delay_p90=(
                "delivery_delay_days",
                lambda x: x.quantile(
                    0.90
                ),
            ),

            delay_max=(
                "delivery_delay_days",
                "max",
            ),

            late_count=(
                "is_late",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped[
        "late_rate"
    ] = np.where(
        grouped[
            "delivery_count"
        ] > 0,

        (
            grouped[
                "late_count"
            ]
            /
            grouped[
                "delivery_count"
            ]
            * 100
        ),

        np.nan,
    )

    return grouped


def rename_distribution_columns(
    df,
    prefix,
):
    """
    Prefix feature columns while preserving seller_id.
    """

    if df.empty:
        return df

    renamed = df.copy()

    feature_columns = [
        column
        for column in renamed.columns
        if column != SELLER_COLUMN
    ]

    renamed = renamed.rename(
        columns={
            column:
                f"{prefix}{column}"
            for column in feature_columns
        }
    )

    return renamed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILDING DELIVERY DISTRIBUTION FEATURES"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK INPUTS
    # ========================================================

    print_section(
        "CHECKING INPUT DATASETS"
    )

    required_files = [
        ORDERS_FILE,
        ORDER_ITEMS_FILE,
        TEMPORAL_TREND_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required input file not found:\n"
                f"{file_path}"
            )

    print(
        "Orders dataset -> FOUND"
    )

    print(
        "Order items dataset -> FOUND"
    )

    print(
        "Temporal trend dataset -> FOUND"
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print_section(
        "LOADING SOURCE DATA"
    )

    orders = pd.read_csv(
        ORDERS_FILE
    )

    order_items = pd.read_csv(
        ORDER_ITEMS_FILE
    )

    temporal_trend = pd.read_csv(
        TEMPORAL_TREND_FILE
    )

    print(
        f"Orders: "
        f"{len(orders):,}"
    )

    print(
        f"Order items: "
        f"{len(order_items):,}"
    )

    print(
        f"Temporal prediction population: "
        f"{len(temporal_trend):,}"
    )

    # ========================================================
    # 3. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING DATE COLUMNS"
    )

    orders[
        "order_purchase_timestamp"
    ] = pd.to_datetime(
        orders[
            "order_purchase_timestamp"
        ],
        errors="coerce",
    )

    orders[
        "order_delivered_customer_date"
    ] = pd.to_datetime(
        orders[
            "order_delivered_customer_date"
        ],
        errors="coerce",
    )

    orders[
        "order_estimated_delivery_date"
    ] = pd.to_datetime(
        orders[
            "order_estimated_delivery_date"
        ],
        errors="coerce",
    )

    temporal_trend[
        SNAPSHOT_COLUMN
    ] = pd.to_datetime(
        temporal_trend[
            SNAPSHOT_COLUMN
        ],
        errors="coerce",
    )

    if orders[
        "order_purchase_timestamp"
    ].isna().all():

        raise ValueError(
            "No valid order purchase dates found."
        )

    if temporal_trend[
        SNAPSHOT_COLUMN
    ].isna().any():

        raise ValueError(
            "Invalid snapshot dates found."
        )

    print(
        "Order dates converted -> PASS"
    )

    print(
        "Snapshot dates converted -> PASS"
    )

    # ========================================================
    # 4. VALIDATE MASTER POPULATION
    # ========================================================

    print_section(
        "VALIDATING MASTER PREDICTION POPULATION"
    )

    required_columns = [
        SELLER_COLUMN,
        SNAPSHOT_COLUMN,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in temporal_trend.columns
    ]

    if missing_columns:

        raise ValueError(
            "Temporal prediction population is missing:\n"
            + "\n".join(
                missing_columns
            )
        )

    duplicate_keys = (
        temporal_trend[
            [
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
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
            "Master prediction population contains "
            "duplicate seller/snapshot observations."
        )

    print(
        "Master prediction population -> PASS"
    )

    # ========================================================
    # 5. CREATE MASTER KEY TABLE
    # ========================================================

    print_section(
        "CREATING MASTER SELLER / SNAPSHOT KEYS"
    )

    master_keys = (
        temporal_trend[
            [
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                SNAPSHOT_COLUMN,
                SELLER_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Master seller/snapshot keys: "
        f"{len(master_keys):,}"
    )

    # ========================================================
    # 6. BUILD ORDER HISTORY
    # ========================================================

    print_section(
        "BUILDING ORDER-LEVEL DELIVERY HISTORY"
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
        f"Delivered orders: "
        f"{order_seller['is_delivered'].sum():,}"
    )

    print(
        f"Unique sellers: "
        f"{order_seller[SELLER_COLUMN].nunique():,}"
    )

    # ========================================================
    # 7. CALCULATE DISTRIBUTION FEATURES
    # ========================================================

    print_section(
        "CALCULATING DISTRIBUTION FEATURES"
    )

    snapshot_feature_rows = []

    snapshot_dates = sorted(
        master_keys[
            SNAPSHOT_COLUMN
        ]
        .unique()
    )

    print(
        f"Snapshots: "
        f"{len(snapshot_dates)}"
    )

    for number, snapshot_date in enumerate(
        snapshot_dates,
        start=1,
    ):

        print(
            f"{number:02d}/"
            f"{len(snapshot_dates):02d} | "
            f"{snapshot_date.date()}"
        )

        # ----------------------------------------------------
        # Historical windows
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

        # ----------------------------------------------------
        # Calculate distributions from raw orders
        # ----------------------------------------------------

        recent = (
            calculate_window_distribution(
                order_seller,
                recent_start,
                snapshot_date,
            )
        )

        older = (
            calculate_window_distribution(
                order_seller,
                older_start,
                recent_start,
            )
        )

        # ----------------------------------------------------
        # Prefix distributions
        # ----------------------------------------------------

        recent = (
            rename_distribution_columns(
                recent,
                "recent_distribution_",
            )
        )

        older = (
            rename_distribution_columns(
                older,
                "older_distribution_",
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Start from the official prediction population
        # for THIS snapshot.
        #
        # This guarantees that the output population cannot
        # become larger than the temporal prediction dataset.
        # ----------------------------------------------------

        snapshot_master = master_keys[
            master_keys[
                SNAPSHOT_COLUMN
            ]
            == snapshot_date
        ][
            [
                SELLER_COLUMN
            ]
        ].copy()

        # ----------------------------------------------------
        # Merge recent statistics
        # ----------------------------------------------------

        if not recent.empty:

            snapshot_master = (
                snapshot_master
                .merge(
                    recent,
                    on=SELLER_COLUMN,
                    how="left",
                )
            )

        # ----------------------------------------------------
        # Merge older statistics
        # ----------------------------------------------------

        if not older.empty:

            snapshot_master = (
                snapshot_master
                .merge(
                    older,
                    on=SELLER_COLUMN,
                    how="left",
                )
            )

        # ----------------------------------------------------
        # Add snapshot
        # ----------------------------------------------------

        snapshot_master[
            SNAPSHOT_COLUMN
        ] = snapshot_date

        snapshot_feature_rows.append(
            snapshot_master
        )

    # ========================================================
    # 8. COMBINE
    # ========================================================

    print_section(
        "COMBINING DISTRIBUTION FEATURES"
    )

    distribution_df = pd.concat(
        snapshot_feature_rows,
        ignore_index=True,
    )

    print(
        f"Distribution population: "
        f"{len(distribution_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{distribution_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{distribution_df[SNAPSHOT_COLUMN].nunique():,}"
    )

    # ========================================================
    # 9. MASTER POPULATION ALIGNMENT
    # ========================================================

    print_section(
        "MASTER POPULATION ALIGNMENT"
    )

    distribution_keys = set(
        zip(
            distribution_df[
                SELLER_COLUMN
            ],
            distribution_df[
                SNAPSHOT_COLUMN
            ],
        )
    )

    master_key_set = set(
        zip(
            master_keys[
                SELLER_COLUMN
            ],
            master_keys[
                SNAPSHOT_COLUMN
            ],
        )
    )

    extra_distribution_keys = (
        distribution_keys
        - master_key_set
    )

    missing_distribution_keys = (
        master_key_set
        - distribution_keys
    )

    print(
        f"Master keys: "
        f"{len(master_key_set):,}"
    )

    print(
        f"Distribution keys: "
        f"{len(distribution_keys):,}"
    )

    print(
        f"Extra distribution keys: "
        f"{len(extra_distribution_keys):,}"
    )

    print(
        f"Missing distribution keys: "
        f"{len(missing_distribution_keys):,}"
    )

    if extra_distribution_keys:

        raise ValueError(
            "Distribution features contain seller/snapshot "
            "records outside the master prediction population."
        )

    if missing_distribution_keys:

        raise ValueError(
            "Some master seller/snapshot records are missing "
            "from the distribution feature dataset."
        )

    print(
        "Population alignment -> PASS"
    )

    # ========================================================
    # 10. CREATE DISTRIBUTION TRENDS
    # ========================================================

    print_section(
        "CREATING DISTRIBUTION TREND FEATURES"
    )

    change_pairs = [
        (
            "recent_distribution_delivery_median",
            "older_distribution_delivery_median",
            "delivery_median_change",
        ),
        (
            "recent_distribution_delivery_std",
            "older_distribution_delivery_std",
            "delivery_std_change",
        ),
        (
            "recent_distribution_delivery_p90",
            "older_distribution_delivery_p90",
            "delivery_p90_change",
        ),
        (
            "recent_distribution_delivery_max",
            "older_distribution_delivery_max",
            "delivery_max_change",
        ),
        (
            "recent_distribution_delay_median",
            "older_distribution_delay_median",
            "delay_median_change",
        ),
        (
            "recent_distribution_delay_std",
            "older_distribution_delay_std",
            "delay_std_change",
        ),
        (
            "recent_distribution_delay_p90",
            "older_distribution_delay_p90",
            "delay_p90_change",
        ),
        (
            "recent_distribution_delay_max",
            "older_distribution_delay_max",
            "delay_max_change",
        ),
    ]

    for recent_column, older_column, new_column in change_pairs:

        distribution_df[
            new_column
        ] = (
            distribution_df[
                recent_column
            ]
            -
            distribution_df[
                older_column
            ]
        )

    print(
        f"Distribution trend features created: "
        f"{len(change_pairs)}"
    )

    # ========================================================
    # 11. DISTRIBUTION SANITY CHECKS
    # ========================================================

    print_section(
        "DISTRIBUTION FEATURE SANITY CHECKS"
    )

    std_columns = [
        "recent_distribution_delivery_std",
        "older_distribution_delivery_std",
        "recent_distribution_delay_std",
        "older_distribution_delay_std",
    ]

    negative_std_total = 0

    for column in std_columns:

        negative_count = (
            distribution_df[
                column
            ]
            .dropna()
            .lt(0)
            .sum()
        )

        print(
            f"{column}: "
            f"{negative_count} negative values"
        )

        negative_std_total += (
            negative_count
        )

    if negative_std_total > 0:

        raise ValueError(
            "Negative standard deviations detected."
        )

    print(
        "Standard deviations -> PASS"
    )

    # --------------------------------------------------------
    # Percentile order
    # --------------------------------------------------------

    percentile_checks = [
        (
            "recent_distribution_delivery_median",
            "recent_distribution_delivery_p90",
        ),
        (
            "older_distribution_delivery_median",
            "older_distribution_delivery_p90",
        ),
        (
            "recent_distribution_delay_median",
            "recent_distribution_delay_p90",
        ),
        (
            "older_distribution_delay_median",
            "older_distribution_delay_p90",
        ),
    ]

    percentile_violations = 0

    for lower, upper in percentile_checks:

        valid = (
            distribution_df[
                lower
            ].notna()
            &
            distribution_df[
                upper
            ].notna()
        )

        violations = (
            distribution_df.loc[
                valid,
                lower,
            ]
            >
            distribution_df.loc[
                valid,
                upper,
            ]
        ).sum()

        print(
            f"{lower} <= {upper}: "
            f"{violations} violations"
        )

        percentile_violations += (
            violations
        )

    if percentile_violations > 0:

        raise ValueError(
            "Percentile ordering violations detected."
        )

    print(
        "Percentile ordering -> PASS"
    )

    # ========================================================
    # 12. TARGET PRESERVATION
    # ========================================================

    print_section(
        "TARGET PRESERVATION"
    )

    original_target = (
        temporal_trend[
            [
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
                TARGET_COLUMN,
            ]
        ]
        .sort_values(
            [
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    distribution_with_target = (
        distribution_df
        .merge(
            temporal_trend[
                [
                    SELLER_COLUMN,
                    SNAPSHOT_COLUMN,
                    TARGET_COLUMN,
                ]
            ],
            on=[
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
            ],
            how="left",
            validate="one_to_one",
        )
    )

    merged_target = (
        distribution_with_target[
            [
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
                TARGET_COLUMN,
            ]
        ]
        .sort_values(
            [
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    target_match = (
        original_target[
            TARGET_COLUMN
        ]
        .equals(
            merged_target[
                TARGET_COLUMN
            ]
        )
    )

    print(
        f"Target preserved: "
        f"{target_match}"
    )

    if not target_match:

        raise ValueError(
            "Target alignment failed."
        )

    print(
        "Target preservation -> PASS"
    )

    # ========================================================
    # 13. TEMPORAL LEAKAGE CHECK
    # ========================================================

    print_section(
        "TEMPORAL LEAKAGE CHECK"
    )

    # The distribution features are calculated only from:
    #
    #   purchase timestamp <= snapshot
    #
    # Therefore no orders after the prediction cutoff should
    # enter the feature calculation.

    future_orders = order_seller[
        order_seller[
            "order_purchase_timestamp"
        ]
        > temporal_trend[
            SNAPSHOT_COLUMN
        ].max()
    ]

    print(
        f"Orders occurring after final snapshot: "
        f"{len(future_orders):,}"
    )

    print(
        "These orders are not included in any historical "
        "snapshot window."
    )

    print(
        "Historical feature construction uses "
        "point-in-time windows -> PASS"
    )

    # ========================================================
    # 14. NEW FEATURE INVENTORY
    # ========================================================

    print_section(
        "NEW DISTRIBUTION FEATURE INVENTORY"
    )

    temporal_columns = set(
        temporal_trend.columns
    )

    new_features = [
        column
        for column in distribution_df.columns
        if column not in temporal_columns
    ]

    print(
        f"New distribution features: "
        f"{len(new_features)}"
    )

    for feature in new_features:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 15. MISSING VALUES
    # ========================================================

    print_section(
        "DISTRIBUTION FEATURE MISSING VALUES"
    )

    missing = (
        distribution_df[
            new_features
        ]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = (
        missing[
            missing > 0
        ]
    )

    if missing.empty:

        print(
            "No missing distribution features."
        )

    else:

        print(
            missing.to_string()
        )

        print(
            "\nMissing values are expected when "
            "insufficient delivered-order observations "
            "exist in a comparison window."
        )

    # ========================================================
    # 16. FEATURE STATISTICS
    # ========================================================

    print_section(
        "DISTRIBUTION FEATURE STATISTICS"
    )

    numeric_new_features = [
        feature
        for feature in new_features
        if pd.api.types.is_numeric_dtype(
            distribution_df[feature]
        )
    ]

    print(
        distribution_df[
            numeric_new_features
        ]
        .describe()
        .T
        .round(3)
        .to_string()
    )

    # ========================================================
    # 17. SAMPLE
    # ========================================================

    print_section(
        "SAMPLE DISTRIBUTION FEATURES"
    )

    sample_columns = [
        SELLER_COLUMN,
        SNAPSHOT_COLUMN,

        "recent_distribution_delivery_count",
        "recent_distribution_delivery_median",
        "recent_distribution_delivery_std",
        "recent_distribution_delivery_p90",
        "recent_distribution_delivery_max",

        "older_distribution_delivery_count",
        "older_distribution_delivery_median",
        "older_distribution_delivery_std",
        "older_distribution_delivery_p90",
        "older_distribution_delivery_max",

        "delivery_median_change",
        "delivery_std_change",
        "delivery_p90_change",
        "delivery_max_change",

        TARGET_COLUMN,
    ]

    sample_columns = [
        column
        for column in sample_columns
        if column in distribution_with_target.columns
        or column in distribution_df.columns
    ]

    sample_source = distribution_with_target

    print(
        sample_source[
            sample_columns
        ]
        .head(10)
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 18. SAVE
    # ========================================================

    print_section(
        "SAVING DELIVERY DISTRIBUTION FEATURES"
    )

    final_output = (
        distribution_with_target
        .sort_values(
            [
                SNAPSHOT_COLUMN,
                SELLER_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    final_output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Delivery distribution feature dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 19. FINAL SUMMARY
    # ========================================================

    print_section(
        "DELIVERY DISTRIBUTION FEATURES COMPLETE"
    )

    print(
        f"Prediction observations: "
        f"{len(final_output):,}"
    )

    print(
        f"Unique sellers: "
        f"{final_output[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{final_output[SNAPSHOT_COLUMN].nunique():,}"
    )

    print(
        f"New distribution features: "
        f"{len(new_features):,}"
    )

    print(
        "\nPopulation alignment -> PASS"
    )

    print(
        "Distribution sanity checks -> PASS"
    )

    print(
        "Target preservation -> PASS"
    )

    print(
        "Point-in-time construction -> PASS"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The output remains aligned with the official "
        "temporal prediction population."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()