"""
Build Delivery Context Features

Purpose
-------
Create additional point-in-time delivery features using
order-level shipping context.

Prediction unit
----------------
seller_id + snapshot_date

Target
------
future_delivery_risk

Historical windows
------------------
Recent:
    30 days before snapshot

Older:
    60 days before the recent window

Features include:

    - order workload
    - revenue workload
    - freight workload
    - units sold
    - cross-state shipping rate
    - destination-state diversity
    - destination concentration
    - freight-to-revenue ratio
    - order-value median
    - order-value variability
    - freight variability
    - items per order

Change features compare recent and older windows.

Important
---------
Seller ID is obtained from order_items because the Olist
orders table does not contain seller_id.

Future target information is never used to construct predictors.

Input
-----
data/raw/olist_orders_dataset.csv
data/raw/olist_order_items_dataset.csv
data/raw/olist_customers_dataset.csv
data/raw/olist_sellers_dataset.csv

data/processed/target_delivery_features.csv

Output
------
data/processed/delivery_context_features.csv
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

CUSTOMERS_FILE = (
    RAW_DATA_DIR
    / "olist_customers_dataset.csv"
)

SELLERS_FILE = (
    RAW_DATA_DIR
    / "olist_sellers_dataset.csv"
)

BASE_FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "target_delivery_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "delivery_context_features.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

SNAPSHOT_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_delivery_risk"


# ============================================================
# HISTORICAL WINDOW CONFIGURATION
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


def validate_file(
    path,
    label,
):
    """Confirm that an input file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def validate_unique_keys(
    df,
    label,
    keys,
):
    """Validate uniqueness of specified keys."""

    duplicate_count = (
        df[keys]
        .duplicated()
        .sum()
    )

    print(
        f"{label} duplicate key rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise ValueError(
            f"{label} contains duplicate keys: "
            f"{keys}"
        )


def validate_required_columns(
    df,
    required_columns,
    label,
):
    """Validate that required columns exist."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{label} is missing columns:\n"
            + "\n".join(
                missing
            )
        )


def build_order_item_summary(
    order_items,
):
    """
    Aggregate item-level information by order and seller.

    This is where seller_id enters the order-level context table.
    """

    return (
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
            order_freight=(
                "freight_value",
                "sum",
            ),
            item_count=(
                "product_id",
                "count",
            ),
        )
    )


def build_order_level_context(
    orders,
    order_items,
    customers,
    sellers,
):
    """
    Build one row per order/seller combination.

    Correct relational order:

        orders
          +
        customers
          ↓
        customer context

        +
        order_items aggregated by order_id + seller_id
          ↓
        seller context enters the table

        +
        seller master
          ↓
        final order-level context
    """

    # --------------------------------------------------------
    # STEP 1
    # Aggregate order items by order and seller.
    # --------------------------------------------------------

    order_item_summary = (
        build_order_item_summary(
            order_items
        )
    )

    # --------------------------------------------------------
    # STEP 2
    # Attach customer information to orders.
    # --------------------------------------------------------

    customer_subset = (
        customers[
            [
                "customer_id",
                "customer_state",
                "customer_zip_code_prefix",
            ]
        ]
        .drop_duplicates(
            "customer_id"
        )
    )

    orders_with_customer = (
        orders[
            [
                "order_id",
                "customer_id",
                "order_purchase_timestamp",
            ]
        ]
        .merge(
            customer_subset,
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # STEP 3
    # IMPORTANT:
    #
    # seller_id does NOT exist in the orders table.
    #
    # It enters through order_items.
    # --------------------------------------------------------

    order_context = (
        orders_with_customer
        .merge(
            order_item_summary,
            on="order_id",
            how="inner",
        )
    )

    # --------------------------------------------------------
    # STEP 4
    # Validate seller_id now exists.
    # --------------------------------------------------------

    if SELLER_COLUMN not in order_context.columns:

        raise ValueError(
            "seller_id was not introduced by the "
            "order-items merge."
        )

    # --------------------------------------------------------
    # STEP 5
    # Attach seller master information.
    # --------------------------------------------------------

    seller_subset = (
        sellers[
            [
                SELLER_COLUMN,
                "seller_state",
                "seller_zip_code_prefix",
            ]
        ]
        .drop_duplicates(
            SELLER_COLUMN
        )
    )

    order_context = (
        order_context
        .merge(
            seller_subset,
            on=SELLER_COLUMN,
            how="inner",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # STEP 6
    # Determine cross-state shipping.
    # --------------------------------------------------------

    order_context[
        "cross_state"
    ] = (
        order_context[
            "seller_state"
        ]
        !=
        order_context[
            "customer_state"
        ]
    ).astype(int)

    # --------------------------------------------------------
    # STEP 7
    # Freight burden relative to revenue.
    # --------------------------------------------------------

    order_context[
        "freight_to_revenue_ratio"
    ] = np.where(
        order_context[
            "order_revenue"
        ] > 0,

        order_context[
            "order_freight"
        ]
        /
        order_context[
            "order_revenue"
        ],

        np.nan,
    )

    return order_context


def calculate_context_window(
    order_context,
    start_date,
    end_date,
):
    """
    Aggregate historical orders into seller-level context features.
    """

    window = order_context[
        (
            order_context[
                "order_purchase_timestamp"
            ]
            > start_date
        )
        &
        (
            order_context[
                "order_purchase_timestamp"
            ]
            <= end_date
        )
    ].copy()

    if window.empty:

        return pd.DataFrame(
            columns=[
                SELLER_COLUMN,
                "context_order_count",
                "context_revenue",
                "context_freight",
                "context_units",
                "context_cross_state_rate",
                "context_unique_destination_states",
                "context_destination_concentration",
                "context_freight_to_revenue",
                "context_revenue_median",
                "context_revenue_std",
                "context_freight_median",
                "context_freight_std",
                "context_items_per_order",
            ]
        )

    rows = []

    for seller_id, seller_orders in (
        window
        .groupby(
            SELLER_COLUMN
        )
    ):

        # ----------------------------------------------------
        # Basic workload
        # ----------------------------------------------------

        order_count = len(
            seller_orders
        )

        revenue = (
            seller_orders[
                "order_revenue"
            ]
            .sum()
        )

        freight = (
            seller_orders[
                "order_freight"
            ]
            .sum()
        )

        units = (
            seller_orders[
                "item_count"
            ]
            .sum()
        )

        # ----------------------------------------------------
        # Cross-state shipping
        # ----------------------------------------------------

        cross_state_rate = (
            seller_orders[
                "cross_state"
            ]
            .mean()
        )

        # ----------------------------------------------------
        # Destination diversity
        # ----------------------------------------------------

        unique_destination_states = (
            seller_orders[
                "customer_state"
            ]
            .nunique(
                dropna=True
            )
        )

        # ----------------------------------------------------
        # Destination concentration
        #
        # Largest share of orders going to any one
        # destination state.
        # ----------------------------------------------------

        destination_distribution = (
            seller_orders[
                "customer_state"
            ]
            .value_counts(
                normalize=True
            )
        )

        if destination_distribution.empty:

            destination_concentration = (
                np.nan
            )

        else:

            destination_concentration = (
                destination_distribution
                .max()
            )

        # ----------------------------------------------------
        # Freight burden
        # ----------------------------------------------------

        if revenue > 0:

            freight_to_revenue = (
                freight
                /
                revenue
            )

        else:

            freight_to_revenue = np.nan

        # ----------------------------------------------------
        # Order-value distribution
        # ----------------------------------------------------

        revenue_median = (
            seller_orders[
                "order_revenue"
            ]
            .median()
        )

        revenue_std = (
            seller_orders[
                "order_revenue"
            ]
            .std()
        )

        # ----------------------------------------------------
        # Freight distribution
        # ----------------------------------------------------

        freight_median = (
            seller_orders[
                "order_freight"
            ]
            .median()
        )

        freight_std = (
            seller_orders[
                "order_freight"
            ]
            .std()
        )

        # ----------------------------------------------------
        # Items per order
        # ----------------------------------------------------

        if order_count > 0:

            items_per_order = (
                units
                /
                order_count
            )

        else:

            items_per_order = np.nan

        rows.append(
            {
                SELLER_COLUMN: seller_id,

                "context_order_count":
                    order_count,

                "context_revenue":
                    revenue,

                "context_freight":
                    freight,

                "context_units":
                    units,

                "context_cross_state_rate":
                    cross_state_rate,

                "context_unique_destination_states":
                    unique_destination_states,

                "context_destination_concentration":
                    destination_concentration,

                "context_freight_to_revenue":
                    freight_to_revenue,

                "context_revenue_median":
                    revenue_median,

                "context_revenue_std":
                    revenue_std,

                "context_freight_median":
                    freight_median,

                "context_freight_std":
                    freight_std,

                "context_items_per_order":
                    items_per_order,
            }
        )

    return pd.DataFrame(
        rows
    )


def prefix_columns(
    df,
    prefix,
):
    """
    Prefix feature columns while preserving seller_id.
    """

    if df.empty:

        return df

    renamed = df.copy()

    columns_to_prefix = [
        column
        for column in renamed.columns
        if column != SELLER_COLUMN
    ]

    renamed = renamed.rename(
        columns={
            column:
                f"{prefix}{column}"
            for column in columns_to_prefix
        }
    )

    return renamed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILDING DELIVERY CONTEXT FEATURES"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK FILES
    # ========================================================

    print_section(
        "CHECKING INPUT DATASETS"
    )

    required_files = [
        (
            ORDERS_FILE,
            "Orders dataset",
        ),
        (
            ORDER_ITEMS_FILE,
            "Order items dataset",
        ),
        (
            CUSTOMERS_FILE,
            "Customers dataset",
        ),
        (
            SELLERS_FILE,
            "Sellers dataset",
        ),
        (
            BASE_FEATURE_FILE,
            "Base delivery feature dataset",
        ),
    ]

    for path, label in required_files:

        validate_file(
            path,
            label,
        )

        print(
            f"{label} -> FOUND"
        )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print_section(
        "LOADING RAW DATA"
    )

    orders = pd.read_csv(
        ORDERS_FILE
    )

    order_items = pd.read_csv(
        ORDER_ITEMS_FILE
    )

    customers = pd.read_csv(
        CUSTOMERS_FILE
    )

    sellers = pd.read_csv(
        SELLERS_FILE
    )

    base_features = pd.read_csv(
        BASE_FEATURE_FILE
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
        f"Customers: "
        f"{len(customers):,}"
    )

    print(
        f"Sellers: "
        f"{len(sellers):,}"
    )

    print(
        f"Base prediction observations: "
        f"{len(base_features):,}"
    )

    # ========================================================
    # 3. SOURCE COLUMN VALIDATION
    # ========================================================

    print_section(
        "VALIDATING SOURCE COLUMNS"
    )

    validate_required_columns(
        orders,
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        ],
        "Orders dataset",
    )

    validate_required_columns(
        order_items,
        [
            "order_id",
            "seller_id",
            "price",
            "freight_value",
            "product_id",
        ],
        "Order items dataset",
    )

    validate_required_columns(
        customers,
        [
            "customer_id",
            "customer_state",
            "customer_zip_code_prefix",
        ],
        "Customers dataset",
    )

    validate_required_columns(
        sellers,
        [
            "seller_id",
            "seller_state",
            "seller_zip_code_prefix",
        ],
        "Sellers dataset",
    )

    validate_required_columns(
        base_features,
        [
            SELLER_COLUMN,
            SNAPSHOT_COLUMN,
            TARGET_COLUMN,
        ],
        "Base delivery feature dataset",
    )

    print(
        "Orders columns -> PASS"
    )

    print(
        "Order-item columns -> PASS"
    )

    print(
        "Customer columns -> PASS"
    )

    print(
        "Seller columns -> PASS"
    )

    print(
        "Base feature columns -> PASS"
    )

    # ========================================================
    # 4. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING DATES"
    )

    orders[
        "order_purchase_timestamp"
    ] = pd.to_datetime(
        orders[
            "order_purchase_timestamp"
        ],
        errors="coerce",
    )

    base_features[
        SNAPSHOT_COLUMN
    ] = pd.to_datetime(
        base_features[
            SNAPSHOT_COLUMN
        ],
        errors="coerce",
    )

    invalid_order_dates = (
        orders[
            "order_purchase_timestamp"
        ]
        .isna()
        .sum()
    )

    invalid_snapshot_dates = (
        base_features[
            SNAPSHOT_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"Invalid order purchase dates: "
        f"{invalid_order_dates}"
    )

    print(
        f"Invalid snapshot dates: "
        f"{invalid_snapshot_dates}"
    )

    if invalid_order_dates > 0:

        raise ValueError(
            "Invalid order purchase timestamps found."
        )

    if invalid_snapshot_dates > 0:

        raise ValueError(
            "Invalid snapshot dates found."
        )

    print(
        "Order dates -> PASS"
    )

    print(
        "Snapshot dates -> PASS"
    )

    # ========================================================
    # 5. SOURCE KEY VALIDATION
    # ========================================================

    print_section(
        "VALIDATING SOURCE KEYS"
    )

    validate_unique_keys(
        customers,
        "Customers",
        ["customer_id"],
    )

    validate_unique_keys(
        sellers,
        "Sellers",
        [SELLER_COLUMN],
    )

    validate_unique_keys(
        base_features,
        "Base feature dataset",
        [
            SELLER_COLUMN,
            SNAPSHOT_COLUMN,
        ],
    )

    print(
        "Source key validation -> PASS"
    )

    # ========================================================
    # 6. BUILD ORDER-LEVEL CONTEXT
    # ========================================================

    print_section(
        "BUILDING ORDER-LEVEL SHIPPING CONTEXT"
    )

    print(
        "Seller ID source: order_items"
    )

    print(
        "Customer context source: orders + customers"
    )

    print(
        "Seller master source: sellers"
    )

    order_context = (
        build_order_level_context(
            orders,
            order_items,
            customers,
            sellers,
        )
    )

    print(
        f"Order-seller-context rows: "
        f"{len(order_context):,}"
    )

    print(
        f"Unique sellers: "
        f"{order_context[SELLER_COLUMN].nunique():,}"
    )

    print(
        "Order-level context created -> PASS"
    )

    # ========================================================
    # 7. IDENTIFY MASTER POPULATION
    # ========================================================

    print_section(
        "IDENTIFYING MASTER PREDICTION POPULATION"
    )

    master_keys = (
        base_features[
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
        f"Master seller/snapshot observations: "
        f"{len(master_keys):,}"
    )

    print(
        f"Snapshots: "
        f"{master_keys[SNAPSHOT_COLUMN].nunique():,}"
    )

    # ========================================================
    # 8. CALCULATE CONTEXT FEATURES
    # ========================================================

    print_section(
        "CALCULATING CONTEXT FEATURES"
    )

    snapshot_dates = sorted(
        master_keys[
            SNAPSHOT_COLUMN
        ]
        .unique()
    )

    feature_rows = []

    for number, snapshot_date in enumerate(
        snapshot_dates,
        start=1,
    ):

        print(
            f"{number:02d}/"
            f"{len(snapshot_dates):02d} | "
            f"Snapshot: "
            f"{snapshot_date.date()}"
        )

        recent_start = (
            snapshot_date
            -
            pd.Timedelta(
                days=RECENT_DAYS
            )
        )

        older_start = (
            snapshot_date
            -
            pd.Timedelta(
                days=RECENT_DAYS
                +
                OLDER_DAYS
            )
        )

        recent = (
            calculate_context_window(
                order_context,
                recent_start,
                snapshot_date,
            )
        )

        older = (
            calculate_context_window(
                order_context,
                older_start,
                recent_start,
            )
        )

        recent = prefix_columns(
            recent,
            "recent_",
        )

        older = prefix_columns(
            older,
            "older_",
        )

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

        if not recent.empty:

            snapshot_master = (
                snapshot_master
                .merge(
                    recent,
                    on=SELLER_COLUMN,
                    how="left",
                    validate="one_to_one",
                )
            )

        if not older.empty:

            snapshot_master = (
                snapshot_master
                .merge(
                    older,
                    on=SELLER_COLUMN,
                    how="left",
                    validate="one_to_one",
                )
            )

        snapshot_master[
            SNAPSHOT_COLUMN
        ] = snapshot_date

        feature_rows.append(
            snapshot_master
        )

    # ========================================================
    # 9. COMBINE
    # ========================================================

    print_section(
        "COMBINING CONTEXT FEATURES"
    )

    context_df = pd.concat(
        feature_rows,
        ignore_index=True,
    )

    print(
        f"Context observations: "
        f"{len(context_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{context_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{context_df[SNAPSHOT_COLUMN].nunique():,}"
    )

    # ========================================================
    # 10. POPULATION ALIGNMENT
    # ========================================================

    print_section(
        "MASTER POPULATION ALIGNMENT"
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

    context_key_set = set(
        zip(
            context_df[
                SELLER_COLUMN
            ],
            context_df[
                SNAPSHOT_COLUMN
            ],
        )
    )

    extra_keys = (
        context_key_set
        -
        master_key_set
    )

    missing_keys = (
        master_key_set
        -
        context_key_set
    )

    print(
        f"Master keys: "
        f"{len(master_key_set):,}"
    )

    print(
        f"Context keys: "
        f"{len(context_key_set):,}"
    )

    print(
        f"Extra context keys: "
        f"{len(extra_keys):,}"
    )

    print(
        f"Missing context keys: "
        f"{len(missing_keys):,}"
    )

    if extra_keys:

        raise ValueError(
            "Context features contain observations "
            "outside the master population."
        )

    if missing_keys:

        raise ValueError(
            "Context features are missing observations "
            "from the master population."
        )

    print(
        "Population alignment -> PASS"
    )

    # ========================================================
    # 11. CONTEXT CHANGE FEATURES
    # ========================================================

    print_section(
        "CREATING CONTEXT CHANGE FEATURES"
    )

    context_change_pairs = [
        (
            "recent_context_order_count",
            "older_context_order_count",
            "context_order_count_change",
        ),
        (
            "recent_context_revenue",
            "older_context_revenue",
            "context_revenue_change",
        ),
        (
            "recent_context_freight",
            "older_context_freight",
            "context_freight_change",
        ),
        (
            "recent_context_units",
            "older_context_units",
            "context_units_change",
        ),
        (
            "recent_context_cross_state_rate",
            "older_context_cross_state_rate",
            "context_cross_state_rate_change",
        ),
        (
            "recent_context_unique_destination_states",
            "older_context_unique_destination_states",
            "context_destination_diversity_change",
        ),
        (
            "recent_context_destination_concentration",
            "older_context_destination_concentration",
            "context_destination_concentration_change",
        ),
        (
            "recent_context_freight_to_revenue",
            "older_context_freight_to_revenue",
            "context_freight_to_revenue_change",
        ),
        (
            "recent_context_revenue_std",
            "older_context_revenue_std",
            "context_revenue_std_change",
        ),
        (
            "recent_context_freight_std",
            "older_context_freight_std",
            "context_freight_std_change",
        ),
    ]

    for recent_column, older_column, new_column in (
        context_change_pairs
    ):

        context_df[
            new_column
        ] = (
            context_df[
                recent_column
            ]
            -
            context_df[
                older_column
            ]
        )

    print(
        f"Context change features created: "
        f"{len(context_change_pairs)}"
    )

    # ========================================================
    # 12. PROPORTION SANITY
    # ========================================================

    print_section(
        "CONTEXT PROPORTION VALIDATION"
    )

    proportion_columns = [
        "recent_context_cross_state_rate",
        "older_context_cross_state_rate",
        "recent_context_destination_concentration",
        "older_context_destination_concentration",
    ]

    invalid_proportion_count = 0

    for column in proportion_columns:

        valid = context_df[
            column
        ].dropna()

        below_zero = (
            valid
            .lt(0)
            .sum()
        )

        above_one = (
            valid
            .gt(1)
            .sum()
        )

        print(
            f"{column}: "
            f"<0 = {below_zero}, "
            f">1 = {above_one}"
        )

        invalid_proportion_count += (
            below_zero
            +
            above_one
        )

    if invalid_proportion_count > 0:

        raise ValueError(
            "Invalid proportion values detected."
        )

    print(
        "Proportion validation -> PASS"
    )

    # ========================================================
    # 13. TARGET PRESERVATION
    # ========================================================

    print_section(
        "TARGET PRESERVATION"
    )

    target_reference = (
        base_features[
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

    context_with_target = (
        context_df
        .merge(
            target_reference,
            on=[
                SELLER_COLUMN,
                SNAPSHOT_COLUMN,
            ],
            how="left",
            validate="one_to_one",
        )
    )

    target_sorted = (
        context_with_target[
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
        target_reference.equals(
            target_sorted
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
    # 14. FUTURE INFORMATION CHECK
    # ========================================================

    print_section(
        "FUTURE INFORMATION CHECK"
    )

    predictor_candidates = [
        column
        for column in context_with_target.columns
        if column
        not in [
            SELLER_COLUMN,
            SNAPSHOT_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    forbidden_future_columns = [
        column
        for column in predictor_candidates
        if (
            column.startswith(
                "future_"
            )
            or column in {
                "future_at_risk",
                "future_delivery_risk",
                "future_review_risk",
                "future_late_delivery_rate",
                "future_average_review_score",
                "future_order_count",
                "future_delivered_orders",
            }
        )
    ]

    if forbidden_future_columns:

        raise ValueError(
            "Future information found among predictors:\n"
            +
            "\n".join(
                forbidden_future_columns
            )
        )

    print(
        "No future columns in context predictors -> PASS"
    )

    # ========================================================
    # 15. NUMERIC VALIDATION
    # ========================================================

    print_section(
        "FEATURE TYPE VALIDATION"
    )

    context_features = [
        column
        for column in context_with_target.columns
        if column
        not in [
            SELLER_COLUMN,
            SNAPSHOT_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    non_numeric = (
        context_with_target[
            context_features
        ]
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric:

        raise ValueError(
            "Non-numeric context features found:\n"
            +
            "\n".join(
                non_numeric
            )
        )

    print(
        "All context predictors numeric -> PASS"
    )

    # ========================================================
    # 16. MISSING VALUES
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing = (
        context_with_target[
            context_features
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
            "No missing context features."
        )

    else:

        print(
            missing.to_string()
        )

        print(
            "\nMissing values are expected when "
            "a seller has insufficient historical "
            "order observations."
        )

    # ========================================================
    # 17. FEATURE INVENTORY
    # ========================================================

    print_section(
        "CONTEXT FEATURE INVENTORY"
    )

    print(
        f"Context predictors: "
        f"{len(context_features)}"
    )

    for number, feature in enumerate(
        context_features,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 18. PREVIEW
    # ========================================================

    print_section(
        "CONTEXT FEATURE PREVIEW"
    )

    preview_columns = [
        SELLER_COLUMN,
        SNAPSHOT_COLUMN,

        "recent_context_order_count",
        "recent_context_cross_state_rate",
        "recent_context_unique_destination_states",
        "recent_context_destination_concentration",
        "recent_context_freight_to_revenue",

        "context_order_count_change",
        "context_cross_state_rate_change",
        "context_destination_diversity_change",
        "context_destination_concentration_change",

        TARGET_COLUMN,
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column
        in context_with_target.columns
    ]

    print(
        context_with_target[
            preview_columns
        ]
        .head(10)
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 19. SAVE
    # ========================================================

    print_section(
        "SAVING DELIVERY CONTEXT FEATURES"
    )

    final_df = (
        context_with_target
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

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Delivery context features saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 20. FINAL SUMMARY
    # ========================================================

    print_section(
        "DELIVERY CONTEXT FEATURES COMPLETE"
    )

    print(
        f"Rows: "
        f"{len(final_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{final_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{final_df[SNAPSHOT_COLUMN].nunique():,}"
    )

    print(
        f"Context predictors: "
        f"{len(context_features):,}"
    )

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print(
        "\nPopulation alignment -> PASS"
    )

    print(
        "Target preservation -> PASS"
    )

    print(
        "Future information check -> PASS"
    )

    print(
        "Numeric validation -> PASS"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The context feature layer is ready for audit."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()