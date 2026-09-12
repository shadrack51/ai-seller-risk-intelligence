"""
Build Target-Specific Temporal Features

Purpose
-------
Create focused feature sets for two separate early-warning
prediction tasks:

    1. Future delivery-risk prediction
    2. Future review-risk prediction

The source dataset contains many candidate features.
This script creates smaller, business-focused feature sets.

Design goals:
    - Keep strong recent-behavior signals
    - Keep meaningful trend features
    - Avoid unnecessary duplicate representations
    - Keep feature groups interpretable
    - Prevent future information from entering predictors

Inputs
------
temporal_delivery_ml_dataset.csv
temporal_review_ml_dataset.csv

Outputs
-------
target_delivery_features.csv
target_review_features.csv
target_specific_feature_lineage.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

DELIVERY_INPUT = (
    DATA_DIR
    / "temporal_delivery_ml_dataset.csv"
)

REVIEW_INPUT = (
    DATA_DIR
    / "temporal_review_ml_dataset.csv"
)

DELIVERY_OUTPUT = (
    DATA_DIR
    / "target_delivery_features.csv"
)

REVIEW_OUTPUT = (
    DATA_DIR
    / "target_review_features.csv"
)

LINEAGE_OUTPUT = (
    DATA_DIR
    / "target_specific_feature_lineage.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

DELIVERY_TARGET = "future_delivery_risk"

REVIEW_TARGET = "future_review_risk"


# ============================================================
# GENERAL FEATURE SET
# ============================================================

GENERAL_FEATURES = [
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
]


# ============================================================
# DELIVERY FEATURES
# ============================================================

DELIVERY_CORE_FEATURES = [
    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",
    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",
    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",
    "recent_delivered_orders",
    "older_delivered_orders",
    "recent_late_orders",
    "older_late_orders",
    "recent_delivery_activity",
    "older_delivery_activity",
    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",
]


# ============================================================
# REVIEW FEATURES
# ============================================================

REVIEW_CORE_FEATURES = [
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


# ============================================================
# FEATURES TO DROP AS REDUNDANT
# ============================================================

REDUNDANT_FEATURES = [
    "recent_orders_per_day",
    "older_orders_per_day",
    "recent_revenue_per_day",
    "older_revenue_per_day",
]


# ============================================================
# FUTURE / FORBIDDEN COLUMNS
# ============================================================

FORBIDDEN_FEATURES = [
    "future_at_risk",
    "future_order_count",
    "future_delivered_orders",
    "future_late_delivery_rate",
    "future_average_review_score",
    "reconstructed_future_at_risk",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_columns(
    df,
    columns,
    label,
):
    """Ensure all required columns exist."""

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {label} columns:\n"
            + "\n".join(missing)
        )


def validate_target(
    df,
    target,
    label,
):
    """Validate a binary prediction target."""

    if target not in df.columns:
        raise ValueError(
            f"{label} target missing: {target}"
        )

    missing = (
        df[target]
        .isna()
        .sum()
    )

    if missing > 0:
        raise ValueError(
            f"{label} target contains "
            f"{missing} missing values."
        )

    invalid = (
        set(
            df[target].unique()
        )
        - {0, 1}
    )

    if invalid:
        raise ValueError(
            f"{label} target contains invalid "
            f"values: {invalid}"
        )


def validate_no_future_features(
    predictors,
    label,
):
    """Ensure future outcome columns are excluded."""

    leakage = [
        feature
        for feature in predictors
        if feature in FORBIDDEN_FEATURES
        or feature.startswith("future_")
    ]

    if leakage:
        raise ValueError(
            f"{label} contains future information:\n"
            + "\n".join(leakage)
        )


def validate_unique_keys(
    df,
    label,
):
    """Ensure one seller has one row per snapshot."""

    duplicates = (
        df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        f"{label} duplicate seller/snapshot rows: "
        f"{duplicates}"
    )

    if duplicates > 0:
        raise ValueError(
            f"{label} contains duplicate "
            "seller/snapshot observations."
        )


def build_lineage(feature_list, feature_group, target):
    """
    Build a feature lineage table.

    Lineage answers:
        Where did this feature come from?
        Why is it present?
        Which prediction task uses it?
    """

    rows = []

    for feature in feature_list:

        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group.get(
                    feature,
                    "General",
                ),
                "target": target,
            }
        )

    return pd.DataFrame(rows)


def print_target_summary(
    df,
    target,
    label,
):
    """Print target distribution."""

    counts = (
        df[target]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[target]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)

    print(
        f"\n{label} counts:"
    )

    print(
        counts
    )

    print(
        f"\n{label} percentages:"
    )

    print(
        percentages
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TARGET-SPECIFIC TEMPORAL FEATURES")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING RISK-SPECIFIC DATASETS"
    )

    if not DELIVERY_INPUT.exists():
        raise FileNotFoundError(
            f"Delivery dataset not found:\n"
            f"{DELIVERY_INPUT}"
        )

    if not REVIEW_INPUT.exists():
        raise FileNotFoundError(
            f"Review dataset not found:\n"
            f"{REVIEW_INPUT}"
        )

    delivery_source = pd.read_csv(
        DELIVERY_INPUT
    )

    review_source = pd.read_csv(
        REVIEW_INPUT
    )

    print(
        f"Delivery source rows: "
        f"{len(delivery_source):,}"
    )

    print(
        f"Review source rows: "
        f"{len(review_source):,}"
    )

    # ========================================================
    # 2. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING TEMPORAL KEYS"
    )

    for df in [
        delivery_source,
        review_source,
    ]:
        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce",
        )

    if (
        delivery_source[DATE_COLUMN]
        .isna()
        .any()
    ):
        raise ValueError(
            "Invalid dates found in delivery dataset."
        )

    if (
        review_source[DATE_COLUMN]
        .isna()
        .any()
    ):
        raise ValueError(
            "Invalid dates found in review dataset."
        )

    print(
        "Temporal key conversion -> PASS"
    )

    # ========================================================
    # 3. VALIDATE SOURCE DATASETS
    # ========================================================

    print_section(
        "SOURCE DATA VALIDATION"
    )

    validate_unique_keys(
        delivery_source,
        "Delivery source",
    )

    validate_unique_keys(
        review_source,
        "Review source",
    )

    validate_target(
        delivery_source,
        DELIVERY_TARGET,
        "Delivery",
    )

    validate_target(
        review_source,
        REVIEW_TARGET,
        "Review",
    )

    print(
        "Source dataset validation -> PASS"
    )

    # ========================================================
    # 4. BUILD DELIVERY FEATURE LIST
    # ========================================================

    print_section(
        "BUILDING DELIVERY FEATURE SET"
    )

    delivery_features = (
        GENERAL_FEATURES
        + DELIVERY_CORE_FEATURES
    )

    delivery_features = list(
        dict.fromkeys(
            delivery_features
        )
    )

    delivery_features = [
        feature
        for feature in delivery_features
        if feature not in REDUNDANT_FEATURES
    ]

    validate_columns(
        delivery_source,
        delivery_features,
        "delivery feature",
    )

    validate_no_future_features(
        delivery_features,
        "Delivery feature set",
    )

    print(
        f"Delivery predictors: "
        f"{len(delivery_features)}"
    )

    for feature in delivery_features:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 5. BUILD REVIEW FEATURE LIST
    # ========================================================

    print_section(
        "BUILDING REVIEW FEATURE SET"
    )

    review_features = (
        GENERAL_FEATURES
        + REVIEW_CORE_FEATURES
    )

    review_features = list(
        dict.fromkeys(
            review_features
        )
    )

    review_features = [
        feature
        for feature in review_features
        if feature not in REDUNDANT_FEATURES
    ]

    validate_columns(
        review_source,
        review_features,
        "review feature",
    )

    validate_no_future_features(
        review_features,
        "Review feature set",
    )

    print(
        f"Review predictors: "
        f"{len(review_features)}"
    )

    for feature in review_features:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 6. FEATURE GROUP MAPPING
    # ========================================================

    print_section(
        "BUILDING FEATURE LINEAGE"
    )

    feature_groups = {}

    for feature in GENERAL_FEATURES:
        feature_groups[feature] = (
            "General seller behavior"
        )

    for feature in DELIVERY_CORE_FEATURES:
        feature_groups[feature] = (
            "Delivery behavior"
        )

    for feature in REVIEW_CORE_FEATURES:
        feature_groups[feature] = (
            "Customer review behavior"
        )

    # ========================================================
    # 7. BUILD DELIVERY DATASET
    # ========================================================

    print_section(
        "CREATING DELIVERY PREDICTION DATASET"
    )

    delivery_columns = (
        [
            SELLER_COLUMN,
            DATE_COLUMN,
        ]
        + delivery_features
        + [
            DELIVERY_TARGET,
        ]
    )

    delivery_df = delivery_source[
        delivery_columns
    ].copy()

    print(
        f"Rows: "
        f"{len(delivery_df):,}"
    )

    print(
        f"Predictors: "
        f"{len(delivery_features)}"
    )

    print_target_summary(
        delivery_df,
        DELIVERY_TARGET,
        "Delivery-risk target",
    )

    # ========================================================
    # 8. BUILD REVIEW DATASET
    # ========================================================

    print_section(
        "CREATING REVIEW PREDICTION DATASET"
    )

    review_columns = (
        [
            SELLER_COLUMN,
            DATE_COLUMN,
        ]
        + review_features
        + [
            REVIEW_TARGET,
        ]
    )

    review_df = review_source[
        review_columns
    ].copy()

    print(
        f"Rows: "
        f"{len(review_df):,}"
    )

    print(
        f"Predictors: "
        f"{len(review_features)}"
    )

    print_target_summary(
        review_df,
        REVIEW_TARGET,
        "Review-risk target",
    )

    # ========================================================
    # 9. NUMERIC VALIDATION
    # ========================================================

    print_section(
        "NUMERIC FEATURE VALIDATION"
    )

    for label, df, features in [
        (
            "Delivery",
            delivery_df,
            delivery_features,
        ),
        (
            "Review",
            review_df,
            review_features,
        ),
    ]:

        non_numeric = (
            df[features]
            .select_dtypes(
                exclude="number"
            )
            .columns
            .tolist()
        )

        if non_numeric:
            raise ValueError(
                f"{label} has non-numeric predictors:\n"
                + "\n".join(non_numeric)
            )

        print(
            f"{label} predictors numeric -> PASS"
        )

    # ========================================================
    # 10. MISSING VALUE SUMMARY
    # ========================================================

    print_section(
        "MISSING VALUE SUMMARY"
    )

    for label, df, features in [
        (
            "Delivery",
            delivery_df,
            delivery_features,
        ),
        (
            "Review",
            review_df,
            review_features,
        ),
    ]:

        missing = (
            df[features]
            .isna()
            .sum()
        )

        missing = (
            missing[
                missing > 0
            ]
            .sort_values(
                ascending=False
            )
        )

        print(
            f"\n{label}:"
        )

        if missing.empty:

            print(
                "No missing predictor values."
            )

        else:

            print(
                missing.to_string()
            )

            print(
                "Missing values will be handled "
                "inside the training pipeline."
            )

    # ========================================================
    # 11. TARGET CORRELATION SCREEN
    # ========================================================

    print_section(
        "DELIVERY FEATURE / TARGET SCREEN"
    )

    delivery_correlations = (
        delivery_df[
            delivery_features
            + [DELIVERY_TARGET]
        ]
        .corr(
            numeric_only=True
        )[DELIVERY_TARGET]
        .drop(
            DELIVERY_TARGET
        )
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    print(
        delivery_correlations
        .round(4)
        .to_string()
    )

    print_section(
        "REVIEW FEATURE / TARGET SCREEN"
    )

    review_correlations = (
        review_df[
            review_features
            + [REVIEW_TARGET]
        ]
        .corr(
            numeric_only=True
        )[REVIEW_TARGET]
        .drop(
            REVIEW_TARGET
        )
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    print(
        review_correlations
        .round(4)
        .to_string()
    )

    # ========================================================
    # 12. BUILD LINEAGE TABLE
    # ========================================================

    delivery_lineage = build_lineage(
        delivery_features,
        feature_groups,
        DELIVERY_TARGET,
    )

    delivery_lineage[
        "dataset"
    ] = "delivery"

    review_lineage = build_lineage(
        review_features,
        feature_groups,
        REVIEW_TARGET,
    )

    review_lineage[
        "dataset"
    ] = "review"

    lineage_df = pd.concat(
        [
            delivery_lineage,
            review_lineage,
        ],
        ignore_index=True,
    )

    # ========================================================
    # 13. FINAL VALIDATION
    # ========================================================

    print_section(
        "FINAL FEATURE VALIDATION"
    )

    print(
        "Delivery future leakage -> PASS"
    )

    print(
        "Review future leakage -> PASS"
    )

    print(
        "Seller identifiers excluded from predictors -> PASS"
    )

    print(
        "Temporal keys preserved -> PASS"
    )

    print(
        "Target columns preserved -> PASS"
    )

    # ========================================================
    # 14. SAVE DATASETS
    # ========================================================

    print_section(
        "SAVING TARGET-SPECIFIC DATASETS"
    )

    delivery_df.to_csv(
        DELIVERY_OUTPUT,
        index=False,
    )

    review_df.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    lineage_df.to_csv(
        LINEAGE_OUTPUT,
        index=False,
    )

    print(
        f"Delivery features saved to:\n"
        f"{DELIVERY_OUTPUT}"
    )

    print(
        f"\nReview features saved to:\n"
        f"{REVIEW_OUTPUT}"
    )

    print(
        f"\nFeature lineage saved to:\n"
        f"{LINEAGE_OUTPUT}"
    )

    # ========================================================
    # 15. FINAL SUMMARY
    # ========================================================

    print_section(
        "TARGET-SPECIFIC FEATURE ENGINEERING COMPLETE"
    )

    print(
        f"Delivery observations: "
        f"{len(delivery_df):,}"
    )

    print(
        f"Delivery predictors: "
        f"{len(delivery_features)}"
    )

    print(
        f"Review observations: "
        f"{len(review_df):,}"
    )

    print(
        f"Review predictors: "
        f"{len(review_features)}"
    )

    print(
        f"Lineage records: "
        f"{len(lineage_df):,}"
    )

    print(
        "\nThe two future-risk mechanisms now have "
        "separate, business-focused feature sets."
    )

    print(
        "\nNo model was trained."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()