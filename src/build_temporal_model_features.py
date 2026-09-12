"""
Build Curated Temporal ML Features

Purpose
-------
Create a cleaner, production-oriented feature set from the
50 temporal trend features.

The previous audit showed that some percentage-change features
are unstable when the previous-period denominator is very small.

This script therefore:

1. Selects a controlled feature whitelist.
2. Retains stable recent and historical measures.
3. Retains absolute change / trend features.
4. Adds explicit activity indicators.
5. Adds history-size indicators.
6. Excludes future target information.
7. Validates the resulting feature set.

This script DOES NOT train a model.

Input:
    data/processed/temporal_trend_features.csv

Output:
    data/processed/temporal_model_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_trend_features.csv"
)

OUTPUT_FILE = (
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
# CURATED FEATURE GROUPS
# ============================================================

# ------------------------------------------------------------
# Activity
# ------------------------------------------------------------

ACTIVITY_FEATURES = [
    "recent_orders",
    "older_orders",
    "orders_change",
    "recent_units_sold",
    "older_units_sold",
    "units_change",
    "recent_revenue",
    "older_revenue",
    "revenue_change",
]


# ------------------------------------------------------------
# Pricing and freight
# ------------------------------------------------------------

COMMERCIAL_FEATURES = [
    "recent_average_item_price",
    "older_average_item_price",
    "recent_average_freight",
    "older_average_freight",
]


# ------------------------------------------------------------
# Delivery
# ------------------------------------------------------------

DELIVERY_FEATURES = [
    "recent_delivered_orders",
    "older_delivered_orders",
    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",
    "recent_average_estimated_delivery_days",
    "older_average_estimated_delivery_days",
    "estimated_delivery_change",
    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",
    "recent_late_orders",
    "older_late_orders",
    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",
]


# ------------------------------------------------------------
# Reviews
# ------------------------------------------------------------

REVIEW_FEATURES = [
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
]


# ------------------------------------------------------------
# Stable activity intensity / history
# ------------------------------------------------------------

HISTORY_FEATURES = [
    "recent_orders_per_day",
    "older_orders_per_day",
    "recent_revenue_per_day",
    "older_revenue_per_day",
    "seller_history_orders",
    "seller_history_reviews",
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


def validate_feature_group(
    df,
    features,
    group_name,
):
    """
    Validate that every feature in a feature group exists.
    """

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing {group_name} features:\n"
            + "\n".join(missing)
        )


def add_activity_indicators(df):
    """
    Add explicit binary indicators for recent and historical
    activity.
    """

    df["recent_order_activity"] = (
        df["recent_orders"] > 0
    ).astype(int)

    df["older_order_activity"] = (
        df["older_orders"] > 0
    ).astype(int)

    df["recent_review_activity"] = (
        df["recent_review_count"] > 0
    ).astype(int)

    df["older_review_activity"] = (
        df["older_review_count"] > 0
    ).astype(int)

    df["recent_delivery_activity"] = (
        df["recent_delivered_orders"] > 0
    ).astype(int)

    df["older_delivery_activity"] = (
        df["older_delivered_orders"] > 0
    ).astype(int)

    return df


def add_history_indicators(df):
    """
    Add explicit indicators describing the amount of
    historical information available.
    """

    df["low_history_orders"] = (
        df["seller_history_orders"] <= 5
    ).astype(int)

    df["low_history_reviews"] = (
        df["seller_history_reviews"] <= 5
    ).astype(int)

    df["sufficient_order_history"] = (
        df["seller_history_orders"] >= 10
    ).astype(int)

    df["sufficient_review_history"] = (
        df["seller_history_reviews"] >= 10
    ).astype(int)

    return df


def add_direction_indicators(df):
    """
    Add interpretable directional indicators.

    Positive delivery changes indicate worsening delivery
    duration / delay.

    Negative review-score changes indicate deterioration
    in review quality.
    """

    df["delivery_deteriorating"] = (
        df["delivery_days_change"] > 0
    ).astype(int)

    df["delivery_delay_increasing"] = (
        df["delivery_delay_change"] > 0
    ).astype(int)

    df["late_rate_increasing"] = (
        df["late_delivery_rate_change"] > 0
    ).astype(int)

    df["review_score_declining"] = (
        df["review_score_change"] < 0
    ).astype(int)

    df["low_review_rate_increasing"] = (
        df["low_review_rate_change"] > 0
    ).astype(int)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING CURATED TEMPORAL ML FEATURES")
    print("=" * 70)

    # ========================================================
    # 1. LOAD
    # ========================================================

    print_section(
        "LOADING TEMPORAL TREND DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_temporal_trend_features.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # ========================================================
    # 2. PREPARE CORE COLUMNS
    # ========================================================

    print_section(
        "PREPARING CORE COLUMNS"
    )

    if SELLER_COLUMN not in df.columns:

        raise ValueError(
            f"Missing identifier column: {SELLER_COLUMN}"
        )

    if DATE_COLUMN not in df.columns:

        raise ValueError(
            f"Missing temporal column: {DATE_COLUMN}"
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates found."
        )

    print(
        "Core columns -> PASS"
    )

    # ========================================================
    # 3. VALIDATE FEATURE GROUPS
    # ========================================================

    print_section(
        "VALIDATING FEATURE GROUPS"
    )

    validate_feature_group(
        df,
        ACTIVITY_FEATURES,
        "activity",
    )

    validate_feature_group(
        df,
        COMMERCIAL_FEATURES,
        "commercial",
    )

    validate_feature_group(
        df,
        DELIVERY_FEATURES,
        "delivery",
    )

    validate_feature_group(
        df,
        REVIEW_FEATURES,
        "review",
    )

    validate_feature_group(
        df,
        HISTORY_FEATURES,
        "history",
    )

    print(
        f"Activity features: "
        f"{len(ACTIVITY_FEATURES)}"
    )

    print(
        f"Commercial features: "
        f"{len(COMMERCIAL_FEATURES)}"
    )

    print(
        f"Delivery features: "
        f"{len(DELIVERY_FEATURES)}"
    )

    print(
        f"Review features: "
        f"{len(REVIEW_FEATURES)}"
    )

    print(
        f"History features: "
        f"{len(HISTORY_FEATURES)}"
    )

    # ========================================================
    # 4. CREATE FEATURE SET
    # ========================================================

    print_section(
        "BUILDING INITIAL FEATURE SET"
    )

    selected_features = (
        ACTIVITY_FEATURES
        + COMMERCIAL_FEATURES
        + DELIVERY_FEATURES
        + REVIEW_FEATURES
        + HISTORY_FEATURES
    )

    # Remove duplicates while preserving order.

    selected_features = list(
        dict.fromkeys(
            selected_features
        )
    )

    print(
        f"Initial selected features: "
        f"{len(selected_features)}"
    )

    # ========================================================
    # 5. ADD ACTIVITY INDICATORS
    # ========================================================

    print_section(
        "ADDING ACTIVITY INDICATORS"
    )

    df = add_activity_indicators(
        df
    )

    activity_indicators = [
        "recent_order_activity",
        "older_order_activity",
        "recent_review_activity",
        "older_review_activity",
        "recent_delivery_activity",
        "older_delivery_activity",
    ]

    selected_features += (
        activity_indicators
    )

    print(
        "Activity indicators added:"
    )

    for feature in activity_indicators:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 6. ADD HISTORY INDICATORS
    # ========================================================

    print_section(
        "ADDING HISTORY INDICATORS"
    )

    df = add_history_indicators(
        df
    )

    history_indicators = [
        "low_history_orders",
        "low_history_reviews",
        "sufficient_order_history",
        "sufficient_review_history",
    ]

    selected_features += (
        history_indicators
    )

    print(
        "History indicators added:"
    )

    for feature in history_indicators:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 7. ADD DIRECTION INDICATORS
    # ========================================================

    print_section(
        "ADDING TREND DIRECTION INDICATORS"
    )

    df = add_direction_indicators(
        df
    )

    direction_indicators = [
        "delivery_deteriorating",
        "delivery_delay_increasing",
        "late_rate_increasing",
        "review_score_declining",
        "low_review_rate_increasing",
    ]

    selected_features += (
        direction_indicators
    )

    print(
        "Direction indicators added:"
    )

    for feature in direction_indicators:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 8. REMOVE UNSTABLE PERCENTAGE CHANGE FEATURES
    # ========================================================

    print_section(
        "REMOVING UNSTABLE PERCENTAGE-CHANGE FEATURES"
    )

    excluded_features = [
        "orders_change_percent",
        "revenue_change_percent",
        "units_change_percent",
        "review_count_change_percent",
    ]

    selected_features = [
        feature
        for feature in selected_features
        if feature not in excluded_features
    ]

    print(
        "Removed from ML predictors:"
    )

    for feature in excluded_features:

        print(
            f" - {feature}"
        )

    print(
        "\nReason:"
    )

    print(
        "These features can become extremely large when "
        "the previous-period denominator is very small."
    )

    print(
        "Absolute changes and explicit activity indicators "
        "are retained instead."
    )

    # ========================================================
    # 9. BUILD OUTPUT
    # ========================================================

    print_section(
        "BUILDING CURATED FEATURE DATASET"
    )

    output_columns = (
        [
            SELLER_COLUMN,
            DATE_COLUMN,
        ]
        + selected_features
        + [
            TARGET_COLUMN
        ]
    )

    # Validate duplicates before selecting.

    if len(output_columns) != len(
        set(output_columns)
    ):

        raise ValueError(
            "Duplicate output columns detected."
        )

    curated_df = df[
        output_columns
    ].copy()

    print(
        f"Rows: {len(curated_df):,}"
    )

    print(
        f"Predictor features: "
        f"{len(selected_features)}"
    )

    print(
        f"Total columns: "
        f"{len(curated_df.columns)}"
    )

    # ========================================================
    # 10. FEATURE TYPE VALIDATION
    # ========================================================

    print_section(
        "FEATURE TYPE VALIDATION"
    )

    non_numeric = (
        curated_df[
            selected_features
        ]
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric:

        raise ValueError(
            "Non-numeric predictor features found:\n"
            + "\n".join(non_numeric)
        )

    print(
        "All predictor features are numeric -> PASS"
    )

    # ========================================================
    # 11. FUTURE LEAKAGE CHECK
    # ========================================================

    print_section(
        "FUTURE INFORMATION CHECK"
    )

    future_prefix_columns = [
        column
        for column in selected_features
        if column.startswith(
            "future_"
        )
    ]

    if future_prefix_columns:

        raise ValueError(
            "Future outcome columns detected "
            "among predictors:\n"
            + "\n".join(
                future_prefix_columns
            )
        )

    if TARGET_COLUMN in selected_features:

        raise ValueError(
            "Target column accidentally included "
            "among predictors."
        )

    print(
        "No future outcome predictors -> PASS"
    )

    # ========================================================
    # 12. IDENTIFIER CHECK
    # ========================================================

    print_section(
        "IDENTIFIER CHECK"
    )

    forbidden_identifiers = [
        "seller_id",
    ]

    identifier_leakage = [
        column
        for column in selected_features
        if column in forbidden_identifiers
    ]

    if identifier_leakage:

        raise ValueError(
            "Identifier columns found among predictors:\n"
            + "\n".join(
                identifier_leakage
            )
        )

    print(
        "Seller identifier excluded from predictors -> PASS"
    )

    # ========================================================
    # 13. DUPLICATE SELLER / SNAPSHOT CHECK
    # ========================================================

    print_section(
        "SELLER / SNAPSHOT VALIDATION"
    )

    duplicate_keys = (
        curated_df[
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
            "Duplicate seller/snapshot records detected."
        )

    print(
        "Seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 14. TARGET VALIDATION
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    missing_target = (
        curated_df[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing targets: "
        f"{missing_target}"
    )

    if missing_target > 0:

        raise ValueError(
            "Missing target values detected."
        )

    invalid_targets = (
        set(
            curated_df[
                TARGET_COLUMN
            ]
            .unique()
        )
        - {0, 1}
    )

    print(
        f"Invalid targets: "
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
    # 15. MISSING VALUES
    # ========================================================

    print_section(
        "CURATED FEATURE MISSING VALUES"
    )

    missing = (
        curated_df[
            selected_features
        ]
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

    if missing.empty:

        print(
            "No missing predictor values."
        )

    else:

        print(
            missing.to_string()
        )

        print(
            "\nMissing values are intentionally retained."
        )

        print(
            "The downstream ML pipeline will handle them "
            "using training-data-only imputation."
        )

    # ========================================================
    # 16. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION"
    )

    print(
        curated_df[
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
            curated_df[
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
    # 17. FEATURE CORRELATION SCREEN
    # ========================================================

    print_section(
        "SIMPLE FEATURE CORRELATION SCREEN"
    )

    numeric_df = curated_df[
        selected_features
        + [TARGET_COLUMN]
    ].copy()

    correlation_to_target = (
        numeric_df
        .corr(
            numeric_only=True
        )[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    print(
        correlation_to_target
        .round(4)
        .to_string()
    )

    print(
        "\nImportant:"
    )

    print(
        "Correlation is used here only as a diagnostic."
    )

    print(
        "A low Pearson correlation does not mean a feature "
        "cannot be useful to a nonlinear Random Forest."
    )

    # ========================================================
    # 18. FEATURE COUNT BY GROUP
    # ========================================================

    print_section(
        "FEATURE GROUP SUMMARY"
    )

    print(
        f"Activity: "
        f"{len(ACTIVITY_FEATURES)}"
    )

    print(
        f"Commercial: "
        f"{len(COMMERCIAL_FEATURES)}"
    )

    print(
        f"Delivery: "
        f"{len(DELIVERY_FEATURES)}"
    )

    print(
        f"Reviews: "
        f"{len(REVIEW_FEATURES)}"
    )

    print(
        f"History: "
        f"{len(HISTORY_FEATURES)}"
    )

    print(
        f"Activity indicators: "
        f"{len(activity_indicators)}"
    )

    print(
        f"History indicators: "
        f"{len(history_indicators)}"
    )

    print(
        f"Direction indicators: "
        f"{len(direction_indicators)}"
    )

    print(
        f"Total curated predictors: "
        f"{len(selected_features)}"
    )

    # ========================================================
    # 19. SAVE
    # ========================================================

    print_section(
        "SAVING CURATED TEMPORAL FEATURES"
    )

    curated_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Curated temporal ML features saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 20. FINAL SUMMARY
    # ========================================================

    print_section(
        "CURATED TEMPORAL ML FEATURES COMPLETE"
    )

    print(
        f"Observations: "
        f"{len(curated_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{curated_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{curated_df[DATE_COLUMN].nunique():,}"
    )

    print(
        f"Curated predictors: "
        f"{len(selected_features)}"
    )

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print(
        "\nUnstable percentage-change predictors were excluded."
    )

    print(
        "Activity, history, delivery, and review signals were "
        "retained in a structured feature set."
    )

    print(
        "\nThe dataset is ready for the next temporal modeling stage."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()