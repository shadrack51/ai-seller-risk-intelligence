"""
Build Risk-Specific Temporal ML Datasets

Purpose
-------
Create two separate temporal machine-learning datasets:

    1. Future delivery-risk prediction
    2. Future review-risk prediction

The original future_at_risk target combines two different
business mechanisms. Separating them gives us more precise
prediction problems.

This script:

1. Loads the curated temporal feature dataset.
2. Loads the decomposed future outcomes.
3. Merges the two datasets by seller and snapshot.
4. Creates delivery-risk and review-risk datasets.
5. Applies target-specific feature whitelists.
6. Removes irrelevant future outcome fields.
7. Validates point-in-time structure.
8. Validates targets.
9. Saves both datasets.

This script DOES NOT train models.

Outputs
-------
data/processed/temporal_delivery_ml_dataset.csv
data/processed/temporal_review_ml_dataset.csv
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

FEATURE_DATASET = (
    DATA_DIR
    / "temporal_model_features.csv"
)

DECOMPOSITION_DATASET = (
    DATA_DIR
    / "temporal_target_decomposition.csv"
)

DELIVERY_OUTPUT = (
    DATA_DIR
    / "temporal_delivery_ml_dataset.csv"
)

REVIEW_OUTPUT = (
    DATA_DIR
    / "temporal_review_ml_dataset.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

DELIVERY_TARGET = "future_delivery_risk"

REVIEW_TARGET = "future_review_risk"


# ============================================================
# GENERAL FEATURE GROUPS
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
    "recent_orders_per_day",
    "older_orders_per_day",
    "recent_revenue_per_day",
    "older_revenue_per_day",
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
# DELIVERY-SPECIFIC FEATURES
# ============================================================

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
    "recent_delivery_activity",
    "older_delivery_activity",
    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",
]


# ============================================================
# REVIEW-SPECIFIC FEATURES
# ============================================================

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
    "recent_review_activity",
    "older_review_activity",
    "review_score_declining",
    "low_review_rate_increasing",
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
    """
    Check that required columns exist.
    """

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
    dataset_name,
):
    """
    Validate a binary target.
    """

    missing = (
        df[target]
        .isna()
        .sum()
    )

    if missing > 0:

        raise ValueError(
            f"{dataset_name} has "
            f"{missing} missing target values."
        )

    invalid = (
        set(
            df[target]
            .unique()
        )
        - {0, 1}
    )

    if invalid:

        raise ValueError(
            f"{dataset_name} contains invalid "
            f"target values: {invalid}"
        )


def validate_temporal_keys(df, dataset_name):
    """
    Ensure one seller has at most one observation per snapshot.
    """

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
        f"{dataset_name} duplicate seller/snapshot rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"{dataset_name} contains duplicate "
            "seller/snapshot observations."
        )


def print_target_summary(
    df,
    target,
    label,
):
    """
    Print class counts and percentages.
    """

    print(
        f"\n{label} target counts:"
    )

    print(
        df[target]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n{label} target percentages:"
    )

    print(
        (
            df[target]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING RISK-SPECIFIC TEMPORAL DATASETS")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING CURATED FEATURES AND TARGET DECOMPOSITION"
    )

    if not FEATURE_DATASET.exists():

        raise FileNotFoundError(
            f"Curated feature dataset not found:\n"
            f"{FEATURE_DATASET}\n\n"
            "Run build_temporal_model_features.py first."
        )

    if not DECOMPOSITION_DATASET.exists():

        raise FileNotFoundError(
            f"Target decomposition dataset not found:\n"
            f"{DECOMPOSITION_DATASET}\n\n"
            "Run temporal_target_decomposition.py first."
        )

    features_df = pd.read_csv(
        FEATURE_DATASET
    )

    decomposition_df = pd.read_csv(
        DECOMPOSITION_DATASET
    )

    print(
        f"Curated feature rows: "
        f"{len(features_df):,}"
    )

    print(
        f"Curated feature columns: "
        f"{len(features_df.columns):,}"
    )

    print(
        f"Decomposition rows: "
        f"{len(decomposition_df):,}"
    )

    print(
        f"Decomposition columns: "
        f"{len(decomposition_df.columns):,}"
    )

    # ========================================================
    # 2. PREPARE KEYS
    # ========================================================

    print_section(
        "PREPARING SELLER / SNAPSHOT KEYS"
    )

    features_df[DATE_COLUMN] = pd.to_datetime(
        features_df[DATE_COLUMN],
        errors="coerce",
    )

    decomposition_df[DATE_COLUMN] = pd.to_datetime(
        decomposition_df[DATE_COLUMN],
        errors="coerce",
    )

    if features_df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates in feature dataset."
        )

    if decomposition_df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates in decomposition dataset."
        )

    validate_temporal_keys(
        features_df,
        "Feature dataset",
    )

    validate_temporal_keys(
        decomposition_df,
        "Decomposition dataset",
    )

    print(
        "Temporal keys -> PASS"
    )

    # ========================================================
    # 3. VALIDATE REQUIRED COLUMNS
    # ========================================================

    print_section(
        "VALIDATING REQUIRED COLUMNS"
    )

    all_feature_columns = (
        GENERAL_FEATURES
        + DELIVERY_FEATURES
        + REVIEW_FEATURES
    )

    all_feature_columns = list(
        dict.fromkeys(
            all_feature_columns
        )
    )

    validate_columns(
        features_df,
        [
            SELLER_COLUMN,
            DATE_COLUMN,
            *all_feature_columns,
        ],
        "feature",
    )

    validate_columns(
        decomposition_df,
        [
            SELLER_COLUMN,
            DATE_COLUMN,
            DELIVERY_TARGET,
            REVIEW_TARGET,
        ],
        "decomposition",
    )

    print(
        "Required columns -> PASS"
    )

    # ========================================================
    # 4. MERGE TARGETS
    # ========================================================

    print_section(
        "MERGING RISK TARGETS"
    )

    target_df = decomposition_df[
        [
            SELLER_COLUMN,
            DATE_COLUMN,
            DELIVERY_TARGET,
            REVIEW_TARGET,
        ]
    ].copy()

    merged_df = features_df.merge(
        target_df,
        on=[
            SELLER_COLUMN,
            DATE_COLUMN,
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged observations: "
        f"{len(merged_df):,}"
    )

    if len(merged_df) != len(features_df):

        raise ValueError(
            "Some curated feature observations "
            "failed to match target observations."
        )

    print(
        "Feature/target alignment -> PASS"
    )

    # ========================================================
    # 5. VALIDATE TARGETS
    # ========================================================

    print_section(
        "VALIDATING RISK TARGETS"
    )

    validate_target(
        merged_df,
        DELIVERY_TARGET,
        "Delivery",
    )

    validate_target(
        merged_df,
        REVIEW_TARGET,
        "Review",
    )

    print(
        "Delivery target -> PASS"
    )

    print(
        "Review target -> PASS"
    )

    print_target_summary(
        merged_df,
        DELIVERY_TARGET,
        "Delivery-risk",
    )

    print_target_summary(
        merged_df,
        REVIEW_TARGET,
        "Review-risk",
    )

    # ========================================================
    # 6. BUILD DELIVERY DATASET
    # ========================================================

    print_section(
        "BUILDING DELIVERY-RISK DATASET"
    )

    delivery_features = (
        GENERAL_FEATURES
        + DELIVERY_FEATURES
    )

    delivery_features = list(
        dict.fromkeys(
            delivery_features
        )
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

    delivery_df = merged_df[
        delivery_columns
    ].copy()

    print(
        f"Delivery predictors: "
        f"{len(delivery_features)}"
    )

    print(
        "\nDelivery-specific features:"
    )

    for feature in DELIVERY_FEATURES:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 7. BUILD REVIEW DATASET
    # ========================================================

    print_section(
        "BUILDING REVIEW-RISK DATASET"
    )

    review_features = (
        GENERAL_FEATURES
        + REVIEW_FEATURES
    )

    review_features = list(
        dict.fromkeys(
            review_features
        )
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

    review_df = merged_df[
        review_columns
    ].copy()

    print(
        f"Review predictors: "
        f"{len(review_features)}"
    )

    print(
        "\nReview-specific features:"
    )

    for feature in REVIEW_FEATURES:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 8. TARGET-SPECIFIC FUTURE COLUMN CHECK
    # ========================================================

    print_section(
        "FUTURE INFORMATION CHECK"
    )

    forbidden_future_features = [
        "future_at_risk",
        "future_order_count",
        "future_delivered_orders",
        "future_late_delivery_rate",
        "future_average_review_score",
        "reconstructed_future_at_risk",
    ]

    for dataset_name, dataset, predictors in [
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

        leaked_columns = [
            column
            for column in predictors
            if column in forbidden_future_features
        ]

        if leaked_columns:

            raise ValueError(
                f"{dataset_name} dataset contains "
                "future information:\n"
                + "\n".join(
                    leaked_columns
                )
            )

        print(
            f"{dataset_name} future-feature check -> PASS"
        )

    # ========================================================
    # 9. IDENTIFIER CHECK
    # ========================================================

    print_section(
        "IDENTIFIER CHECK"
    )

    for dataset_name, predictors in [
        (
            "Delivery",
            delivery_features,
        ),
        (
            "Review",
            review_features,
        ),
    ]:

        if SELLER_COLUMN in predictors:

            raise ValueError(
                f"{dataset_name} dataset contains "
                "seller_id as a predictor."
            )

        print(
            f"{dataset_name} identifier check -> PASS"
        )

    # ========================================================
    # 10. NUMERIC FEATURE CHECK
    # ========================================================

    print_section(
        "NUMERIC FEATURE VALIDATION"
    )

    for dataset_name, dataset, predictors in [
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
            dataset[
                predictors
            ]
            .select_dtypes(
                exclude="number"
            )
            .columns
            .tolist()
        )

        if non_numeric:

            raise ValueError(
                f"{dataset_name} dataset contains "
                "non-numeric predictors:\n"
                + "\n".join(
                    non_numeric
                )
            )

        print(
            f"{dataset_name} predictors numeric -> PASS"
        )

    # ========================================================
    # 11. MISSING VALUES
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    for dataset_name, dataset, predictors in [
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
            dataset[
                predictors
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

        print(
            f"\n{dataset_name}:"
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
                "These will be handled by the "
                "training pipeline."
            )

    # ========================================================
    # 12. SELLER / SNAPSHOT UNIQUENESS
    # ========================================================

    print_section(
        "FINAL SELLER / SNAPSHOT VALIDATION"
    )

    validate_temporal_keys(
        delivery_df,
        "Delivery dataset",
    )

    validate_temporal_keys(
        review_df,
        "Review dataset",
    )

    print(
        "Delivery seller/snapshot uniqueness -> PASS"
    )

    print(
        "Review seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 13. DATASET SIZE
    # ========================================================

    print_section(
        "DATASET SIZE SUMMARY"
    )

    print(
        f"Delivery dataset rows: "
        f"{len(delivery_df):,}"
    )

    print(
        f"Delivery predictors: "
        f"{len(delivery_features)}"
    )

    print(
        f"Review dataset rows: "
        f"{len(review_df):,}"
    )

    print(
        f"Review predictors: "
        f"{len(review_features)}"
    )

    # ========================================================
    # 14. CORRELATION SCREEN
    # ========================================================

    print_section(
        "DELIVERY TARGET CORRELATION SCREEN"
    )

    delivery_corr = (
        delivery_df[
            delivery_features
            + [
                DELIVERY_TARGET
            ]
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
        delivery_corr
        .round(4)
        .to_string()
    )

    print_section(
        "REVIEW TARGET CORRELATION SCREEN"
    )

    review_corr = (
        review_df[
            review_features
            + [
                REVIEW_TARGET
            ]
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
        review_corr
        .round(4)
        .to_string()
    )

    print(
        "\nImportant:"
    )

    print(
        "These correlations are diagnostic only."
    )

    print(
        "They do not determine which features a Random Forest "
        "can or cannot use."
    )

    # ========================================================
    # 15. SAVE
    # ========================================================

    print_section(
        "SAVING RISK-SPECIFIC DATASETS"
    )

    delivery_df.to_csv(
        DELIVERY_OUTPUT,
        index=False,
    )

    review_df.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    print(
        f"Delivery-risk dataset saved to:\n"
        f"{DELIVERY_OUTPUT}"
    )

    print(
        f"\nReview-risk dataset saved to:\n"
        f"{REVIEW_OUTPUT}"
    )

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print_section(
        "RISK-SPECIFIC DATASETS COMPLETE"
    )

    print(
        "Delivery prediction:"
    )

    print(
        f"Rows: {len(delivery_df):,}"
    )

    print(
        f"Predictors: {len(delivery_features)}"
    )

    print(
        f"Target: {DELIVERY_TARGET}"
    )

    print(
        "\nReview prediction:"
    )

    print(
        f"Rows: {len(review_df):,}"
    )

    print(
        f"Predictors: {len(review_features)}"
    )

    print(
        f"Target: {REVIEW_TARGET}"
    )

    print(
        "\nThe combined future-risk problem has been "
        "separated into two target-specific prediction tasks."
    )

    print(
        "\nNo models were trained."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()