"""
Audit Delivery Context Features

Purpose
-------
Audit the contextual delivery features created by:

    build_delivery_context_features.py

The audit checks:

1. Dataset structure
2. Seller/snapshot grain
3. Target validity
4. Numeric feature types
5. Missingness
6. Bounded proportion levels
7. Bounded proportion changes
8. Ratio levels
9. Ratio changes
10. Count levels
11. Count changes
12. Feature/target correlations
13. Feature redundancy
14. Temporal coverage

Important mathematical distinction
----------------------------------
A level and the change in that level do not necessarily have
the same valid range.

Examples:

    cross_state_rate:
        0 to 1

    cross_state_rate_change:
        -1 to +1

    freight_to_revenue:
        >= 0

    freight_to_revenue_change:
        can be negative or positive

This script does NOT train a model.

Input
-----
data/processed/delivery_context_features.csv

Output
------
data/processed/delivery_context_feature_audit.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    DATA_DIR
    / "delivery_context_features.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "delivery_context_feature_audit.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_delivery_risk"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_required_columns(
    df,
    required_columns,
):
    """Ensure required columns exist."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing
            )
        )


def validate_target(df):
    """Validate the binary future delivery-risk target."""

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Missing target values detected."
        )

    invalid = (
        set(
            df[TARGET_COLUMN]
            .dropna()
            .unique()
        )
        - {0, 1}
    )

    if invalid:

        raise ValueError(
            f"Invalid target values: {invalid}"
        )


def validate_grain(df):
    """Validate seller/snapshot uniqueness."""

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
        f"Duplicate seller/snapshot rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate seller/snapshot rows detected."
        )


def get_features(df):
    """
    Return predictor columns.

    Identifiers, dates, targets, and future-prefixed
    information are excluded.
    """

    excluded = {
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    }

    return [
        column
        for column in df.columns
        if (
            column not in excluded
            and not column.startswith(
                "future_"
            )
        )
    ]


def validate_numeric_features(
    df,
    features,
):
    """Ensure all predictors are numeric."""

    non_numeric = (
        df[
            features
        ]
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric:

        raise ValueError(
            "Non-numeric predictors found:\n"
            + "\n".join(
                non_numeric
            )
        )


def find_correlated_pairs(
    df,
    features,
    threshold=0.90,
):
    """
    Find highly correlated predictor pairs.

    Only one copy of each pair is reported.
    """

    matrix = (
        df[
            features
        ]
        .corr(
            numeric_only=True
        )
        .abs()
    )

    upper = (
        matrix
        .where(
            np.triu(
                np.ones(
                    matrix.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )
    )

    rows = []

    for column in upper.columns:

        for index in upper.index:

            value = (
                upper.loc[
                    index,
                    column,
                ]
            )

            if (
                pd.notna(value)
                and value >= threshold
            ):

                rows.append(
                    {
                        "feature_1": index,
                        "feature_2": column,
                        "absolute_correlation": value,
                    }
                )

    if not rows:

        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AUDITING DELIVERY CONTEXT FEATURES"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD
    # ========================================================

    print_section(
        "LOADING CONTEXT FEATURE DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_delivery_context_features.py first."
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
    # 2. REQUIRED COLUMNS
    # ========================================================

    print_section(
        "COLUMN VALIDATION"
    )

    validate_required_columns(
        df,
        [
            SELLER_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        ],
    )

    print(
        "Required columns -> PASS"
    )

    # ========================================================
    # 3. DATE VALIDATION
    # ========================================================

    print_section(
        "DATE VALIDATION"
    )

    df[
        DATE_COLUMN
    ] = pd.to_datetime(
        df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    invalid_dates = (
        df[
            DATE_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"Invalid snapshot dates: "
        f"{invalid_dates}"
    )

    if invalid_dates > 0:

        raise ValueError(
            "Invalid snapshot dates found."
        )

    print(
        "Snapshot dates -> PASS"
    )

    # ========================================================
    # 4. TARGET VALIDATION
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    validate_target(
        df
    )

    print(
        "Target values -> PASS"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget percentages:"
    )

    print(
        (
            df[
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
    # 5. GRAIN
    # ========================================================

    print_section(
        "PREDICTION GRAIN VALIDATION"
    )

    validate_grain(
        df
    )

    print(
        "Seller/snapshot uniqueness -> PASS"
    )

    print(
        f"Unique sellers: "
        f"{df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Unique snapshots: "
        f"{df[DATE_COLUMN].nunique():,}"
    )

    # ========================================================
    # 6. FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING CONTEXT PREDICTORS"
    )

    features = get_features(
        df
    )

    print(
        f"Context predictors: "
        f"{len(features)}"
    )

    for number, feature in enumerate(
        features,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 7. NUMERIC VALIDATION
    # ========================================================

    print_section(
        "NUMERIC FEATURE VALIDATION"
    )

    validate_numeric_features(
        df,
        features,
    )

    print(
        "All context predictors numeric -> PASS"
    )

    # ========================================================
    # 8. FUTURE INFORMATION CHECK
    # ========================================================

    print_section(
        "FUTURE INFORMATION CHECK"
    )

    future_features = [
        feature
        for feature in features
        if feature.startswith(
            "future_"
        )
    ]

    if future_features:

        raise ValueError(
            "Future predictors detected:\n"
            +
            "\n".join(
                future_features
            )
        )

    print(
        "No future predictors -> PASS"
    )

    # ========================================================
    # 9. MISSING VALUES
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing = (
        df[
            features
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
            "No missing context predictors."
        )

    else:

        print(
            missing.to_string()
        )

        print(
            "\nMissing values are expected where "
            "historical order evidence is insufficient."
        )

    # ========================================================
    # 10. BOUNDED PROPORTION LEVELS
    # ========================================================

    print_section(
        "BOUNDED PROPORTION FEATURE VALIDATION"
    )

    proportion_level_features = [
        "recent_context_cross_state_rate",
        "older_context_cross_state_rate",
        "recent_context_destination_concentration",
        "older_context_destination_concentration",
    ]

    proportion_level_features = [
        feature
        for feature
        in proportion_level_features
        if feature in df.columns
    ]

    print(
        f"Bounded level features: "
        f"{len(proportion_level_features)}"
    )

    invalid_levels = []

    for feature in proportion_level_features:

        values = (
            df[
                feature
            ]
            .dropna()
        )

        minimum = (
            values.min()
            if not values.empty
            else np.nan
        )

        maximum = (
            values.max()
            if not values.empty
            else np.nan
        )

        invalid = (
            (
                values < 0
            )
            |
            (
                values > 1
            )
        ).sum()

        print(
            f"{feature}: "
            f"min={minimum:.4f}, "
            f"max={maximum:.4f}, "
            f"invalid={invalid}"
        )

        if invalid > 0:

            invalid_levels.append(
                (
                    feature,
                    invalid,
                )
            )

    if invalid_levels:

        raise ValueError(
            "Invalid bounded proportion levels found."
        )

    print(
        "Bounded proportion levels -> PASS"
    )

    # ========================================================
    # 11. BOUNDED PROPORTION CHANGES
    # ========================================================

    print_section(
        "PROPORTION CHANGE FEATURE VALIDATION"
    )

    proportion_change_features = [
        "context_cross_state_rate_change",
        "context_destination_concentration_change",
    ]

    proportion_change_features = [
        feature
        for feature
        in proportion_change_features
        if feature in df.columns
    ]

    print(
        f"Bounded proportion changes: "
        f"{len(proportion_change_features)}"
    )

    invalid_changes = []

    for feature in proportion_change_features:

        values = (
            df[
                feature
            ]
            .dropna()
        )

        minimum = (
            values.min()
            if not values.empty
            else np.nan
        )

        maximum = (
            values.max()
            if not values.empty
            else np.nan
        )

        invalid = (
            (
                values < -1
            )
            |
            (
                values > 1
            )
        ).sum()

        print(
            f"{feature}: "
            f"min={minimum:.4f}, "
            f"max={maximum:.4f}, "
            f"invalid={invalid}"
        )

        if invalid > 0:

            invalid_changes.append(
                (
                    feature,
                    invalid,
                )
            )

    if invalid_changes:

        raise ValueError(
            "Invalid bounded proportion changes found."
        )

    print(
        "Proportion change validation -> PASS"
    )

    print(
        "\nNegative proportion changes are valid."
    )

    # ========================================================
    # 12. RATIO LEVELS
    # ========================================================

    print_section(
        "RATIO LEVEL VALIDATION"
    )

    ratio_level_features = [
        "recent_context_freight_to_revenue",
        "older_context_freight_to_revenue",
    ]

    ratio_level_features = [
        feature
        for feature
        in ratio_level_features
        if feature in df.columns
    ]

    print(
        f"Ratio level features: "
        f"{len(ratio_level_features)}"
    )

    invalid_ratio_levels = []

    for feature in ratio_level_features:

        values = (
            df[
                feature
            ]
            .dropna()
        )

        minimum = (
            values.min()
            if not values.empty
            else np.nan
        )

        maximum = (
            values.max()
            if not values.empty
            else np.nan
        )

        negative_count = (
            values
            .lt(0)
            .sum()
        )

        print(
            f"{feature}: "
            f"min={minimum:.4f}, "
            f"max={maximum:.4f}, "
            f"negative={negative_count}"
        )

        if negative_count > 0:

            invalid_ratio_levels.append(
                (
                    feature,
                    negative_count,
                )
            )

    if invalid_ratio_levels:

        raise ValueError(
            "Negative ratio levels found."
        )

    print(
        "Ratio levels -> PASS"
    )

    # ========================================================
    # 13. RATIO CHANGE FEATURES
    # ========================================================

    print_section(
        "RATIO CHANGE FEATURE VALIDATION"
    )

    ratio_change_features = [
        "context_freight_to_revenue_change",
    ]

    ratio_change_features = [
        feature
        for feature
        in ratio_change_features
        if feature in df.columns
    ]

    print(
        f"Ratio change features: "
        f"{len(ratio_change_features)}"
    )

    for feature in ratio_change_features:

        values = (
            df[
                feature
            ]
            .dropna()
        )

        if values.empty:

            print(
                f"{feature}: no non-missing values"
            )

            continue

        print(
            f"{feature}: "
            f"min={values.min():.4f}, "
            f"median={values.median():.4f}, "
            f"max={values.max():.4f}"
        )

    print(
        "Ratio changes -> PASS"
    )

    print(
        "\nNegative ratio changes are valid."
    )

    # ========================================================
    # 14. COUNT LEVEL FEATURES
    # ========================================================

    print_section(
        "COUNT FEATURE VALIDATION"
    )

    count_level_features = [
        feature
        for feature in features
        if (
            "count" in feature
            or feature.endswith(
                "_units"
            )
            or feature.endswith(
                "_states"
            )
        )
        and not feature.endswith(
            "_change"
        )
    ]

    invalid_counts = []

    for feature in count_level_features:

        values = (
            df[
                feature
            ]
            .dropna()
        )

        negative_count = (
            values
            .lt(0)
            .sum()
        )

        print(
            f"{feature}: "
            f"{negative_count} negative values"
        )

        if negative_count > 0:

            invalid_counts.append(
                (
                    feature,
                    negative_count,
                )
            )

    if invalid_counts:

        raise ValueError(
            "Negative count levels found."
        )

    print(
        "Count levels -> PASS"
    )

    # ========================================================
    # 15. CHANGE FEATURE INSPECTION
    # ========================================================

    print_section(
        "CHANGE FEATURE INSPECTION"
    )

    change_features = [
        feature
        for feature in features
        if feature.endswith(
            "_change"
        )
    ]

    print(
        f"Change features identified: "
        f"{len(change_features)}"
    )

    for feature in change_features:

        values = (
            df[
                feature
            ]
            .dropna()
        )

        if values.empty:

            continue

        print(
            f"{feature}: "
            f"min={values.min():.4f}, "
            f"median={values.median():.4f}, "
            f"max={values.max():.4f}"
        )

    print(
        "Change-feature inspection -> PASS"
    )

    # ========================================================
    # 16. FEATURE/TARGET CORRELATION
    # ========================================================

    print_section(
        "FEATURE / TARGET CORRELATION SCREEN"
    )

    correlation = (
        df[
            features
            + [
                TARGET_COLUMN
            ]
        ]
        .corr(
            numeric_only=True
        )[
            TARGET_COLUMN
        ]
        .drop(
            TARGET_COLUMN
        )
        .sort_values(
            key=lambda values:
                values.abs(),
            ascending=False,
        )
    )

    correlation_table = (
        correlation
        .to_frame(
            "target_correlation"
        )
    )

    correlation_table[
        "absolute_correlation"
    ] = (
        correlation_table[
            "target_correlation"
        ]
        .abs()
    )

    print(
        correlation_table
        .round(4)
        .to_string()
    )

    print(
        "\nCorrelation is diagnostic only."
    )

    print(
        "It does not establish causality or determine "
        "which features a Random Forest will use."
    )

    # ========================================================
    # 17. FEATURE REDUNDANCY
    # ========================================================

    print_section(
        "FEATURE REDUNDANCY CHECK"
    )

    correlated_pairs = (
        find_correlated_pairs(
            df,
            features,
            threshold=0.90,
        )
    )

    if correlated_pairs.empty:

        print(
            "No feature pairs with "
            "|correlation| >= 0.90."
        )

    else:

        print(
            f"Highly correlated feature pairs: "
            f"{len(correlated_pairs)}"
        )

        print(
            correlated_pairs
            .round(4)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 18. TEMPORAL COVERAGE
    # ========================================================

    print_section(
        "TEMPORAL COVERAGE"
    )

    temporal_summary = (
        df
        .groupby(
            DATE_COLUMN
        )
        .agg(
            observations=(
                SELLER_COLUMN,
                "count",
            ),
            sellers=(
                SELLER_COLUMN,
                "nunique",
            ),
            positive_count=(
                TARGET_COLUMN,
                "sum",
            ),
            positive_rate=(
                TARGET_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    temporal_summary[
        "positive_rate"
    ] = (
        temporal_summary[
            "positive_rate"
        ]
        * 100
    )

    print(
        temporal_summary
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 19. FEATURE STATISTICS
    # ========================================================

    print_section(
        "CONTEXT FEATURE STATISTICS"
    )

    statistics = (
        df[
            features
        ]
        .describe()
        .T
    )

    print(
        statistics
        .round(3)
        .to_string()
    )

    # ========================================================
    # 20. BUILD AUDIT TABLE
    # ========================================================

    print_section(
        "BUILDING FEATURE AUDIT TABLE"
    )

    audit_rows = []

    for feature in features:

        series = df[
            feature
        ]

        audit_rows.append(
            {
                "feature": feature,
                "dtype": str(
                    series.dtype
                ),
                "missing_count": (
                    series.isna().sum()
                ),
                "missing_percentage": (
                    series.isna().mean()
                    * 100
                ),
                "non_missing_count": (
                    series.notna().sum()
                ),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "minimum": series.min(),
                "maximum": series.max(),
                "target_correlation": (
                    correlation.get(
                        feature,
                        np.nan,
                    )
                ),
            }
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df[
        "absolute_target_correlation"
    ] = (
        audit_df[
            "target_correlation"
        ]
        .abs()
    )

    # ========================================================
    # 21. SAVE
    # ========================================================

    print_section(
        "SAVING CONTEXT FEATURE AUDIT"
    )

    audit_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Feature audit saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 22. FINAL SUMMARY
    # ========================================================

    print_section(
        "DELIVERY CONTEXT FEATURE AUDIT COMPLETE"
    )

    print(
        f"Observations audited: "
        f"{len(df):,}"
    )

    print(
        f"Unique sellers: "
        f"{df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{df[DATE_COLUMN].nunique():,}"
    )

    print(
        f"Context predictors: "
        f"{len(features):,}"
    )

    print(
        f"Highly correlated pairs: "
        f"{len(correlated_pairs):,}"
    )

    print(
        "\nBounded proportion levels -> PASS"
    )

    print(
        "Bounded proportion changes -> PASS"
    )

    print(
        "Ratio levels -> PASS"
    )

    print(
        "Ratio changes -> PASS"
    )

    print(
        "Count levels -> PASS"
    )

    print(
        "Future information check -> PASS"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The context feature layer is ready for "
        "the controlled temporal experiment."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()