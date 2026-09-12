"""
Temporal Feature / Target Audit

Purpose
-------
Investigate why the temporal Random Forest performs extremely
well on training data but poorly on later validation/test periods.

The audit examines:

1. Target stability over time
2. Feature distribution stability across train/validation/test
3. Feature-target correlations across periods
4. Seller historical sample sizes
5. Small-history seller observations
6. Missing-value patterns across time

This script DOES NOT train a model.

Outputs
-------
data/processed/temporal_feature_stability.csv
data/processed/temporal_feature_target_correlations.csv
data/processed/temporal_history_size_analysis.csv
data/processed/temporal_target_by_snapshot.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

TEMPORAL_DATASET = (
    DATA_DIR / "temporal_ml_dataset.csv"
)

TRAIN_FILE = (
    DATA_DIR / "seller_temporal_train.csv"
)

VALIDATION_FILE = (
    DATA_DIR / "seller_temporal_validation.csv"
)

TEST_FILE = (
    DATA_DIR / "seller_temporal_test.csv"
)

FEATURE_STABILITY_FILE = (
    DATA_DIR / "temporal_feature_stability.csv"
)

CORRELATION_FILE = (
    DATA_DIR
    / "temporal_feature_target_correlations.csv"
)

HISTORY_SIZE_FILE = (
    DATA_DIR
    / "temporal_history_size_analysis.csv"
)

TARGET_SNAPSHOT_FILE = (
    DATA_DIR
    / "temporal_target_by_snapshot.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "future_at_risk"

DATE_COLUMN = "snapshot_date"

FEATURES = [
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

# Small-history thresholds.
HISTORY_THRESHOLDS = [
    1,
    2,
    3,
    5,
    10,
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


def summarize_feature_distribution(
    df,
    feature,
):
    """
    Produce descriptive statistics for one feature.
    """

    series = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

    return {
        "count": series.count(),
        "missing": series.isna().sum(),
        "missing_rate": series.isna().mean(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "p25": series.quantile(0.25),
        "p75": series.quantile(0.75),
        "max": series.max(),
    }


def calculate_correlation(
    df,
    feature,
):
    """
    Calculate Pearson correlation between one feature
    and the binary future-risk target.

    Pandas uses pairwise complete observations when
    calculating the correlation.
    """

    subset = df[
        [
            feature,
            TARGET_COLUMN,
        ]
    ].dropna()

    if len(subset) < 2:

        return np.nan

    return subset[
        feature
    ].corr(
        subset[TARGET_COLUMN]
    )


def history_bucket(orders):
    """
    Classify historical order volume into interpretable
    sample-size buckets.
    """

    if orders <= 2:
        return "1-2"

    if orders <= 5:
        return "3-5"

    if orders <= 10:
        return "6-10"

    if orders <= 20:
        return "11-20"

    return "21+"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL FEATURE / TARGET AUDIT")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATASETS
    # ========================================================

    print_section(
        "LOADING TEMPORAL DATA"
    )

    if not TEMPORAL_DATASET.exists():

        raise FileNotFoundError(
            f"Temporal ML dataset not found:\n"
            f"{TEMPORAL_DATASET}"
        )

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation dataset not found:\n"
            f"{VALIDATION_FILE}"
        )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_FILE}"
        )

    full_df = pd.read_csv(
        TEMPORAL_DATASET
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Full temporal observations: "
        f"{len(full_df):,}"
    )

    print(
        f"Training observations: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation observations: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test observations: "
        f"{len(test_df):,}"
    )

    # ========================================================
    # 2. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING TEMPORAL DATA"
    )

    for df in [
        full_df,
        train_df,
        validation_df,
        test_df,
    ]:

        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce",
        )

    print(
        "Snapshot date conversion -> PASS"
    )

    # ========================================================
    # 3. VALIDATE FEATURES
    # ========================================================

    print_section(
        "FEATURE VALIDATION"
    )

    for feature in FEATURES:

        if feature not in full_df.columns:

            raise ValueError(
                f"Feature missing: {feature}"
            )

    if TARGET_COLUMN not in full_df.columns:

        raise ValueError(
            f"Target missing: {TARGET_COLUMN}"
        )

    print(
        f"Historical features validated: "
        f"{len(FEATURES)}"
    )

    print(
        f"Target: {TARGET_COLUMN}"
    )

    # ========================================================
    # 4. TARGET STABILITY BY SNAPSHOT
    # ========================================================

    print_section(
        "FUTURE TARGET STABILITY BY SNAPSHOT"
    )

    target_snapshot = (
        full_df
        .groupby(DATE_COLUMN)
        [TARGET_COLUMN]
        .agg(
            observations="count",
            at_risk_count="sum",
            at_risk_rate="mean",
        )
        .reset_index()
    )

    target_snapshot[
        "at_risk_rate"
    ] = (
        target_snapshot[
            "at_risk_rate"
        ]
        * 100
    )

    target_snapshot[
        "at_risk_rate"
    ] = target_snapshot[
        "at_risk_rate"
    ].round(2)

    print(
        target_snapshot.to_string(
            index=False
        )
    )

    target_snapshot.to_csv(
        TARGET_SNAPSHOT_FILE,
        index=False,
    )

    # ========================================================
    # 5. TARGET STABILITY BY SPLIT
    # ========================================================

    print_section(
        "FUTURE TARGET STABILITY BY SPLIT"
    )

    split_target_rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:

        rate = (
            split_df[TARGET_COLUMN]
            .mean()
            * 100
        )

        split_target_rows.append(
            {
                "split": split_name,
                "observations": len(split_df),
                "at_risk_count": (
                    split_df[
                        TARGET_COLUMN
                    ].sum()
                ),
                "at_risk_rate": rate,
            }
        )

    split_target_df = pd.DataFrame(
        split_target_rows
    )

    split_target_df[
        "at_risk_rate"
    ] = split_target_df[
        "at_risk_rate"
    ].round(2)

    print(
        split_target_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 6. FEATURE DISTRIBUTION STABILITY
    # ========================================================

    print_section(
        "FEATURE DISTRIBUTION STABILITY"
    )

    distribution_rows = []

    split_datasets = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for feature in FEATURES:

        for split_name, split_df in (
            split_datasets.items()
        ):

            stats = summarize_feature_distribution(
                split_df,
                feature,
            )

            stats["feature"] = feature
            stats["split"] = split_name

            distribution_rows.append(
                stats
            )

    feature_stability_df = pd.DataFrame(
        distribution_rows
    )

    feature_stability_df = (
        feature_stability_df[
            [
                "feature",
                "split",
                "count",
                "missing",
                "missing_rate",
                "mean",
                "median",
                "std",
                "min",
                "p25",
                "p75",
                "max",
            ]
        ]
    )

    print(
        feature_stability_df
        .round(3)
        .to_string(index=False)
    )

    feature_stability_df.to_csv(
        FEATURE_STABILITY_FILE,
        index=False,
    )

    # ========================================================
    # 7. FEATURE MEAN SHIFT
    # ========================================================

    print_section(
        "FEATURE MEAN SHIFT"
    )

    mean_shift_rows = []

    for feature in FEATURES:

        train_mean = (
            train_df[feature]
            .mean()
        )

        validation_mean = (
            validation_df[feature]
            .mean()
        )

        test_mean = (
            test_df[feature]
            .mean()
        )

        validation_shift = (
            validation_mean
            - train_mean
        )

        test_shift = (
            test_mean
            - train_mean
        )

        if train_mean != 0:

            validation_relative_shift = (
                validation_shift
                / abs(train_mean)
                * 100
            )

            test_relative_shift = (
                test_shift
                / abs(train_mean)
                * 100
            )

        else:

            validation_relative_shift = np.nan

            test_relative_shift = np.nan

        mean_shift_rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "validation_mean": validation_mean,
                "test_mean": test_mean,
                "validation_absolute_shift": (
                    validation_shift
                ),
                "test_absolute_shift": (
                    test_shift
                ),
                "validation_relative_shift_percent": (
                    validation_relative_shift
                ),
                "test_relative_shift_percent": (
                    test_relative_shift
                ),
            }
        )

    mean_shift_df = pd.DataFrame(
        mean_shift_rows
    )

    print(
        mean_shift_df
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 8. FEATURE-TARGET CORRELATIONS
    # ========================================================

    print_section(
        "FEATURE / TARGET CORRELATIONS"
    )

    correlation_rows = []

    for feature in FEATURES:

        all_corr = calculate_correlation(
            full_df,
            feature,
        )

        train_corr = calculate_correlation(
            train_df,
            feature,
        )

        validation_corr = calculate_correlation(
            validation_df,
            feature,
        )

        test_corr = calculate_correlation(
            test_df,
            feature,
        )

        correlation_rows.append(
            {
                "feature": feature,
                "all_data_correlation": all_corr,
                "train_correlation": train_corr,
                "validation_correlation": (
                    validation_corr
                ),
                "test_correlation": test_corr,
            }
        )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    correlation_df[
        "train_to_test_correlation_change"
    ] = (
        correlation_df[
            "test_correlation"
        ]
        -
        correlation_df[
            "train_correlation"
        ]
    )

    correlation_df[
        "train_to_validation_correlation_change"
    ] = (
        correlation_df[
            "validation_correlation"
        ]
        -
        correlation_df[
            "train_correlation"
        ]
    )

    correlation_df = (
        correlation_df
        .sort_values(
            "test_correlation",
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    print(
        correlation_df
        .round(4)
        .to_string(index=False)
    )

    correlation_df.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    # ========================================================
    # 9. HISTORICAL SAMPLE SIZE ANALYSIS
    # ========================================================

    print_section(
        "HISTORICAL SAMPLE SIZE ANALYSIS"
    )

    # total_orders represents the historical number of orders
    # available for each seller snapshot.

    history_df = full_df[
        [
            "seller_id",
            DATE_COLUMN,
            "total_orders",
            "delivered_orders",
            "review_count",
        ]
    ].copy()

    history_df[
        "history_bucket"
    ] = (
        history_df[
            "total_orders"
        ]
        .fillna(0)
        .apply(
            history_bucket
        )
    )

    bucket_summary = (
        history_df
        .groupby("history_bucket")
        .agg(
            observations=(
                "seller_id",
                "count",
            ),
            unique_sellers=(
                "seller_id",
                "nunique",
            ),
            average_orders=(
                "total_orders",
                "mean",
            ),
            average_reviews=(
                "review_count",
                "mean",
            ),
            future_risk_rate=(
                TARGET_COLUMN
                if TARGET_COLUMN in history_df
                else "total_orders",
                "mean",
            ),
        )
        .reset_index()
    )

    # The conditional aggregation above cannot obtain the target
    # because history_df deliberately contains only selected
    # history columns, so calculate future risk by bucket
    # separately.

    target_by_bucket = (
        full_df[
            [
                "seller_id",
                DATE_COLUMN,
                "total_orders",
                TARGET_COLUMN,
            ]
        ]
        .copy()
    )

    target_by_bucket[
        "history_bucket"
    ] = (
        target_by_bucket[
            "total_orders"
        ]
        .fillna(0)
        .apply(
            history_bucket
        )
    )

    target_bucket_rates = (
        target_by_bucket
        .groupby("history_bucket")
        [TARGET_COLUMN]
        .agg(
            observations="count",
            at_risk_count="sum",
            future_risk_rate="mean",
        )
        .reset_index()
    )

    target_bucket_rates[
        "future_risk_rate"
    ] = (
        target_bucket_rates[
            "future_risk_rate"
        ]
        * 100
    )

    history_summary = (
        history_df
        .groupby("history_bucket")
        .agg(
            observations=(
                "seller_id",
                "count",
            ),
            unique_sellers=(
                "seller_id",
                "nunique",
            ),
            average_orders=(
                "total_orders",
                "mean",
            ),
            average_reviews=(
                "review_count",
                "mean",
            ),
        )
        .reset_index()
        .merge(
            target_bucket_rates[
                [
                    "history_bucket",
                    "at_risk_count",
                    "future_risk_rate",
                ]
            ],
            on="history_bucket",
            how="left",
        )
    )

    print(
        history_summary
        .round(3)
        .to_string(index=False)
    )

    history_summary.to_csv(
        HISTORY_SIZE_FILE,
        index=False,
    )

    # ========================================================
    # 10. EXTREMELY SMALL HISTORY
    # ========================================================

    print_section(
        "SMALL-HISTORY OBSERVATIONS"
    )

    for threshold in HISTORY_THRESHOLDS:

        count = (
            full_df[
                "total_orders"
            ]
            .fillna(0)
            .le(threshold)
            .sum()
        )

        percentage = (
            count
            / len(full_df)
            * 100
        )

        print(
            f"Observations with "
            f"<= {threshold} historical orders: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # 11. MISSINGNESS BY SPLIT
    # ========================================================

    print_section(
        "MISSINGNESS BY SPLIT"
    )

    missingness_rows = []

    for feature in FEATURES:

        for split_name, split_df in (
            split_datasets.items()
        ):

            missing_count = (
                split_df[
                    feature
                ]
                .isna()
                .sum()
            )

            missing_rate = (
                missing_count
                / len(split_df)
                * 100
            )

            missingness_rows.append(
                {
                    "feature": feature,
                    "split": split_name,
                    "missing_count": missing_count,
                    "missing_rate": missing_rate,
                }
            )

    missingness_df = pd.DataFrame(
        missingness_rows
    )

    print(
        missingness_df
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 12. AUDIT CONCLUSIONS
    # ========================================================

    print_section(
        "AUDIT INTERPRETATION"
    )

    print(
        "The following questions should be considered "
        "before retraining the temporal model:"
    )

    print(
        "\n1. Do feature distributions shift substantially "
        "between training and testing?"
    )

    print(
        "2. Do feature-target relationships remain stable "
        "over time?"
    )

    print(
        "3. Are many seller snapshots based on very small "
        "historical samples?"
    )

    print(
        "4. Does missingness change materially across time?"
    )

    print(
        "\nThe audit does not automatically remove features."
    )

    print(
        "Its purpose is to identify where the temporal "
        "prediction problem may be unstable."
    )

    # ========================================================
    # 13. SAVE SUMMARY OUTPUTS
    # ========================================================

    print_section(
        "AUDIT OUTPUTS"
    )

    print(
        f"Feature stability saved to:\n"
        f"{FEATURE_STABILITY_FILE}"
    )

    print(
        f"\nFeature-target correlations saved to:\n"
        f"{CORRELATION_FILE}"
    )

    print(
        f"\nHistory-size analysis saved to:\n"
        f"{HISTORY_SIZE_FILE}"
    )

    print(
        f"\nTarget-by-snapshot analysis saved to:\n"
        f"{TARGET_SNAPSHOT_FILE}"
    )

    # ========================================================
    # 14. COMPLETE
    # ========================================================

    print_section(
        "TEMPORAL FEATURE / TARGET AUDIT COMPLETE"
    )

    print(
        f"Features audited: "
        f"{len(FEATURES)}"
    )

    print(
        f"Temporal observations: "
        f"{len(full_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{full_df['seller_id'].nunique():,}"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The next modeling decision should be based on "
        "the audit findings."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()