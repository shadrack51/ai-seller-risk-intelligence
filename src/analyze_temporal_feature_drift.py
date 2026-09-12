"""
Temporal Feature Drift Analysis

Purpose
-------
Measure how predictor distributions and target prevalence
change between historical training data and later periods.

This is a diagnostic / model-monitoring analysis.

It does NOT:

    - retrain models
    - select thresholds
    - modify predictions
    - calibrate models
    - use drift results to tune the final test

Concepts
--------
Reference population
    Historical training data used as the baseline.

Comparison population
    Later validation or test data compared with the baseline.

Feature drift
    A predictor distribution changes over time.

Target drift / prevalence drift
    The positive-target rate changes over time.

PSI
    Population Stability Index, used here as a quantitative
    measure of distribution movement.

Inputs
------
Delivery:
    target_delivery_train.csv
    target_delivery_validation.csv
    target_delivery_test.csv

Review:
    target_review_train.csv
    target_review_validation.csv
    target_review_test.csv

Outputs
-------
temporal_feature_drift_summary.csv
temporal_feature_drift_detail.csv
temporal_target_drift_summary.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# INPUT DATASETS
# ============================================================

DELIVERY_TRAIN_FILE = (
    DATA_DIR
    / "target_delivery_train.csv"
)

DELIVERY_VALIDATION_FILE = (
    DATA_DIR
    / "target_delivery_validation.csv"
)

DELIVERY_TEST_FILE = (
    DATA_DIR
    / "target_delivery_test.csv"
)


REVIEW_TRAIN_FILE = (
    DATA_DIR
    / "target_review_train.csv"
)

REVIEW_VALIDATION_FILE = (
    DATA_DIR
    / "target_review_validation.csv"
)

REVIEW_TEST_FILE = (
    DATA_DIR
    / "target_review_test.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SUMMARY_OUTPUT_FILE = (
    DATA_DIR
    / "temporal_feature_drift_summary.csv"
)

DETAIL_OUTPUT_FILE = (
    DATA_DIR
    / "temporal_feature_drift_detail.csv"
)

TARGET_OUTPUT_FILE = (
    DATA_DIR
    / "temporal_target_drift_summary.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"


DELIVERY_TARGET = (
    "future_delivery_risk"
)

REVIEW_TARGET = (
    "future_review_risk"
)


# ============================================================
# PSI SETTINGS
# ============================================================

NUMBER_OF_BINS = 10

EPSILON = 1e-6


# ============================================================
# PSI INTERPRETATION
# ============================================================

PSI_LOW_THRESHOLD = 0.10

PSI_MODERATE_THRESHOLD = 0.25


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
    """Validate that a required file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def prepare_dataset(
    path,
    label,
    target,
):
    """
    Load, prepare, and validate one temporal dataset.
    """

    validate_file(
        path,
        label,
    )

    df = pd.read_csv(
        path
    )

    print(
        f"{label} rows: "
        f"{len(df):,}"
    )

    required_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        target,
    ]

    missing = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{label} missing columns:\n"
            +
            "\n".join(
                missing
            )
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

    if invalid_dates > 0:

        raise ValueError(
            f"{label} contains "
            f"{invalid_dates} invalid dates."
        )

    missing_target = (
        df[
            target
        ]
        .isna()
        .sum()
    )

    if missing_target > 0:

        raise ValueError(
            f"{label} contains "
            f"{missing_target} missing target values."
        )

    invalid_target = (
        set(
            df[
                target
            ]
            .unique()
        )
        -
        {0, 1}
    )

    if invalid_target:

        raise ValueError(
            f"{label} contains invalid target values: "
            f"{invalid_target}"
        )

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

    if duplicates > 0:

        raise ValueError(
            f"{label} contains "
            f"{duplicates} duplicate seller/snapshot rows."
        )

    return df


def identify_features(
    df,
    target,
):
    """
    Identify numeric predictor features.

    Excludes identifiers, dates, target, and future outcome
    columns.
    """

    excluded = {
        SELLER_COLUMN,
        DATE_COLUMN,
        target,
    }

    features = [
        column
        for column
        in df.columns
        if (
            column not in excluded
            and not column.startswith(
                "future_"
            )
        )
    ]

    numeric_features = [
        feature
        for feature
        in features
        if pd.api.types.is_numeric_dtype(
            df[
                feature
            ]
        )
    ]

    if not numeric_features:

        raise ValueError(
            "No numeric predictor features found."
        )

    return numeric_features


def get_quantile_edges(
    reference,
):
    """
    Build distribution bins from the reference population.

    Quantile-based boundaries make better use of the observed
    reference distribution than arbitrary fixed intervals.
    """

    clean = (
        pd.Series(
            reference
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if clean.empty:

        return None

    if clean.nunique() <= 1:

        return None

    quantiles = np.linspace(
        0,
        1,
        NUMBER_OF_BINS + 1,
    )

    edges = (
        clean
        .quantile(
            quantiles
        )
        .to_numpy()
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 3:

        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def calculate_psi(
    reference,
    comparison,
):
    """
    Calculate PSI using bins derived only from the reference
    population.

    PSI formula:

        sum(
            (comparison_pct - reference_pct)
            *
            ln(
                comparison_pct
                /
                reference_pct
            )
        )

    Small epsilon values prevent division/logarithm problems
    when a bin has zero observations.
    """

    reference_series = (
        pd.Series(
            reference
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    comparison_series = (
        pd.Series(
            comparison
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if (
        reference_series.empty
        or
        comparison_series.empty
    ):

        return {
            "psi":
                np.nan,
            "reference_observations":
                len(reference_series),
            "comparison_observations":
                len(comparison_series),
            "bins":
                0,
        }

    edges = get_quantile_edges(
        reference_series
    )

    if edges is None:

        return {
            "psi":
                0.0,
            "reference_observations":
                len(reference_series),
            "comparison_observations":
                len(comparison_series),
            "bins":
                1,
        }

    reference_bins = pd.cut(
        reference_series,
        bins=edges,
        include_lowest=True,
    )

    comparison_bins = pd.cut(
        comparison_series,
        bins=edges,
        include_lowest=True,
    )

    reference_distribution = (
        reference_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    comparison_distribution = (
        comparison_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    reference_distribution = (
        reference_distribution
        .reindex(
            reference_distribution.index,
            fill_value=0,
        )
    )

    comparison_distribution = (
        comparison_distribution
        .reindex(
            reference_distribution.index,
            fill_value=0,
        )
    )

    reference_values = (
        reference_distribution
        .to_numpy()
        + EPSILON
    )

    comparison_values = (
        comparison_distribution
        .to_numpy()
        + EPSILON
    )

    psi = np.sum(
        (
            comparison_values
            -
            reference_values
        )
        *
        np.log(
            comparison_values
            /
            reference_values
        )
    )

    return {
        "psi":
            float(psi),
        "reference_observations":
            len(reference_series),
        "comparison_observations":
            len(comparison_series),
        "bins":
            len(reference_distribution),
    }


def classify_psi(
    psi,
):
    """Classify PSI magnitude."""

    if pd.isna(psi):

        return "UNAVAILABLE"

    if psi < PSI_LOW_THRESHOLD:

        return "LOW"

    if psi < PSI_MODERATE_THRESHOLD:

        return "MODERATE"

    return "HIGH"


def summarize_feature(
    feature,
    reference_df,
    comparison_df,
    reference_label,
    comparison_label,
):
    """
    Calculate descriptive distribution statistics and PSI.
    """

    reference = (
        reference_df[
            feature
        ]
    )

    comparison = (
        comparison_df[
            feature
        ]
    )

    reference_clean = (
        reference
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    comparison_clean = (
        comparison
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    psi_result = calculate_psi(
        reference,
        comparison,
    )

    reference_mean = (
        reference_clean.mean()
        if not reference_clean.empty
        else np.nan
    )

    comparison_mean = (
        comparison_clean.mean()
        if not comparison_clean.empty
        else np.nan
    )

    reference_median = (
        reference_clean.median()
        if not reference_clean.empty
        else np.nan
    )

    comparison_median = (
        comparison_clean.median()
        if not comparison_clean.empty
        else np.nan
    )

    reference_std = (
        reference_clean.std(
            ddof=0
        )
        if not reference_clean.empty
        else np.nan
    )

    comparison_std = (
        comparison_clean.std(
            ddof=0
        )
        if not comparison_clean.empty
        else np.nan
    )

    return {
        "reference_period":
            reference_label,

        "comparison_period":
            comparison_label,

        "feature":
            feature,

        "reference_missing_rate":
            reference.isna().mean(),

        "comparison_missing_rate":
            comparison.isna().mean(),

        "reference_mean":
            reference_mean,

        "comparison_mean":
            comparison_mean,

        "mean_change":
            (
                comparison_mean
                -
                reference_mean
            ),

        "reference_median":
            reference_median,

        "comparison_median":
            comparison_median,

        "median_change":
            (
                comparison_median
                -
                reference_median
            ),

        "reference_std":
            reference_std,

        "comparison_std":
            comparison_std,

        "psi":
            psi_result[
                "psi"
            ],

        "psi_interpretation":
            classify_psi(
                psi_result[
                    "psi"
                ]
            ),

        "reference_observations":
            psi_result[
                "reference_observations"
            ],

        "comparison_observations":
            psi_result[
                "comparison_observations"
            ],

        "bins":
            psi_result[
                "bins"
            ],
    }


def build_target_summary(
    dataset_label,
    reference_df,
    comparison_df,
    target,
):
    """
    Calculate target prevalence drift.
    """

    reference_rate = (
        reference_df[
            target
        ]
        .mean()
    )

    comparison_rate = (
        comparison_df[
            target
        ]
        .mean()
    )

    difference = (
        comparison_rate
        -
        reference_rate
    )

    relative_change = (
        difference
        /
        reference_rate
        if reference_rate > 0
        else np.nan
    )

    return {
        "target":
            target,

        "comparison":
            dataset_label,

        "reference_rows":
            len(reference_df),

        "comparison_rows":
            len(comparison_df),

        "reference_positive_count":
            int(
                reference_df[
                    target
                ].sum()
            ),

        "comparison_positive_count":
            int(
                comparison_df[
                    target
                ].sum()
            ),

        "reference_positive_rate":
            reference_rate,

        "comparison_positive_rate":
            comparison_rate,

        "absolute_rate_change":
            difference,

        "relative_rate_change":
            relative_change,
    }


def print_target_summary(
    dataset_label,
    summary,
):
    """Print target drift summary."""

    print(
        f"{dataset_label.upper()}:"
    )

    print(
        f"Reference positive rate: "
        f"{summary['reference_positive_rate'] * 100:.2f}%"
    )

    print(
        f"Comparison positive rate: "
        f"{summary['comparison_positive_rate'] * 100:.2f}%"
    )

    print(
        f"Absolute change: "
        f"{summary['absolute_rate_change'] * 100:+.2f} pp"
    )

    if pd.notna(
        summary[
            "relative_rate_change"
        ]
    ):

        print(
            f"Relative change: "
            f"{summary['relative_rate_change'] * 100:+.2f}%"
        )


def run_feature_drift_analysis(
    model_name,
    train_df,
    validation_df,
    test_df,
    target,
):
    """
    Compare training distribution against validation and test.
    """

    features = identify_features(
        train_df,
        target,
    )

    validation_features = identify_features(
        validation_df,
        target,
    )

    test_features = identify_features(
        test_df,
        target,
    )

    if features != validation_features:

        raise ValueError(
            f"{model_name} validation feature set "
            "differs from training."
        )

    if features != test_features:

        raise ValueError(
            f"{model_name} test feature set "
            "differs from training."
        )

    results = []

    for feature in features:

        validation_result = summarize_feature(
            feature,
            train_df,
            validation_df,
            "TRAIN",
            "VALIDATION",
        )

        validation_result[
            "model_target"
        ] = target

        validation_result[
            "model_family"
        ] = model_name

        results.append(
            validation_result
        )

        test_result = summarize_feature(
            feature,
            train_df,
            test_df,
            "TRAIN",
            "TEST",
        )

        test_result[
            "model_target"
        ] = target

        test_result[
            "model_family"
        ] = model_name

        results.append(
            test_result
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TEMPORAL FEATURE DRIFT ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD DELIVERY DATA
    # ========================================================

    print_section(
        "LOADING DELIVERY DATASETS"
    )

    delivery_train = prepare_dataset(
        DELIVERY_TRAIN_FILE,
        "Delivery training dataset",
        DELIVERY_TARGET,
    )

    delivery_validation = prepare_dataset(
        DELIVERY_VALIDATION_FILE,
        "Delivery validation dataset",
        DELIVERY_TARGET,
    )

    delivery_test = prepare_dataset(
        DELIVERY_TEST_FILE,
        "Delivery test dataset",
        DELIVERY_TARGET,
    )

    print(
        "Delivery datasets -> PASS"
    )

    # ========================================================
    # 2. LOAD REVIEW DATA
    # ========================================================

    print_section(
        "LOADING REVIEW DATASETS"
    )

    review_train = prepare_dataset(
        REVIEW_TRAIN_FILE,
        "Review training dataset",
        REVIEW_TARGET,
    )

    review_validation = prepare_dataset(
        REVIEW_VALIDATION_FILE,
        "Review validation dataset",
        REVIEW_TARGET,
    )

    review_test = prepare_dataset(
        REVIEW_TEST_FILE,
        "Review test dataset",
        REVIEW_TARGET,
    )

    print(
        "Review datasets -> PASS"
    )

    # ========================================================
    # 3. TEMPORAL BOUNDARIES
    # ========================================================

    print_section(
        "VALIDATING TEMPORAL BOUNDARIES"
    )

    datasets = [
        (
            "Delivery training",
            delivery_train,
        ),
        (
            "Delivery validation",
            delivery_validation,
        ),
        (
            "Delivery test",
            delivery_test,
        ),
        (
            "Review training",
            review_train,
        ),
        (
            "Review validation",
            review_validation,
        ),
        (
            "Review test",
            review_test,
        ),
    ]

    for label, df in datasets:

        print(
            f"{label}: "
            f"{df[DATE_COLUMN].min().date()} "
            f"-> "
            f"{df[DATE_COLUMN].max().date()}"
        )

    # ========================================================
    # 4. TARGET DRIFT
    # ========================================================

    print_section(
        "TARGET PREVALENCE DRIFT"
    )

    delivery_validation_target = (
        build_target_summary(
            "VALIDATION",
            delivery_train,
            delivery_validation,
            DELIVERY_TARGET,
        )
    )

    delivery_test_target = (
        build_target_summary(
            "TEST",
            delivery_train,
            delivery_test,
            DELIVERY_TARGET,
        )
    )

    review_validation_target = (
        build_target_summary(
            "VALIDATION",
            review_train,
            review_validation,
            REVIEW_TARGET,
        )
    )

    review_test_target = (
        build_target_summary(
            "TEST",
            review_train,
            review_test,
            REVIEW_TARGET,
        )
    )

    target_summaries = [
        delivery_validation_target,
        delivery_test_target,
        review_validation_target,
        review_test_target,
    ]

    print(
        "\nDELIVERY RISK"
    )

    print_target_summary(
        "Validation",
        delivery_validation_target,
    )

    print_target_summary(
        "Test",
        delivery_test_target,
    )

    print(
        "\nREVIEW RISK"
    )

    print_target_summary(
        "Validation",
        review_validation_target,
    )

    print_target_summary(
        "Test",
        review_test_target,
    )

    # ========================================================
    # 5. DELIVERY FEATURE DRIFT
    # ========================================================

    print_section(
        "DELIVERY FEATURE DRIFT"
    )

    delivery_results = (
        run_feature_drift_analysis(
            "DELIVERY",
            delivery_train,
            delivery_validation,
            delivery_test,
            DELIVERY_TARGET,
        )
    )

    delivery_df = pd.DataFrame(
        delivery_results
    )

    print(
        f"Delivery feature comparisons: "
        f"{len(delivery_df):,}"
    )

    # ========================================================
    # 6. REVIEW FEATURE DRIFT
    # ========================================================

    print_section(
        "REVIEW FEATURE DRIFT"
    )

    review_results = (
        run_feature_drift_analysis(
            "REVIEW",
            review_train,
            review_validation,
            review_test,
            REVIEW_TARGET,
        )
    )

    review_df = pd.DataFrame(
        review_results
    )

    print(
        f"Review feature comparisons: "
        f"{len(review_df):,}"
    )

    # ========================================================
    # 7. COMBINE
    # ========================================================

    print_section(
        "COMBINING DRIFT RESULTS"
    )

    feature_drift_df = pd.concat(
        [
            delivery_df,
            review_df,
        ],
        ignore_index=True,
    )

    print(
        f"Total feature drift comparisons: "
        f"{len(feature_drift_df):,}"
    )

    # ========================================================
    # 8. DRIFT SUMMARY
    # ========================================================

    print_section(
        "DRIFT SUMMARY"
    )

    summary_rows = []

    grouped = feature_drift_df.groupby(
        [
            "model_family",
            "comparison_period",
        ]
    )

    for (
        model_family,
        comparison_period,
    ), group in grouped:

        psi_values = (
            group[
                "psi"
            ]
            .dropna()
        )

        summary_rows.append(
            {
                "model_family":
                    model_family,

                "comparison_period":
                    comparison_period,

                "features":
                    len(group),

                "low_drift_features":
                    (
                        group[
                            "psi_interpretation"
                        ]
                        ==
                        "LOW"
                    ).sum(),

                "moderate_drift_features":
                    (
                        group[
                            "psi_interpretation"
                        ]
                        ==
                        "MODERATE"
                    ).sum(),

                "high_drift_features":
                    (
                        group[
                            "psi_interpretation"
                        ]
                        ==
                        "HIGH"
                    ).sum(),

                "mean_psi":
                    (
                        psi_values.mean()
                        if not psi_values.empty
                        else np.nan
                    ),

                "median_psi":
                    (
                        psi_values.median()
                        if not psi_values.empty
                        else np.nan
                    ),

                "maximum_psi":
                    (
                        psi_values.max()
                        if not psi_values.empty
                        else np.nan
                    ),
            }
        )

    drift_summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    print(
        drift_summary_df
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 9. TOP DRIFTED FEATURES
    # ========================================================

    print_section(
        "TOP DRIFTED FEATURES"
    )

    top_drift = (
        feature_drift_df
        .sort_values(
            [
                "comparison_period",
                "psi",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            [
                "model_family",
                "comparison_period",
            ],
            group_keys=False,
        )
        .head(10)
    )

    print(
        top_drift[
            [
                "model_family",
                "comparison_period",
                "feature",
                "psi",
                "psi_interpretation",
                "reference_mean",
                "comparison_mean",
                "mean_change",
                "reference_missing_rate",
                "comparison_missing_rate",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 10. MISSINGNESS DRIFT
    # ========================================================

    print_section(
        "MISSINGNESS DRIFT"
    )

    missingness_change = (
        feature_drift_df
        .assign(
            missing_rate_change=
                (
                    feature_drift_df[
                        "comparison_missing_rate"
                    ]
                    -
                    feature_drift_df[
                        "reference_missing_rate"
                    ]
                )
        )
        .sort_values(
            "missing_rate_change",
            ascending=False,
        )
    )

    print(
        missingness_change[
            [
                "model_family",
                "comparison_period",
                "feature",
                "reference_missing_rate",
                "comparison_missing_rate",
                "missing_rate_change",
            ]
        ]
        .head(20)
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 11. INTERPRETATION
    # ========================================================

    print_section(
        "TEMPORAL DRIFT INTERPRETATION"
    )

    print(
        "PSI interpretation used:"
    )

    print(
        f"Below {PSI_LOW_THRESHOLD:.2f}: LOW"
    )

    print(
        f"{PSI_LOW_THRESHOLD:.2f} to "
        f"{PSI_MODERATE_THRESHOLD:.2f}: MODERATE"
    )

    print(
        f"Above {PSI_MODERATE_THRESHOLD:.2f}: HIGH"
    )

    print(
        "\nImportant:"
    )

    print(
        "PSI indicates distribution movement."
    )

    print(
        "It does not prove that drift caused model failure."
    )

    print(
        "Feature drift, target drift, and model performance "
        "drift should be interpreted together."
    )

    # ========================================================
    # 12. SAVE
    # ========================================================

    print_section(
        "SAVING TEMPORAL DRIFT RESULTS"
    )

    feature_drift_df.to_csv(
        DETAIL_OUTPUT_FILE,
        index=False,
    )

    drift_summary_df.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    target_summary_df = (
        pd.DataFrame(
            target_summaries
        )
    )

    target_summary_df.to_csv(
        TARGET_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Feature drift detail saved to:\n"
        f"{DETAIL_OUTPUT_FILE}"
    )

    print(
        f"\nFeature drift summary saved to:\n"
        f"{SUMMARY_OUTPUT_FILE}"
    )

    print(
        f"\nTarget drift summary saved to:\n"
        f"{TARGET_OUTPUT_FILE}"
    )

    # ========================================================
    # 13. FINAL
    # ========================================================

    print_section(
        "TEMPORAL FEATURE DRIFT ANALYSIS COMPLETE"
    )

    print(
        f"Delivery features analyzed: "
        f"{delivery_df['feature'].nunique():,}"
    )

    print(
        f"Review features analyzed: "
        f"{review_df['feature'].nunique():,}"
    )

    print(
        f"Feature comparison rows: "
        f"{len(feature_drift_df):,}"
    )

    print(
        f"Target comparisons: "
        f"{len(target_summary_df):,}"
    )

    print(
        "\nNo models were trained."
    )

    print(
        "No thresholds were changed."
    )

    print(
        "No calibration was performed."
    )

    print(
        "The final test data was used only "
        "for drift diagnosis."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()