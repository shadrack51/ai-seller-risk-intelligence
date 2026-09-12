"""
Analyze Review-Risk Model Errors

Purpose
-------
Investigate where the review-risk model succeeds and fails
on the untouched out-of-time test period.

The analysis separates:

    True Positives
    True Negatives
    False Positives
    False Negatives

It also examines:

    - probability distributions by error type
    - high-confidence false positives
    - false negatives near the operating threshold
    - temporal concentration of errors
    - feature characteristics of each error group

Important
---------
The saved review-risk model is used exactly as trained.

No retraining occurs.
No threshold is changed.
The final test period is diagnostic only.

Input
-----
data/processed/review_risk_model.joblib
data/processed/target_review_test.csv

Output
------
data/processed/review_risk_error_analysis.csv
data/processed/review_risk_error_summary.csv
data/processed/review_risk_error_by_snapshot.csv
"""


from pathlib import Path

import joblib
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

MODEL_FILE = (
    DATA_DIR
    / "review_risk_model.joblib"
)

TEST_FILE = (
    DATA_DIR
    / "target_review_test.csv"
)

ERROR_OUTPUT_FILE = (
    DATA_DIR
    / "review_risk_error_analysis.csv"
)

SUMMARY_OUTPUT_FILE = (
    DATA_DIR
    / "review_risk_error_summary.csv"
)

SNAPSHOT_OUTPUT_FILE = (
    DATA_DIR
    / "review_risk_error_by_snapshot.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_review_risk"


# ============================================================
# ERROR ANALYSIS SETTINGS
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 0.80


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
    """Ensure a required file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def prepare_dates(
    df,
):
    """Convert snapshot dates and validate them."""

    if DATE_COLUMN not in df.columns:

        raise ValueError(
            f"Missing date column: "
            f"{DATE_COLUMN}"
        )

    df[
        DATE_COLUMN
    ] = pd.to_datetime(
        df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    invalid = (
        df[
            DATE_COLUMN
        ]
        .isna()
        .sum()
    )

    if invalid > 0:

        raise ValueError(
            f"Invalid snapshot dates: "
            f"{invalid}"
        )


def validate_target(
    df,
):
    """Validate the binary review-risk target."""

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Missing target column: "
            f"{TARGET_COLUMN}"
        )

    if (
        df[
            TARGET_COLUMN
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing target values found."
        )

    invalid = (
        set(
            df[
                TARGET_COLUMN
            ]
            .unique()
        )
        -
        {0, 1}
    )

    if invalid:

        raise ValueError(
            f"Invalid target values: "
            f"{invalid}"
        )


def validate_grain(
    df,
):
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

    if duplicates > 0:

        raise ValueError(
            "Duplicate seller/snapshot rows detected: "
            f"{duplicates}"
        )


def validate_model_package(
    package,
):
    """Validate the saved review model package."""

    required = [
        "model",
        "features",
        "target",
        "threshold",
    ]

    missing = [
        field
        for field
        in required
        if field not in package
    ]

    if missing:

        raise ValueError(
            "Model package missing:\n"
            +
            "\n".join(
                missing
            )
        )

    if (
        package[
            "target"
        ]
        !=
        TARGET_COLUMN
    ):

        raise ValueError(
            "Model target does not match "
            "future_review_risk."
        )


def generate_predictions(
    package,
    test_df,
):
    """
    Generate review-risk probabilities and binary
    decisions using the saved operating threshold.
    """

    model = package[
        "model"
    ]

    features = package[
        "features"
    ]

    threshold = float(
        package[
            "threshold"
        ]
    )

    missing_features = [
        feature
        for feature
        in features
        if feature not in test_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Test dataset missing model features:\n"
            +
            "\n".join(
                missing_features
            )
        )

    X_test = (
        test_df[
            features
        ]
        .copy()
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    return (
        probabilities,
        predictions,
        features,
        threshold,
    )


def classify_error(
    actual,
    predicted,
):
    """Convert actual/predicted states into an error category."""

    if (
        actual == 1
        and
        predicted == 1
    ):

        return "TRUE_POSITIVE"

    if (
        actual == 0
        and
        predicted == 0
    ):

        return "TRUE_NEGATIVE"

    if (
        actual == 0
        and
        predicted == 1
    ):

        return "FALSE_POSITIVE"

    if (
        actual == 1
        and
        predicted == 0
    ):

        return "FALSE_NEGATIVE"

    raise ValueError(
        "Unknown prediction state."
    )


def summarize_group(
    df,
    error_type,
    feature_columns,
):
    """Summarize one confusion-matrix group."""

    subset = df[
        df[
            "error_type"
        ]
        ==
        error_type
    ].copy()

    row = {
        "error_type":
            error_type,

        "observations":
            len(subset),

        "percentage_of_test":
            (
                len(subset)
                /
                len(df)
                *
                100
            ),

        "mean_probability":
            (
                subset[
                    "risk_probability"
                ].mean()
                if not subset.empty
                else np.nan
            ),

        "median_probability":
            (
                subset[
                    "risk_probability"
                ].median()
                if not subset.empty
                else np.nan
            ),
    }

    for feature in feature_columns:

        if feature not in subset.columns:

            continue

        if subset.empty:

            row[
                f"{feature}_mean"
            ] = np.nan

        else:

            row[
                f"{feature}_mean"
            ] = subset[
                feature
            ].mean()

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "REVIEW RISK MODEL ERROR ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # 1. INPUT CHECKS
    # ========================================================

    print_section(
        "CHECKING REQUIRED INPUTS"
    )

    validate_file(
        MODEL_FILE,
        "Review risk model",
    )

    validate_file(
        TEST_FILE,
        "Review risk test dataset",
    )

    print(
        "Review risk model -> FOUND"
    )

    print(
        "Review risk test dataset -> FOUND"
    )

    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    print_section(
        "LOADING SAVED REVIEW MODEL"
    )

    package = joblib.load(
        MODEL_FILE
    )

    validate_model_package(
        package
    )

    print(
        "Model package -> PASS"
    )

    print(
        f"Predictor features: "
        f"{len(package['features'])}"
    )

    print(
        f"Saved threshold: "
        f"{package['threshold']:.2f}"
    )

    print(
        "No model retraining will occur."
    )

    # ========================================================
    # 3. LOAD TEST DATA
    # ========================================================

    print_section(
        "LOADING FINAL TEST DATA"
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    # ========================================================
    # 4. VALIDATION
    # ========================================================

    print_section(
        "TEST DATA VALIDATION"
    )

    prepare_dates(
        test_df
    )

    validate_target(
        test_df
    )

    validate_grain(
        test_df
    )

    print(
        "Date validation -> PASS"
    )

    print(
        "Target validation -> PASS"
    )

    print(
        "Seller/snapshot grain -> PASS"
    )

    # ========================================================
    # 5. FINAL TEST PERIOD
    # ========================================================

    print_section(
        "FINAL TEST PERIOD"
    )

    test_start = (
        test_df[
            DATE_COLUMN
        ]
        .min()
    )

    test_end = (
        test_df[
            DATE_COLUMN
        ]
        .max()
    )

    print(
        f"Test starts: "
        f"{test_start.date()}"
    )

    print(
        f"Test ends: "
        f"{test_end.date()}"
    )

    print(
        "The final test period is being analyzed only."
    )

    print(
        "No model modification will occur."
    )

    # ========================================================
    # 6. GENERATE PREDICTIONS
    # ========================================================

    print_section(
        "GENERATING TEST PREDICTIONS"
    )

    (
        probabilities,
        predictions,
        features,
        threshold,
    ) = generate_predictions(
        package,
        test_df,
    )

    test_df[
        "risk_probability"
    ] = probabilities

    test_df[
        "predicted_review_risk"
    ] = predictions

    print(
        f"Probabilities generated: "
        f"{len(probabilities):,}"
    )

    print(
        f"Predictions generated: "
        f"{len(predictions):,}"
    )

    # ========================================================
    # 7. CLASSIFY ERROR TYPES
    # ========================================================

    print_section(
        "CLASSIFYING PREDICTION OUTCOMES"
    )

    test_df[
        "error_type"
    ] = [
        classify_error(
            actual,
            predicted,
        )
        for actual, predicted
        in zip(
            test_df[
                TARGET_COLUMN
            ],
            test_df[
                "predicted_review_risk"
            ],
        )
    ]

    error_counts = (
        test_df[
            "error_type"
        ]
        .value_counts()
        .reindex(
            [
                "TRUE_NEGATIVE",
                "FALSE_POSITIVE",
                "FALSE_NEGATIVE",
                "TRUE_POSITIVE",
            ],
            fill_value=0,
        )
    )

    print(
        error_counts
    )

    # ========================================================
    # 8. ERROR DISTRIBUTION
    # ========================================================

    print_section(
        "ERROR DISTRIBUTION"
    )

    error_percentages = (
        (
            error_counts
            /
            len(test_df)
        )
        * 100
    ).round(2)

    for error_type in error_counts.index:

        print(
            f"{error_type}: "
            f"{error_counts[error_type]:,} "
            f"("
            f"{error_percentages[error_type]:.2f}%"
            f")"
        )

    # ========================================================
    # 9. FALSE POSITIVES
    # ========================================================

    print_section(
        "FALSE POSITIVE ANALYSIS"
    )

    false_positives = test_df[
        test_df[
            "error_type"
        ]
        ==
        "FALSE_POSITIVE"
    ].copy()

    print(
        f"False positives: "
        f"{len(false_positives):,}"
    )

    if not false_positives.empty:

        print(
            f"Mean risk probability: "
            f"{false_positives['risk_probability'].mean():.4f}"
        )

        print(
            f"Median risk probability: "
            f"{false_positives['risk_probability'].median():.4f}"
        )

        print(
            f"Maximum risk probability: "
            f"{false_positives['risk_probability'].max():.4f}"
        )

    # ========================================================
    # 10. FALSE NEGATIVES
    # ========================================================

    print_section(
        "FALSE NEGATIVE ANALYSIS"
    )

    false_negatives = test_df[
        test_df[
            "error_type"
        ]
        ==
        "FALSE_NEGATIVE"
    ].copy()

    print(
        f"False negatives: "
        f"{len(false_negatives):,}"
    )

    if not false_negatives.empty:

        print(
            f"Mean risk probability: "
            f"{false_negatives['risk_probability'].mean():.4f}"
        )

        print(
            f"Median risk probability: "
            f"{false_negatives['risk_probability'].median():.4f}"
        )

        print(
            f"Maximum risk probability: "
            f"{false_negatives['risk_probability'].max():.4f}"
        )

    # ========================================================
    # 11. HIGH-CONFIDENCE FALSE POSITIVES
    # ========================================================

    print_section(
        "HIGH-CONFIDENCE FALSE POSITIVE ANALYSIS"
    )

    high_confidence_fp = (
        false_positives[
            false_positives[
                "risk_probability"
            ]
            >= HIGH_CONFIDENCE_THRESHOLD
        ]
        .copy()
    )

    print(
        f"False positives with probability >= "
        f"{HIGH_CONFIDENCE_THRESHOLD:.2f}: "
        f"{len(high_confidence_fp):,}"
    )

    if not high_confidence_fp.empty:

        print(
            "\nHighest-confidence false positives:"
        )

        print(
            high_confidence_fp[
                [
                    SELLER_COLUMN,
                    DATE_COLUMN,
                    "risk_probability",
                    TARGET_COLUMN,
                ]
            ]
            .sort_values(
                "risk_probability",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 12. FALSE NEGATIVES NEAR THRESHOLD
    # ========================================================

    print_section(
        "FALSE NEGATIVES NEAR OPERATING THRESHOLD"
    )

    near_threshold_fn = (
        false_negatives[
            false_negatives[
                "risk_probability"
            ]
            >= (
                threshold
                * 0.75
            )
        ]
        .copy()
        .sort_values(
            "risk_probability",
            ascending=False,
        )
    )

    print(
        f"False negatives at >= 75% "
        f"of the operating threshold: "
        f"{len(near_threshold_fn):,}"
    )

    if not near_threshold_fn.empty:

        print(
            "\nClosest false negatives "
            "to the decision threshold:"
        )

        print(
            near_threshold_fn[
                [
                    SELLER_COLUMN,
                    DATE_COLUMN,
                    "risk_probability",
                    TARGET_COLUMN,
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 13. PROBABILITY DISTRIBUTION BY ERROR TYPE
    # ========================================================

    print_section(
        "ERROR PROBABILITY DISTRIBUTION"
    )

    probability_summary = (
        test_df
        .groupby(
            "error_type"
        )[
            "risk_probability"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        probability_summary
        .round(4)
        .to_string()
    )

    # ========================================================
    # 14. FEATURE CHARACTERISTICS
    # ========================================================

    print_section(
        "ERROR GROUP FEATURE CHARACTERISTICS"
    )

    diagnostic_features = [
        feature
        for feature in [
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
            "recent_orders",
            "older_orders",
            "recent_revenue",
            "older_revenue",
            "revenue_change",
            "recent_average_item_price",
            "older_average_item_price",
            "recent_average_freight",
            "older_average_freight",
        ]
        if feature in test_df.columns
    ]

    group_feature_summary = []

    for error_type in [
        "TRUE_NEGATIVE",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
        "TRUE_POSITIVE",
    ]:

        subset = test_df[
            test_df[
                "error_type"
            ]
            ==
            error_type
        ]

        if subset.empty:

            continue

        for feature in diagnostic_features:

            group_feature_summary.append(
                {
                    "error_type":
                        error_type,

                    "feature":
                        feature,

                    "mean":
                        subset[
                            feature
                        ].mean(),

                    "median":
                        subset[
                            feature
                        ].median(),
                }
            )

    feature_summary_df = (
        pd.DataFrame(
            group_feature_summary
        )
    )

    if not feature_summary_df.empty:

        print(
            feature_summary_df
            .round(4)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 15. TEMPORAL ERROR DISTRIBUTION
    # ========================================================

    print_section(
        "TEMPORAL ERROR DISTRIBUTION"
    )

    snapshot_error_summary = (
        test_df
        .groupby(
            DATE_COLUMN
        )
        .agg(
            observations=(
                SELLER_COLUMN,
                "count",
            ),

            false_positives=(
                "error_type",
                lambda values:
                    (
                        values
                        ==
                        "FALSE_POSITIVE"
                    ).sum(),
            ),

            false_negatives=(
                "error_type",
                lambda values:
                    (
                        values
                        ==
                        "FALSE_NEGATIVE"
                    ).sum(),
            ),

            true_positives=(
                "error_type",
                lambda values:
                    (
                        values
                        ==
                        "TRUE_POSITIVE"
                    ).sum(),
            ),

            true_negatives=(
                "error_type",
                lambda values:
                    (
                        values
                        ==
                        "TRUE_NEGATIVE"
                    ).sum(),
            ),

            actual_risk_rate=(
                TARGET_COLUMN,
                "mean",
            ),

            mean_predicted_probability=(
                "risk_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    snapshot_error_summary[
        "false_positive_rate"
    ] = (
        snapshot_error_summary[
            "false_positives"
        ]
        /
        snapshot_error_summary[
            "observations"
        ]
    )

    snapshot_error_summary[
        "false_negative_rate"
    ] = (
        snapshot_error_summary[
            "false_negatives"
        ]
        /
        snapshot_error_summary[
            "observations"
        ]
    )

    print(
        snapshot_error_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 16. BUILD ERROR SUMMARY
    # ========================================================

    print_section(
        "BUILDING ERROR SUMMARY"
    )

    summary_rows = []

    for error_type in [
        "TRUE_NEGATIVE",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
        "TRUE_POSITIVE",
    ]:

        row = summarize_group(
            test_df,
            error_type,
            diagnostic_features,
        )

        summary_rows.append(
            row
        )

    error_summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    print(
        error_summary_df
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 17. OVERALL ERROR METRICS
    # ========================================================

    print_section(
        "OVERALL ERROR METRICS"
    )

    actual = (
        test_df[
            TARGET_COLUMN
        ]
        .to_numpy()
    )

    predicted = (
        test_df[
            "predicted_review_risk"
        ]
        .to_numpy()
    )

    tp = (
        (
            (actual == 1)
            &
            (predicted == 1)
        )
        .sum()
    )

    tn = (
        (
            (actual == 0)
            &
            (predicted == 0)
        )
        .sum()
    )

    fp = (
        (
            (actual == 0)
            &
            (predicted == 1)
        )
        .sum()
    )

    fn = (
        (
            (actual == 1)
            &
            (predicted == 0)
        )
        .sum()
    )

    precision = (
        tp
        /
        (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp
        /
        (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    false_positive_rate = (
        fp
        /
        (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn
        /
        (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    print(
        f"True positives:  "
        f"{tp:,}"
    )

    print(
        f"True negatives:  "
        f"{tn:,}"
    )

    print(
        f"False positives: "
        f"{fp:,}"
    )

    print(
        f"False negatives: "
        f"{fn:,}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"False-negative rate: "
        f"{false_negative_rate:.4f}"
    )

    # ========================================================
    # 18. INTERPRETATION
    # ========================================================

    print_section(
        "ERROR ANALYSIS INTERPRETATION"
    )

    print(
        "False positives are sellers flagged by the model "
        "that do not become future review risk."
    )

    print(
        "False negatives are sellers that become future "
        "review risk but were not flagged."
    )

    print(
        "False negatives are especially important for an "
        "early-warning system because they represent missed risk."
    )

    print(
        "\nHigh-confidence false positives reveal situations "
        "where the model is strongly convinced but wrong."
    )

    print(
        "False negatives near the threshold show cases "
        "where the model almost flagged the future-risk seller."
    )

    print(
        "\nThis analysis is diagnostic only."
    )

    print(
        "The model and threshold are not modified."
    )

    # ========================================================
    # 19. SAVE DETAILED ERROR DATA
    # ========================================================

    print_section(
        "SAVING ERROR ANALYSIS"
    )

    output_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
        "risk_probability",
        "predicted_review_risk",
        "error_type",
    ]

    output_columns = [
        column
        for column
        in output_columns
        if column in test_df.columns
    ]

    test_df[
        output_columns
    ].to_csv(
        ERROR_OUTPUT_FILE,
        index=False,
    )

    error_summary_df.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    snapshot_error_summary.to_csv(
        SNAPSHOT_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Detailed error analysis saved to:\n"
        f"{ERROR_OUTPUT_FILE}"
    )

    print(
        f"\nError summary saved to:\n"
        f"{SUMMARY_OUTPUT_FILE}"
    )

    print(
        f"\nSnapshot error analysis saved to:\n"
        f"{SNAPSHOT_OUTPUT_FILE}"
    )

    # ========================================================
    # 20. FINAL SUMMARY
    # ========================================================

    print_section(
        "REVIEW RISK ERROR ANALYSIS COMPLETE"
    )

    print(
        f"Test observations analyzed: "
        f"{len(test_df):,}"
    )

    print(
        f"False positives: "
        f"{fp:,}"
    )

    print(
        f"False negatives: "
        f"{fn:,}"
    )

    print(
        f"True positives: "
        f"{tp:,}"
    )

    print(
        f"True negatives: "
        f"{tn:,}"
    )

    print(
        f"High-confidence false positives: "
        f"{len(high_confidence_fp):,}"
    )

    print(
        "\nNo model was retrained."
    )

    print(
        "No threshold was changed."
    )

    print(
        "The final test period remains diagnostic-only."
    )

    print(
        "The next stage is temporal drift analysis."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()