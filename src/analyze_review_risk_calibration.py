"""
Analyze Review-Risk Probability Calibration

Purpose
-------
Evaluate whether the review-risk model's predicted
probabilities correspond reasonably well to observed
future review-risk outcomes.

This is different from discrimination:

    ROC-AUC
        asks whether risky sellers tend to receive higher
        scores than non-risky sellers.

    Calibration
        asks whether predicted probability levels correspond
        to observed event frequencies.

Example:

    If sellers predicted around 20% risk have an observed
    event rate close to 20%, that probability region is
    reasonably calibrated.

Metrics
-------
- Brier score
- Reliability tables
- Mean predicted probability
- Observed event rate
- Mean absolute calibration error
- Maximum calibration error
- ROC-AUC

Data design
-----------
Training:
    in-sample diagnostic

Validation:
    primary calibration analysis

Final test:
    out-of-time diagnostic only

No calibration model is fitted.

Input
-----
data/processed/review_risk_model.joblib
data/processed/target_review_train.csv
data/processed/target_review_validation.csv
data/processed/target_review_test.csv

Outputs
-------
data/processed/review_risk_calibration_validation.csv
data/processed/review_risk_calibration_test.csv
data/processed/review_risk_calibration_summary.csv
"""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)


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

TRAIN_FILE = (
    DATA_DIR
    / "target_review_train.csv"
)

VALIDATION_FILE = (
    DATA_DIR
    / "target_review_validation.csv"
)

TEST_FILE = (
    DATA_DIR
    / "target_review_test.csv"
)

VALIDATION_OUTPUT_FILE = (
    DATA_DIR
    / "review_risk_calibration_validation.csv"
)

TEST_OUTPUT_FILE = (
    DATA_DIR
    / "review_risk_calibration_test.csv"
)

SUMMARY_OUTPUT_FILE = (
    DATA_DIR
    / "review_risk_calibration_summary.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_review_risk"


# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================

NUMBER_OF_BINS = 10


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
    label,
):
    """Convert and validate snapshot dates."""

    if DATE_COLUMN not in df.columns:

        raise ValueError(
            f"{label} does not contain "
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

    print(
        f"{label} invalid dates: "
        f"{invalid}"
    )

    if invalid > 0:

        raise ValueError(
            f"{label} contains invalid snapshot dates."
        )


def validate_target(
    df,
    label,
):
    """Validate the binary future review-risk target."""

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"{label} does not contain "
            f"{TARGET_COLUMN}"
        )

    missing = (
        df[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    invalid = (
        set(
            df[
                TARGET_COLUMN
            ]
            .dropna()
            .unique()
        )
        -
        {0, 1}
    )

    print(
        f"{label} missing targets: "
        f"{missing}"
    )

    print(
        f"{label} invalid targets: "
        f"{len(invalid)}"
    )

    if missing > 0:

        raise ValueError(
            f"{label} contains missing target values."
        )

    if invalid:

        raise ValueError(
            f"{label} contains invalid target values: "
            f"{invalid}"
        )


def load_model_package():
    """Load and validate the saved review-risk model package."""

    package = joblib.load(
        MODEL_FILE
    )

    required_fields = [
        "model",
        "features",
        "target",
        "threshold",
    ]

    missing = [
        field
        for field in required_fields
        if field not in package
    ]

    if missing:

        raise ValueError(
            "Review model package missing:\n"
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
            "Saved review model target does not match "
            "future_review_risk."
        )

    return package


def generate_probabilities(
    package,
    df,
):
    """Generate class-1 review-risk probabilities."""

    model = package[
        "model"
    ]

    features = package[
        "features"
    ]

    missing_features = [
        feature
        for feature
        in features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Dataset missing review model features:\n"
            +
            "\n".join(
                missing_features
            )
        )

    X = (
        df[
            features
        ]
        .copy()
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    if (
        np.any(
            probabilities < 0
        )
        or
        np.any(
            probabilities > 1
        )
    ):

        raise ValueError(
            "Model generated probabilities outside [0, 1]."
        )

    return probabilities


def build_reliability_table(
    probabilities,
    y_true,
    number_of_bins=10,
):
    """
    Build equal-width probability bins.

    Bins:

        0.0-0.1
        0.1-0.2
        ...
        0.9-1.0

    The final bin includes probability = 1.0.
    """

    probability_series = pd.Series(
        probabilities,
        name="predicted_probability",
    )

    target_series = pd.Series(
        y_true,
        name="actual_target",
    )

    bin_edges = np.linspace(
        0,
        1,
        number_of_bins + 1,
    )

    bin_labels = [
        f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
        for i in range(
            number_of_bins
        )
    ]

    bin_assignment = pd.cut(
        probability_series,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )

    bin_assignment = bin_assignment.astype(
        "object"
    )

    # Explicitly retain exact probability = 1.0.
    bin_assignment[
        probability_series
        == 1.0
    ] = bin_labels[
        -1
    ]

    calibration_df = (
        pd.DataFrame(
            {
                "probability_bin":
                    bin_assignment,
                "predicted_probability":
                    probability_series,
                "actual_target":
                    target_series,
            }
        )
        .groupby(
            "probability_bin",
            observed=False,
        )
        .agg(
            observations=(
                "actual_target",
                "size",
            ),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_event_rate=(
                "actual_target",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration_df[
        "absolute_calibration_error"
    ] = (
        (
            calibration_df[
                "mean_predicted_probability"
            ]
            -
            calibration_df[
                "observed_event_rate"
            ]
        )
        .abs()
    )

    return calibration_df


def calculate_summary(
    dataset_label,
    probabilities,
    y_true,
):
    """Calculate global probability diagnostics."""

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    mean_probability = (
        probabilities.mean()
    )

    observed_rate = (
        np.mean(
            y_true
        )
    )

    overall_gap = abs(
        mean_probability
        -
        observed_rate
    )

    return {
        "dataset":
            dataset_label,

        "observations":
            len(y_true),

        "mean_predicted_probability":
            mean_probability,

        "observed_event_rate":
            observed_rate,

        "overall_probability_rate_difference":
            overall_gap,

        "brier_score":
            brier,

        "roc_auc":
            auc,
    }


def print_calibration_interpretation(
    calibration_df,
):
    """Print aggregate reliability diagnostics."""

    populated = (
        calibration_df[
            calibration_df[
                "observations"
            ]
            > 0
        ]
    )

    if populated.empty:

        print(
            "No populated calibration bins."
        )

        return

    mean_absolute_error = (
        populated[
            "absolute_calibration_error"
        ]
        .mean()
    )

    max_absolute_error = (
        populated[
            "absolute_calibration_error"
        ]
        .max()
    )

    print(
        f"Mean absolute bin calibration error: "
        f"{mean_absolute_error:.4f}"
    )

    print(
        f"Maximum bin calibration error: "
        f"{max_absolute_error:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "REVIEW RISK PROBABILITY CALIBRATION ANALYSIS"
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
        TRAIN_FILE,
        "Review training data",
    )

    validate_file(
        VALIDATION_FILE,
        "Review validation data",
    )

    validate_file(
        TEST_FILE,
        "Review testing data",
    )

    print(
        "Review model -> FOUND"
    )

    print(
        "Training data -> FOUND"
    )

    print(
        "Validation data -> FOUND"
    )

    print(
        "Testing data -> FOUND"
    )

    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    print_section(
        "LOADING SAVED REVIEW MODEL"
    )

    package = load_model_package()

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

    print(
        "Model package -> PASS"
    )

    print(
        f"Predictor features: "
        f"{len(features)}"
    )

    print(
        f"Saved operating threshold: "
        f"{threshold:.2f}"
    )

    print(
        "No model retraining will occur."
    )

    # ========================================================
    # 3. LOAD DATA
    # ========================================================

    print_section(
        "LOADING TEMPORAL DATA"
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
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Testing rows: "
        f"{len(test_df):,}"
    )

    # ========================================================
    # 4. VALIDATION
    # ========================================================

    print_section(
        "DATA VALIDATION"
    )

    for label, dataset in [
        (
            "Training",
            train_df,
        ),
        (
            "Validation",
            validation_df,
        ),
        (
            "Testing",
            test_df,
        ),
    ]:

        prepare_dates(
            dataset,
            label,
        )

        validate_target(
            dataset,
            label,
        )

    print(
        "Temporal data validation -> PASS"
    )

    # ========================================================
    # 5. TEST PROTECTION
    # ========================================================

    print_section(
        "FINAL TEST PROTECTION"
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
        f"Final test period: "
        f"{test_start.date()} "
        f"-> "
        f"{test_end.date()}"
    )

    print(
        "The final test period is used only "
        "for out-of-time diagnostics."
    )

    # ========================================================
    # 6. GENERATE PROBABILITIES
    # ========================================================

    print_section(
        "GENERATING REVIEW-RISK PROBABILITIES"
    )

    train_probabilities = (
        generate_probabilities(
            package,
            train_df,
        )
    )

    validation_probabilities = (
        generate_probabilities(
            package,
            validation_df,
        )
    )

    test_probabilities = (
        generate_probabilities(
            package,
            test_df,
        )
    )

    print(
        f"Training probabilities: "
        f"{len(train_probabilities):,}"
    )

    print(
        f"Validation probabilities: "
        f"{len(validation_probabilities):,}"
    )

    print(
        f"Testing probabilities: "
        f"{len(test_probabilities):,}"
    )

    # ========================================================
    # 7. PROBABILITY DISTRIBUTION
    # ========================================================

    print_section(
        "PROBABILITY DISTRIBUTION"
    )

    for label, probabilities in [
        (
            "Training",
            train_probabilities,
        ),
        (
            "Validation",
            validation_probabilities,
        ),
        (
            "Testing",
            test_probabilities,
        ),
    ]:

        print(
            f"\n{label}:"
        )

        print(
            f"Minimum: "
            f"{probabilities.min():.4f}"
        )

        print(
            f"25th percentile: "
            f"{np.percentile(probabilities, 25):.4f}"
        )

        print(
            f"Median: "
            f"{np.median(probabilities):.4f}"
        )

        print(
            f"75th percentile: "
            f"{np.percentile(probabilities, 75):.4f}"
        )

        print(
            f"Maximum: "
            f"{probabilities.max():.4f}"
        )

        print(
            f"Mean: "
            f"{probabilities.mean():.4f}"
        )

    # ========================================================
    # 8. GLOBAL METRICS
    # ========================================================

    print_section(
        "OVERALL CALIBRATION METRICS"
    )

    train_summary = calculate_summary(
        "train",
        train_probabilities,
        train_df[
            TARGET_COLUMN
        ],
    )

    validation_summary = calculate_summary(
        "validation",
        validation_probabilities,
        validation_df[
            TARGET_COLUMN
        ],
    )

    test_summary = calculate_summary(
        "test",
        test_probabilities,
        test_df[
            TARGET_COLUMN
        ],
    )

    summaries = [
        train_summary,
        validation_summary,
        test_summary,
    ]

    for summary in summaries:

        print(
            f"\n{summary['dataset'].upper()}"
        )

        print(
            f"Observations: "
            f"{summary['observations']:,}"
        )

        print(
            f"Mean predicted probability: "
            f"{summary['mean_predicted_probability']:.4f}"
        )

        print(
            f"Observed event rate: "
            f"{summary['observed_event_rate']:.4f}"
        )

        print(
            f"Overall probability-rate difference: "
            f"{summary['overall_probability_rate_difference']:.4f}"
        )

        print(
            f"Brier score: "
            f"{summary['brier_score']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{summary['roc_auc']:.4f}"
        )

    # ========================================================
    # 9. VALIDATION RELIABILITY
    # ========================================================

    print_section(
        "VALIDATION RELIABILITY ANALYSIS"
    )

    validation_calibration = (
        build_reliability_table(
            validation_probabilities,
            validation_df[
                TARGET_COLUMN
            ].to_numpy(),
            NUMBER_OF_BINS,
        )
    )

    print(
        validation_calibration
        .round(4)
        .to_string(
            index=False
        )
    )

    print_calibration_interpretation(
        validation_calibration
    )

    # ========================================================
    # 10. TEST RELIABILITY
    # ========================================================

    print_section(
        "FINAL TEST RELIABILITY ANALYSIS"
    )

    test_calibration = (
        build_reliability_table(
            test_probabilities,
            test_df[
                TARGET_COLUMN
            ].to_numpy(),
            NUMBER_OF_BINS,
        )
    )

    print(
        test_calibration
        .round(4)
        .to_string(
            index=False
        )
    )

    print_calibration_interpretation(
        test_calibration
    )

    # ========================================================
    # 11. BASE-RATE CHECK
    # ========================================================

    print_section(
        "PROBABILITY / BASE-RATE CHECK"
    )

    validation_rate = (
        validation_df[
            TARGET_COLUMN
        ]
        .mean()
    )

    test_rate = (
        test_df[
            TARGET_COLUMN
        ]
        .mean()
    )

    validation_mean_probability = (
        validation_probabilities.mean()
    )

    test_mean_probability = (
        test_probabilities.mean()
    )

    print(
        f"Validation observed review-risk rate: "
        f"{validation_rate:.4f}"
    )

    print(
        f"Validation mean predicted risk: "
        f"{validation_mean_probability:.4f}"
    )

    print(
        f"Test observed review-risk rate: "
        f"{test_rate:.4f}"
    )

    print(
        f"Test mean predicted risk: "
        f"{test_mean_probability:.4f}"
    )

    # ========================================================
    # 12. THRESHOLD CONTEXT
    # ========================================================

    print_section(
        "OPERATING THRESHOLD CONTEXT"
    )

    print(
        f"Saved operating threshold: "
        f"{threshold:.2f}"
    )

    validation_above_threshold = (
        (
            validation_probabilities
            >= threshold
        )
        .mean()
        * 100
    )

    test_above_threshold = (
        (
            test_probabilities
            >= threshold
        )
        .mean()
        * 100
    )

    print(
        f"Validation observations at/above threshold: "
        f"{validation_above_threshold:.2f}%"
    )

    print(
        f"Test observations at/above threshold: "
        f"{test_above_threshold:.2f}%"
    )

    print(
        "\nThis describes operating-point behavior."
    )

    print(
        "It does not establish that the threshold probability "
        "is a calibrated probability."
    )

    # ========================================================
    # 13. INTERPRETATION
    # ========================================================

    print_section(
        "CALIBRATION INTERPRETATION"
    )

    validation_brier = (
        validation_summary[
            "brier_score"
        ]
    )

    test_brier = (
        test_summary[
            "brier_score"
        ]
    )

    validation_gap = (
        validation_summary[
            "overall_probability_rate_difference"
        ]
    )

    test_gap = (
        test_summary[
            "overall_probability_rate_difference"
        ]
    )

    print(
        "Brier score:"
    )

    print(
        "Lower values indicate better probabilistic accuracy."
    )

    print(
        f"\nValidation Brier score: "
        f"{validation_brier:.4f}"
    )

    print(
        f"Test Brier score: "
        f"{test_brier:.4f}"
    )

    print(
        "\nMean predicted probability vs observed rate:"
    )

    print(
        f"Validation absolute gap: "
        f"{validation_gap:.4f}"
    )

    print(
        f"Test absolute gap: "
        f"{test_gap:.4f}"
    )

    print(
        "\nImportant:"
    )

    print(
        "A useful ROC-AUC does not automatically imply "
        "well-calibrated probability estimates."
    )

    print(
        "The reliability tables are therefore the key diagnostic."
    )

    # ========================================================
    # 14. SAVE RESULTS
    # ========================================================

    print_section(
        "SAVING CALIBRATION RESULTS"
    )

    validation_calibration.to_csv(
        VALIDATION_OUTPUT_FILE,
        index=False,
    )

    test_calibration.to_csv(
        TEST_OUTPUT_FILE,
        index=False,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Validation calibration saved to:\n"
        f"{VALIDATION_OUTPUT_FILE}"
    )

    print(
        f"\nTest calibration saved to:\n"
        f"{TEST_OUTPUT_FILE}"
    )

    print(
        f"\nCalibration summary saved to:\n"
        f"{SUMMARY_OUTPUT_FILE}"
    )

    # ========================================================
    # 15. FINAL SUMMARY
    # ========================================================

    print_section(
        "REVIEW RISK CALIBRATION ANALYSIS COMPLETE"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_summary['roc_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_summary['roc_auc']:.4f}"
    )

    print(
        f"Validation Brier score: "
        f"{validation_brier:.4f}"
    )

    print(
        f"Test Brier score: "
        f"{test_brier:.4f}"
    )

    print(
        "\nNo model was retrained."
    )

    print(
        "No calibration model was fitted."
    )

    print(
        "The final test period was used only "
        "as an out-of-time diagnostic."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()