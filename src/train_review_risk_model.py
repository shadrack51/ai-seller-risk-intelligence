"""
Train Target-Specific Review Risk Model

Purpose
-------
Train a review-specific early-warning Random Forest for:

    future_review_risk

The workflow is:

    TRAIN
       ↓
    VALIDATION
       ↓
    select operating threshold
       ↓
    freeze threshold
       ↓
    FINAL OUT-OF-TIME TEST

The final test period is never used for model or threshold
selection.

Expected inputs:

    target_review_train.csv
    target_review_validation.csv
    target_review_test.csv

Expected target:

    future_review_risk

Model:

    Median imputation
    +
    Random Forest classifier

Hyperparameters:

    n_estimators = 300
    max_depth = None
    min_samples_leaf = 2
    random_state = 42

Outputs:

    review_risk_model.joblib
    review_risk_validation_results.csv
    review_risk_test_results.csv
    review_risk_threshold_analysis.csv
"""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
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

MODEL_FILE = (
    DATA_DIR
    / "review_risk_model.joblib"
)

VALIDATION_RESULTS_FILE = (
    DATA_DIR
    / "review_risk_validation_results.csv"
)

TEST_RESULTS_FILE = (
    DATA_DIR
    / "review_risk_test_results.csv"
)

THRESHOLD_RESULTS_FILE = (
    DATA_DIR
    / "review_risk_threshold_analysis.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_review_risk"


# ============================================================
# PROTECTED FINAL TEST PERIOD
# ============================================================

FINAL_TEST_START = pd.Timestamp(
    "2018-06-01"
)

FINAL_TEST_END = pd.Timestamp(
    "2018-08-01"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 2


# ============================================================
# THRESHOLD GRID
# ============================================================

THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.91,
        0.05,
    ),
    2,
)


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

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
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
    """Validate the binary review-risk target."""

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
        f"{label} missing target values: "
        f"{missing}"
    )

    print(
        f"{label} invalid target values: "
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


def validate_grain(
    df,
    label,
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

    print(
        f"{label} duplicate seller/snapshot rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"{label} contains duplicate "
            "seller/snapshot observations."
        )


def get_features(
    df,
):
    """
    Identify predictor features.

    Excludes:

    - seller_id
    - snapshot_date
    - future_review_risk
    - all future-prefixed columns
    """

    excluded = {
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    }

    features = [
        column
        for column in df.columns
        if (
            column not in excluded
            and not column.startswith(
                "future_"
            )
        )
    ]

    if not features:

        raise ValueError(
            "No review-risk predictor features identified."
        )

    return features


def validate_features(
    df,
    features,
):
    """Validate existence, numeric type, and leakage."""

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing predictor columns:\n"
            +
            "\n".join(
                missing
            )
        )

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
            +
            "\n".join(
                non_numeric
            )
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
            "Future information detected:\n"
            +
            "\n".join(
                future_features
            )
        )


def validate_chronology(
    train_df,
    validation_df,
    test_df,
):
    """Ensure temporal order is strictly maintained."""

    train_end = (
        train_df[
            DATE_COLUMN
        ]
        .max()
    )

    validation_start = (
        validation_df[
            DATE_COLUMN
        ]
        .min()
    )

    validation_end = (
        validation_df[
            DATE_COLUMN
        ]
        .max()
    )

    test_start = (
        test_df[
            DATE_COLUMN
        ]
        .min()
    )

    print(
        f"Training ends: "
        f"{train_end.date()}"
    )

    print(
        f"Validation starts: "
        f"{validation_start.date()}"
    )

    print(
        f"Validation ends: "
        f"{validation_end.date()}"
    )

    print(
        f"Testing starts: "
        f"{test_start.date()}"
    )

    if not (
        train_end
        <
        validation_start
        <=
        validation_end
        <
        test_start
    ):

        raise ValueError(
            "Temporal ordering failed."
        )

    print(
        "Chronological ordering -> PASS"
    )


def print_target_distribution(
    df,
    label,
):
    """Print target counts and percentages."""

    counts = (
        df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[
            TARGET_COLUMN
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)

    print(
        f"{label} counts:"
    )

    print(
        counts
    )

    print(
        f"{label} percentages:"
    )

    print(
        percentages
    )


def build_model():
    """Build the review-risk Random Forest pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_probabilities(
    model,
    X,
):
    """Return class-1 risk probabilities."""

    return (
        model.predict_proba(
            X
        )[:, 1]
    )


def evaluate_threshold(
    probabilities,
    y_true,
    threshold,
):
    """Evaluate binary predictions at one threshold."""

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def choose_threshold(
    probabilities,
    y_true,
):
    """
    Select a threshold using VALIDATION data only.

    Selection criterion:

        maximum F1 score
    """

    rows = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            probabilities,
            y_true,
            threshold,
        )

        rows.append(
            result
        )

    threshold_df = (
        pd.DataFrame(
            rows
        )
    )

    best_index = (
        threshold_df[
            "f1"
        ]
        .idxmax()
    )

    best_threshold = float(
        threshold_df.loc[
            best_index,
            "threshold",
        ]
    )

    return (
        threshold_df,
        best_threshold,
    )


def calculate_probability_metrics(
    probabilities,
    y_true,
):
    """Calculate threshold-independent metrics."""

    return {
        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "average_precision":
            average_precision_score(
                y_true,
                probabilities,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TARGET-SPECIFIC REVIEW RISK MODEL"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK INPUTS
    # ========================================================

    print_section(
        "CHECKING INPUT DATASETS"
    )

    validate_file(
        TRAIN_FILE,
        "Review training split",
    )

    validate_file(
        VALIDATION_FILE,
        "Review validation split",
    )

    validate_file(
        TEST_FILE,
        "Review testing split",
    )

    print(
        "Training split -> FOUND"
    )

    print(
        "Validation split -> FOUND"
    )

    print(
        "Testing split -> FOUND"
    )

    # ========================================================
    # 2. LOAD
    # ========================================================

    print_section(
        "LOADING REVIEW-RISK DATA"
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
    # 3. DATES
    # ========================================================

    print_section(
        "PREPARING SNAPSHOT DATES"
    )

    prepare_dates(
        train_df,
        "Training dataset",
    )

    prepare_dates(
        validation_df,
        "Validation dataset",
    )

    prepare_dates(
        test_df,
        "Testing dataset",
    )

    print(
        "Snapshot dates -> PASS"
    )

    # ========================================================
    # 4. VALIDATION
    # ========================================================

    print_section(
        "DATA VALIDATION"
    )

    for label, dataset in [
        (
            "Training dataset",
            train_df,
        ),
        (
            "Validation dataset",
            validation_df,
        ),
        (
            "Testing dataset",
            test_df,
        ),
    ]:

        validate_target(
            dataset,
            label,
        )

        validate_grain(
            dataset,
            label,
        )

    print(
        "Target validation -> PASS"
    )

    print(
        "Seller/snapshot grain -> PASS"
    )

    # ========================================================
    # 5. TARGET-SPECIFIC FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING REVIEW-RISK FEATURES"
    )

    features = get_features(
        train_df
    )

    validate_features(
        train_df,
        features,
    )

    print(
        f"Predictor features: "
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
    # 6. CROSS-SPLIT FEATURE CONSISTENCY
    # ========================================================

    print_section(
        "FEATURE CONSISTENCY ACROSS SPLITS"
    )

    for label, dataset in [
        (
            "Validation",
            validation_df,
        ),
        (
            "Testing",
            test_df,
        ),
    ]:

        split_features = (
            get_features(
                dataset
            )
        )

        if split_features != features:

            raise ValueError(
                f"{label} feature set/order "
                "differs from training."
            )

        print(
            f"{label} feature consistency -> PASS"
        )

    # ========================================================
    # 7. CHRONOLOGY
    # ========================================================

    print_section(
        "CHRONOLOGICAL VALIDATION"
    )

    validate_chronology(
        train_df,
        validation_df,
        test_df,
    )

    # ========================================================
    # 8. FINAL TEST PROTECTION
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
        f"Expected test start: "
        f"{FINAL_TEST_START.date()}"
    )

    print(
        f"Actual test start: "
        f"{test_start.date()}"
    )

    print(
        f"Expected test end: "
        f"{FINAL_TEST_END.date()}"
    )

    print(
        f"Actual test end: "
        f"{test_end.date()}"
    )

    if test_start != FINAL_TEST_START:

        raise ValueError(
            "Final test start is incorrect."
        )

    if test_end != FINAL_TEST_END:

        raise ValueError(
            "Final test end is incorrect."
        )

    print(
        "Final test period -> PROTECTED"
    )

    # ========================================================
    # 9. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTIONS"
    )

    print_target_distribution(
        train_df,
        "Training target",
    )

    print()

    print_target_distribution(
        validation_df,
        "Validation target",
    )

    print()

    print_target_distribution(
        test_df,
        "Testing target",
    )

    # ========================================================
    # 10. MATRICES
    # ========================================================

    print_section(
        "PREPARING FEATURE MATRICES"
    )

    X_train = (
        train_df[
            features
        ]
        .copy()
    )

    y_train = (
        train_df[
            TARGET_COLUMN
        ]
        .copy()
    )

    X_validation = (
        validation_df[
            features
        ]
        .copy()
    )

    y_validation = (
        validation_df[
            TARGET_COLUMN
        ]
        .copy()
    )

    X_test = (
        test_df[
            features
        ]
        .copy()
    )

    y_test = (
        test_df[
            TARGET_COLUMN
        ]
        .copy()
    )

    print(
        f"Training feature matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Validation feature matrix: "
        f"{X_validation.shape}"
    )

    print(
        f"Testing feature matrix: "
        f"{X_test.shape}"
    )

    # ========================================================
    # 11. MISSING VALUES
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing_training = (
        X_train
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_training = (
        missing_training[
            missing_training > 0
        ]
    )

    if missing_training.empty:

        print(
            "No missing training predictors."
        )

    else:

        print(
            missing_training.to_string()
        )

        print(
            "\nMedian imputation will be fitted "
            "only on training data."
        )

    # ========================================================
    # 12. BUILD MODEL
    # ========================================================

    print_section(
        "BUILDING REVIEW RISK MODEL PIPELINE"
    )

    print(
        "Pipeline:"
    )

    print(
        "1. Median imputation"
    )

    print(
        "2. Random Forest classifier"
    )

    print(
        "\nHyperparameters:"
    )

    print(
        f"Number of trees: "
        f"{N_ESTIMATORS}"
    )

    print(
        f"Maximum depth: "
        f"{MAX_DEPTH}"
    )

    print(
        f"Minimum samples per leaf: "
        f"{MIN_SAMPLES_LEAF}"
    )

    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )

    model = build_model()

    print(
        "\nReview risk model pipeline created -> PASS"
    )

    # ========================================================
    # 13. TRAIN
    # ========================================================

    print_section(
        "TRAINING REVIEW RISK MODEL"
    )

    print(
        "Fitting model on TRAINING data only..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ========================================================
    # 14. TRAINING PERFORMANCE
    # ========================================================

    print_section(
        "TRAINING PERFORMANCE"
    )

    train_probabilities = (
        get_probabilities(
            model,
            X_train,
        )
    )

    train_probability_metrics = (
        calculate_probability_metrics(
            train_probabilities,
            y_train,
        )
    )

    train_default = (
        evaluate_threshold(
            train_probabilities,
            y_train,
            0.50,
        )
    )

    print(
        f"Accuracy:           "
        f"{train_default['accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{train_default['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{train_default['recall']:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{train_default['f1']:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{train_probability_metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{train_probability_metrics['average_precision']:.4f}"
    )

    # ========================================================
    # 15. VALIDATION PERFORMANCE
    # ========================================================

    print_section(
        "VALIDATION PERFORMANCE"
    )

    validation_probabilities = (
        get_probabilities(
            model,
            X_validation,
        )
    )

    validation_probability_metrics = (
        calculate_probability_metrics(
            validation_probabilities,
            y_validation,
        )
    )

    validation_default = (
        evaluate_threshold(
            validation_probabilities,
            y_validation,
            0.50,
        )
    )

    print(
        f"Accuracy at 0.50:   "
        f"{validation_default['accuracy']:.4f}"
    )

    print(
        f"Precision at 0.50:  "
        f"{validation_default['precision']:.4f}"
    )

    print(
        f"Recall at 0.50:     "
        f"{validation_default['recall']:.4f}"
    )

    print(
        f"F1 at 0.50:         "
        f"{validation_default['f1']:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{validation_probability_metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{validation_probability_metrics['average_precision']:.4f}"
    )

    # ========================================================
    # 16. VALIDATION THRESHOLD SELECTION
    # ========================================================

    print_section(
        "VALIDATION THRESHOLD ANALYSIS"
    )

    threshold_df, selected_threshold = (
        choose_threshold(
            validation_probabilities,
            y_validation,
        )
    )

    print(
        threshold_df
        .round(4)
        .to_string(
            index=False
        )
    )

    best_row = (
        threshold_df[
            threshold_df[
                "threshold"
            ]
            ==
            selected_threshold
        ]
        .iloc[0]
    )

    print(
        "\nBEST VALIDATION THRESHOLD"
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Precision: "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{best_row['f1']:.4f}"
    )

    print(
        f"False positives: "
        f"{int(best_row['false_positives'])}"
    )

    print(
        f"False negatives: "
        f"{int(best_row['false_negatives'])}"
    )

    print(
        "\nThreshold selected using VALIDATION data only."
    )

    print(
        "The threshold is now frozen."
    )

    # ========================================================
    # 17. FINAL OUT-OF-TIME TEST
    # ========================================================

    print_section(
        "FINAL OUT-OF-TIME TEST PERFORMANCE"
    )

    print(
        "The June-August 2018 test period "
        "was not used for model or threshold selection."
    )

    test_probabilities = (
        get_probabilities(
            model,
            X_test,
        )
    )

    test_probability_metrics = (
        calculate_probability_metrics(
            test_probabilities,
            y_test,
        )
    )

    test_default = (
        evaluate_threshold(
            test_probabilities,
            y_test,
            0.50,
        )
    )

    test_selected = (
        evaluate_threshold(
            test_probabilities,
            y_test,
            selected_threshold,
        )
    )

    print(
        "\nPERFORMANCE AT DEFAULT 0.50"
    )

    print(
        f"Accuracy:           "
        f"{test_default['accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{test_default['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{test_default['recall']:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{test_default['f1']:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{test_probability_metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{test_probability_metrics['average_precision']:.4f}"
    )

    print(
        "\nPERFORMANCE AT FROZEN VALIDATION THRESHOLD"
    )

    print(
        f"Threshold:          "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Accuracy:           "
        f"{test_selected['accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{test_selected['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{test_selected['recall']:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{test_selected['f1']:.4f}"
    )

    print(
        f"False positives:    "
        f"{test_selected['false_positives']}"
    )

    print(
        f"False negatives:    "
        f"{test_selected['false_negatives']}"
    )

    # ========================================================
    # 18. CONFUSION MATRIX
    # ========================================================

    print_section(
        "FINAL TEST CONFUSION MATRIX"
    )

    test_predictions = (
        (
            test_probabilities
            >= selected_threshold
        )
        .astype(int)
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print(
        test_cm
    )

    tn, fp, fn, tp = (
        test_cm.ravel()
    )

    print(
        f"\nTrue Negatives:  {tn}"
    )

    print(
        f"False Positives: {fp}"
    )

    print(
        f"False Negatives: {fn}"
    )

    print(
        f"True Positives:  {tp}"
    )

    # ========================================================
    # 19. CLASSIFICATION REPORT
    # ========================================================

    print_section(
        "FINAL TEST CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Not Review Risk",
                "Future Review Risk",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # 20. GENERALIZATION
    # ========================================================

    print_section(
        "GENERALIZATION ANALYSIS"
    )

    train_auc = (
        train_probability_metrics[
            "roc_auc"
        ]
    )

    validation_auc = (
        validation_probability_metrics[
            "roc_auc"
        ]
    )

    test_auc = (
        test_probability_metrics[
            "roc_auc"
        ]
    )

    print(
        f"Training ROC-AUC:   "
        f"{train_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_auc:.4f}"
    )

    print(
        f"Test ROC-AUC:       "
        f"{test_auc:.4f}"
    )

    print(
        f"\nTrain → Validation gap: "
        f"{train_auc - validation_auc:.4f}"
    )

    print(
        f"Train → Test gap:       "
        f"{train_auc - test_auc:.4f}"
    )

    if test_auc < 0.60:

        print(
            "\nWARNING:"
        )

        print(
            "Final out-of-time discrimination remains weak."
        )

        print(
            "The model should not be presented as "
            "a high-confidence review-risk predictor."
        )

    # ========================================================
    # 21. FEATURE IMPORTANCE
    # ========================================================

    print_section(
        "REVIEW MODEL FEATURE IMPORTANCE"
    )

    rf_model = (
        model.named_steps[
            "model"
        ]
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                features,

            "importance":
                rf_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance_df
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 22. BUILD VALIDATION RESULTS
    # ========================================================

    print_section(
        "BUILDING VALIDATION RESULTS"
    )

    validation_result = {
        "validation_start":
            validation_df[
                DATE_COLUMN
            ].min(),

        "validation_end":
            validation_df[
                DATE_COLUMN
            ].max(),

        "rows":
            len(validation_df),

        "roc_auc":
            validation_probability_metrics[
                "roc_auc"
            ],

        "average_precision":
            validation_probability_metrics[
                "average_precision"
            ],

        "threshold":
            selected_threshold,

        "accuracy":
            best_row[
                "accuracy"
            ],

        "precision":
            best_row[
                "precision"
            ],

        "recall":
            best_row[
                "recall"
            ],

        "f1":
            best_row[
                "f1"
            ],

        "false_positives":
            int(
                best_row[
                    "false_positives"
                ]
            ),

        "false_negatives":
            int(
                best_row[
                    "false_negatives"
                ]
            ),

        "true_negatives":
            int(
                best_row[
                    "true_negatives"
                ]
            ),

        "true_positives":
            int(
                best_row[
                    "true_positives"
                ]
            ),
    }

    validation_results_df = (
        pd.DataFrame(
            [
                validation_result
            ]
        )
    )

    # ========================================================
    # 23. BUILD TEST RESULTS
    # ========================================================

    print_section(
        "BUILDING FINAL TEST RESULTS"
    )

    test_result = {
        "test_start":
            test_df[
                DATE_COLUMN
            ].min(),

        "test_end":
            test_df[
                DATE_COLUMN
            ].max(),

        "rows":
            len(test_df),

        "threshold":
            selected_threshold,

        "roc_auc":
            test_probability_metrics[
                "roc_auc"
            ],

        "average_precision":
            test_probability_metrics[
                "average_precision"
            ],

        "accuracy":
            test_selected[
                "accuracy"
            ],

        "precision":
            test_selected[
                "precision"
            ],

        "recall":
            test_selected[
                "recall"
            ],

        "f1":
            test_selected[
                "f1"
            ],

        "false_positives":
            test_selected[
                "false_positives"
            ],

        "false_negatives":
            test_selected[
                "false_negatives"
            ],

        "true_negatives":
            test_selected[
                "true_negatives"
            ],

        "true_positives":
            test_selected[
                "true_positives"
            ],
    }

    test_results_df = (
        pd.DataFrame(
            [
                test_result
            ]
        )
    )

    # ========================================================
    # 24. SAVE THRESHOLD ANALYSIS
    # ========================================================

    print_section(
        "SAVING THRESHOLD ANALYSIS"
    )

    threshold_df.to_csv(
        THRESHOLD_RESULTS_FILE,
        index=False,
    )

    print(
        f"Threshold analysis saved to:\n"
        f"{THRESHOLD_RESULTS_FILE}"
    )

    # ========================================================
    # 25. SAVE RESULTS
    # ========================================================

    print_section(
        "SAVING MODEL RESULTS"
    )

    validation_results_df.to_csv(
        VALIDATION_RESULTS_FILE,
        index=False,
    )

    test_results_df.to_csv(
        TEST_RESULTS_FILE,
        index=False,
    )

    print(
        f"Validation results saved to:\n"
        f"{VALIDATION_RESULTS_FILE}"
    )

    print(
        f"\nTest results saved to:\n"
        f"{TEST_RESULTS_FILE}"
    )

    # ========================================================
    # 26. SAVE MODEL PACKAGE
    # ========================================================

    print_section(
        "SAVING REVIEW RISK MODEL"
    )

    model_package = {
        "model":
            model,

        "features":
            features,

        "target":
            TARGET_COLUMN,

        "threshold":
            selected_threshold,

        "random_state":
            RANDOM_STATE,

        "n_estimators":
            N_ESTIMATORS,

        "max_depth":
            MAX_DEPTH,

        "min_samples_leaf":
            MIN_SAMPLES_LEAF,

        "training_period":
            {
                "start":
                    train_df[
                        DATE_COLUMN
                    ].min(),

                "end":
                    train_df[
                        DATE_COLUMN
                    ].max(),
            },

        "validation_period":
            {
                "start":
                    validation_df[
                        DATE_COLUMN
                    ].min(),

                "end":
                    validation_df[
                        DATE_COLUMN
                    ].max(),
            },

        "final_test_period":
            {
                "start":
                    test_df[
                        DATE_COLUMN
                    ].min(),

                "end":
                    test_df[
                        DATE_COLUMN
                    ].max(),
            },
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    print(
        f"Review risk model saved to:\n"
        f"{MODEL_FILE}"
    )

    # ========================================================
    # 27. FINAL SUMMARY
    # ========================================================

    print_section(
        "REVIEW RISK MODEL COMPLETE"
    )

    print(
        f"Predictor features: "
        f"{len(features)}"
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
        f"Final test rows: "
        f"{len(test_df):,}"
    )

    print(
        f"Selected validation threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_auc:.4f}"
    )

    print(
        f"Final test ROC-AUC: "
        f"{test_auc:.4f}"
    )

    print(
        f"Final test recall: "
        f"{test_selected['recall']:.4f}"
    )

    print(
        f"Final test F1: "
        f"{test_selected['f1']:.4f}"
    )

    print(
        "\nThreshold selected using validation data only."
    )

    print(
        "Final test period remained untouched "
        "until evaluation."
    )

    print(
        "Model trained only on historical training data."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()