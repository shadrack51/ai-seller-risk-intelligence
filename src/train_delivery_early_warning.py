"""
Train Delivery Early-Warning Random Forest

Purpose
-------
Train a specialized Random Forest model that predicts:

    future_delivery_risk

The model uses historical seller behavior only.

Temporal design:

    TRAIN:
        2017-01 through 2018-02

    VALIDATION:
        2018-03 through 2018-05

    TEST:
        2018-06 through 2018-08

The validation set is used for model assessment and future
tuning decisions.

The final test period is kept as the out-of-time benchmark.

Outputs
-------
data/processed/delivery_early_warning_model.joblib
data/processed/delivery_early_warning_validation_results.csv
data/processed/delivery_early_warning_test_results.csv
data/processed/delivery_early_warning_feature_importance.csv
"""


from pathlib import Path

import joblib
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

TRAIN_FILE = (
    DATA_DIR / "target_delivery_train.csv"
)

VALIDATION_FILE = (
    DATA_DIR / "target_delivery_validation.csv"
)

TEST_FILE = (
    DATA_DIR / "target_delivery_test.csv"
)

MODEL_FILE = (
    DATA_DIR
    / "delivery_early_warning_model.joblib"
)

VALIDATION_RESULTS_FILE = (
    DATA_DIR
    / "delivery_early_warning_validation_results.csv"
)

TEST_RESULTS_FILE = (
    DATA_DIR
    / "delivery_early_warning_test_results.csv"
)

FEATURE_IMPORTANCE_FILE = (
    DATA_DIR
    / "delivery_early_warning_feature_importance.csv"
)


# ============================================================
# CORE CONFIGURATION
# ============================================================

TARGET_COLUMN = "future_delivery_risk"

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"


# ============================================================
# DELIVERY MODEL FEATURES
# ============================================================

FEATURES = [
    # --------------------------------------------------------
    # General seller activity
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Seller history
    # --------------------------------------------------------

    "seller_history_orders",
    "seller_history_reviews",

    "recent_order_activity",
    "older_order_activity",

    "low_history_orders",
    "low_history_reviews",

    "sufficient_order_history",
    "sufficient_review_history",

    # --------------------------------------------------------
    # Delivery behavior
    # --------------------------------------------------------

    "recent_delivered_orders",
    "older_delivered_orders",

    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",

    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",

    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",

    "recent_late_orders",
    "older_late_orders",

    "recent_delivery_activity",
    "older_delivery_activity",

    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",
]


# ============================================================
# RANDOM FOREST CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 2


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(
    df,
    dataset_name,
):
    """
    Validate one delivery dataset.
    """

    required_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
        *FEATURES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing columns:\n"
            + "\n".join(missing)
        )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            f"{dataset_name} contains invalid dates."
        )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            f"{dataset_name} contains missing target values."
        )

    invalid_targets = (
        set(
            df[TARGET_COLUMN]
            .unique()
        )
        - {0, 1}
    )

    if invalid_targets:

        raise ValueError(
            f"{dataset_name} contains invalid target values:\n"
            f"{invalid_targets}"
        )

    duplicate_keys = (
        df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_keys > 0:

        raise ValueError(
            f"{dataset_name} contains duplicate "
            "seller/snapshot rows."
        )


def evaluate_model(
    model,
    X,
    y,
):
    """
    Generate predictions and calculate classification metrics.
    """

    predictions = model.predict(
        X
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    metrics = {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "average_precision": average_precision_score(
            y,
            probabilities,
        ),
    }

    return (
        predictions,
        probabilities,
        metrics,
    )


def metrics_to_dataframe(
    metrics,
    split_name,
):
    """
    Convert metrics dictionary into a tidy DataFrame.
    """

    rows = []

    for metric_name, value in metrics.items():

        rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "value": value,
            }
        )

    return pd.DataFrame(rows)


def print_metrics(
    metrics,
):
    """Print model metrics."""

    print(
        f"Accuracy:           "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{metrics['average_precision']:.4f}"
    )


def print_target_distribution(
    df,
    label,
):
    """
    Print counts and percentages for the target.
    """

    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[TARGET_COLUMN]
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
    print("TRAINING DELIVERY EARLY-WARNING RANDOM FOREST")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATASETS
    # ========================================================

    print_section(
        "LOADING DELIVERY DATASETS"
    )

    for file_path, label in [
        (
            TRAIN_FILE,
            "Training",
        ),
        (
            VALIDATION_FILE,
            "Validation",
        ),
        (
            TEST_FILE,
            "Testing",
        ),
    ]:

        if not file_path.exists():

            raise FileNotFoundError(
                f"{label} dataset not found:\n"
                f"{file_path}\n\n"
                "Run build_risk_specific_temporal_split.py first."
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
    # 2. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING TEMPORAL DATES"
    )

    for df in [
        train_df,
        validation_df,
        test_df,
    ]:

        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce",
        )

    print(
        "Snapshot dates converted -> PASS"
    )

    # ========================================================
    # 3. DATA VALIDATION
    # ========================================================

    print_section(
        "DATA VALIDATION"
    )

    validate_dataset(
        train_df,
        "Training",
    )

    validate_dataset(
        validation_df,
        "Validation",
    )

    validate_dataset(
        test_df,
        "Testing",
    )

    print(
        "Training dataset -> PASS"
    )

    print(
        "Validation dataset -> PASS"
    )

    print(
        "Testing dataset -> PASS"
    )

    # ========================================================
    # 4. CHRONOLOGICAL VALIDATION
    # ========================================================

    print_section(
        "CHRONOLOGICAL VALIDATION"
    )

    train_last = (
        train_df[DATE_COLUMN].max()
    )

    validation_first = (
        validation_df[DATE_COLUMN].min()
    )

    validation_last = (
        validation_df[DATE_COLUMN].max()
    )

    test_first = (
        test_df[DATE_COLUMN].min()
    )

    print(
        f"Training ends: "
        f"{train_last.date()}"
    )

    print(
        f"Validation starts: "
        f"{validation_first.date()}"
    )

    print(
        f"Validation ends: "
        f"{validation_last.date()}"
    )

    print(
        f"Testing starts: "
        f"{test_first.date()}"
    )

    if not (
        train_last
        < validation_first
    ):

        raise ValueError(
            "Training and validation periods overlap."
        )

    if not (
        validation_last
        < test_first
    ):

        raise ValueError(
            "Validation and test periods overlap."
        )

    print(
        "\nChronological ordering -> PASS"
    )

    # ========================================================
    # 5. FEATURE / TARGET SEPARATION
    # ========================================================

    print_section(
        "PREPARING MODEL MATRICES"
    )

    X_train = train_df[
        FEATURES
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].copy()

    X_validation = validation_df[
        FEATURES
    ].copy()

    y_validation = validation_df[
        TARGET_COLUMN
    ].copy()

    X_test = test_df[
        FEATURES
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    print(
        f"Training features: "
        f"{X_train.shape}"
    )

    print(
        f"Validation features: "
        f"{X_validation.shape}"
    )

    print(
        f"Testing features: "
        f"{X_test.shape}"
    )

    print(
        f"\nPredictor count: "
        f"{len(FEATURES)}"
    )

    # ========================================================
    # 6. FEATURE LIST
    # ========================================================

    print_section(
        "DELIVERY MODEL FEATURES"
    )

    for number, feature in enumerate(
        FEATURES,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 7. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION"
    )

    print_target_distribution(
        train_df,
        "Training",
    )

    print_target_distribution(
        validation_df,
        "Validation",
    )

    print_target_distribution(
        test_df,
        "Testing",
    )

    # ========================================================
    # 8. MISSING VALUES
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    for label, X in [
        (
            "Training",
            X_train,
        ),
        (
            "Validation",
            X_validation,
        ),
        (
            "Testing",
            X_test,
        ),
    ]:

        missing = (
            X.isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

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

    # ========================================================
    # 9. BUILD PIPELINE
    # ========================================================

    print_section(
        "BUILDING DELIVERY RANDOM FOREST PIPELINE"
    )

    print(
        "Pipeline steps:"
    )

    print(
        "1. Median imputation"
    )

    print(
        "2. Random Forest classifier"
    )

    print(
        "\nRandom Forest hyperparameters:"
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

    model = Pipeline(
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

    print(
        "\nPipeline created -> PASS"
    )

    # ========================================================
    # 10. TRAIN
    # ========================================================

    print_section(
        "TRAINING DELIVERY MODEL"
    )

    print(
        "Fitting on training data ONLY..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ========================================================
    # 11. TRAIN PERFORMANCE
    # ========================================================

    print_section(
        "TRAINING PERFORMANCE"
    )

    (
        train_predictions,
        train_probabilities,
        train_metrics,
    ) = evaluate_model(
        model,
        X_train,
        y_train,
    )

    print_metrics(
        train_metrics
    )

    # ========================================================
    # 12. VALIDATION PERFORMANCE
    # ========================================================

    print_section(
        "VALIDATION PERFORMANCE"
    )

    (
        validation_predictions,
        validation_probabilities,
        validation_metrics,
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print_metrics(
        validation_metrics
    )

    # ========================================================
    # 13. TEST PERFORMANCE
    # ========================================================

    print_section(
        "FINAL OUT-OF-TIME TEST PERFORMANCE"
    )

    print(
        "The final test period remains untouched "
        "for model fitting."
    )

    (
        test_predictions,
        test_probabilities,
        test_metrics,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_metrics(
        test_metrics
    )

    # ========================================================
    # 14. GENERALIZATION ANALYSIS
    # ========================================================

    print_section(
        "TEMPORAL GENERALIZATION ANALYSIS"
    )

    validation_auc_drop = (
        train_metrics["roc_auc"]
        - validation_metrics["roc_auc"]
    )

    test_auc_drop = (
        train_metrics["roc_auc"]
        - test_metrics["roc_auc"]
    )

    validation_f1_drop = (
        train_metrics["f1_score"]
        - validation_metrics["f1_score"]
    )

    test_f1_drop = (
        train_metrics["f1_score"]
        - test_metrics["f1_score"]
    )

    print(
        f"Train → Validation ROC-AUC drop: "
        f"{validation_auc_drop:.4f}"
    )

    print(
        f"Train → Test ROC-AUC drop: "
        f"{test_auc_drop:.4f}"
    )

    print(
        f"Train → Validation F1 drop: "
        f"{validation_f1_drop:.4f}"
    )

    print(
        f"Train → Test F1 drop: "
        f"{test_f1_drop:.4f}"
    )

    if test_auc_drop > 0.10:

        print(
            "\nWARNING:"
        )

        print(
            "The test ROC-AUC is more than "
            "0.10 below training ROC-AUC."
        )

        print(
            "This indicates weak temporal generalization "
            "and should be investigated."
        )

    else:

        print(
            "\nTemporal ROC-AUC degradation "
            "is below 0.10."
        )

    # ========================================================
    # 15. VALIDATION CONFUSION MATRIX
    # ========================================================

    print_section(
        "VALIDATION CONFUSION MATRIX"
    )

    validation_cm = confusion_matrix(
        y_validation,
        validation_predictions,
    )

    print(
        validation_cm
    )

    val_tn, val_fp, val_fn, val_tp = (
        validation_cm.ravel()
    )

    print()

    print(
        f"True Negatives:  {val_tn}"
    )

    print(
        f"False Positives: {val_fp}"
    )

    print(
        f"False Negatives: {val_fn}"
    )

    print(
        f"True Positives:  {val_tp}"
    )

    # ========================================================
    # 16. FINAL TEST CONFUSION MATRIX
    # ========================================================

    print_section(
        "FINAL TEST CONFUSION MATRIX"
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print(
        test_cm
    )

    test_tn, test_fp, test_fn, test_tp = (
        test_cm.ravel()
    )

    print()

    print(
        f"True Negatives:  {test_tn}"
    )

    print(
        f"False Positives: {test_fp}"
    )

    print(
        f"False Negatives: {test_fn}"
    )

    print(
        f"True Positives:  {test_tp}"
    )

    # ========================================================
    # 17. TEST CLASSIFICATION REPORT
    # ========================================================

    print_section(
        "FINAL TEST CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "No Future Delivery Risk",
                "Future Delivery Risk",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # 18. FEATURE IMPORTANCE
    # ========================================================

    print_section(
        "DELIVERY MODEL FEATURE IMPORTANCE"
    )

    forest = model.named_steps[
        "model"
    ]

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                forest.feature_importances_
            ),
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 19. PROBABILITY SUMMARY
    # ========================================================

    print_section(
        "TEST RISK PROBABILITY SUMMARY"
    )

    probability_summary = (
        pd.Series(
            test_probabilities,
            name="delivery_risk_probability",
        )
        .describe()
    )

    print(
        probability_summary.round(4)
    )

    # ========================================================
    # 20. SAVE MODEL
    # ========================================================

    print_section(
        "SAVING DELIVERY EARLY-WARNING MODEL"
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_FILE}"
    )

    # ========================================================
    # 21. SAVE VALIDATION RESULTS
    # ========================================================

    validation_results = (
        metrics_to_dataframe(
            validation_metrics,
            "validation",
        )
    )

    validation_results.to_csv(
        VALIDATION_RESULTS_FILE,
        index=False,
    )

    print(
        f"\nValidation results saved to:\n"
        f"{VALIDATION_RESULTS_FILE}"
    )

    # ========================================================
    # 22. SAVE TEST RESULTS
    # ========================================================

    test_results = (
        metrics_to_dataframe(
            test_metrics,
            "test",
        )
    )

    test_results.to_csv(
        TEST_RESULTS_FILE,
        index=False,
    )

    print(
        f"\nTest results saved to:\n"
        f"{TEST_RESULTS_FILE}"
    )

    # ========================================================
    # 23. SAVE FEATURE IMPORTANCE
    # ========================================================

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    # ========================================================
    # 24. FINAL SUMMARY
    # ========================================================

    print_section(
        "DELIVERY EARLY-WARNING MODEL COMPLETE"
    )

    print(
        f"Features: "
        f"{len(FEATURES)}"
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

    print(
        f"\nValidation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"\nValidation Recall: "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"\nValidation Average Precision: "
        f"{validation_metrics['average_precision']:.4f}"
    )

    print(
        f"Test Average Precision: "
        f"{test_metrics['average_precision']:.4f}"
    )

    print(
        "\nThe model was trained only on historical "
        "training observations."
    )

    print(
        "Validation data was used for assessment."
    )

    print(
        "The final test period was used only for "
        "out-of-time evaluation."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()