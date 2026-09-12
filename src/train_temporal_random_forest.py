"""
Train Temporal Random Forest

Purpose
-------
Train a forward-looking seller risk model using historical
seller behavior to predict risk in a future 60-day window.

Temporal split:

    TRAIN       -> historical periods
    VALIDATION  -> later periods used for model assessment
    TEST        -> final future-period evaluation

Important:
    The test set is NOT used to fit the model or make
    model-selection decisions.

Outputs
-------
data/processed/temporal_random_forest_model.joblib
data/processed/temporal_random_forest_validation_results.csv
data/processed/temporal_random_forest_test_results.csv
data/processed/temporal_random_forest_feature_importance.csv
"""


from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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
    DATA_DIR
    / "seller_temporal_train.csv"
)

VALIDATION_FILE = (
    DATA_DIR
    / "seller_temporal_validation.csv"
)

TEST_FILE = (
    DATA_DIR
    / "seller_temporal_test.csv"
)

MODEL_FILE = (
    DATA_DIR
    / "temporal_random_forest_model.joblib"
)

VALIDATION_RESULTS_FILE = (
    DATA_DIR
    / "temporal_random_forest_validation_results.csv"
)

TEST_RESULTS_FILE = (
    DATA_DIR
    / "temporal_random_forest_test_results.csv"
)

FEATURE_IMPORTANCE_FILE = (
    DATA_DIR
    / "temporal_random_forest_feature_importance.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "future_at_risk"

IDENTIFIER_COLUMNS = [
    "seller_id",
    "snapshot_date",
    "observation_start",
    "prediction_start",
    "prediction_end",
]


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


def validate_split(
    df,
    split_name,
):
    """
    Validate one temporal dataset.
    """

    print(
        f"\nValidating {split_name} dataset..."
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"{split_name} dataset is missing "
            f"target column '{TARGET_COLUMN}'."
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"{split_name} dataset is missing features:\n"
            + "\n".join(missing_features)
        )

    missing_target = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    if missing_target > 0:

        raise ValueError(
            f"{split_name} dataset contains "
            f"{missing_target} missing target values."
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
            f"{split_name} dataset contains "
            f"invalid target values:\n"
            f"{invalid_targets}"
        )

    print(
        f"{split_name} dataset validation -> PASS"
    )


def evaluate_predictions(
    y_true,
    y_pred,
    y_probability,
):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


def build_results_dataframe(
    metrics,
    split_name,
):
    """
    Convert evaluation metrics into a DataFrame.
    """

    return pd.DataFrame(
        {
            "split": [
                split_name
            ] * 5,
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ],
            "value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["roc_auc"],
            ],
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING TEMPORAL RANDOM FOREST")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATASETS
    # ========================================================

    print_section(
        "LOADING TEMPORAL DATASETS"
    )

    print("Loading training data...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print("Loading validation data...")

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print("Loading testing data...")

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Testing rows: "
        f"{len(test_df):,}"
    )

    # ========================================================
    # 2. VALIDATE DATASETS
    # ========================================================

    print_section(
        "DATA VALIDATION"
    )

    validate_split(
        train_df,
        "Training",
    )

    validate_split(
        validation_df,
        "Validation",
    )

    validate_split(
        test_df,
        "Testing",
    )

    # ========================================================
    # 3. VERIFY CHRONOLOGICAL ORDER
    # ========================================================

    print_section(
        "CHRONOLOGICAL VALIDATION"
    )

    train_last = pd.to_datetime(
        train_df["snapshot_date"]
    ).max()

    validation_first = pd.to_datetime(
        validation_df["snapshot_date"]
    ).min()

    validation_last = pd.to_datetime(
        validation_df["snapshot_date"]
    ).max()

    test_first = pd.to_datetime(
        test_df["snapshot_date"]
    ).min()

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

    if not train_last < validation_first:

        raise ValueError(
            "Training data occurs after or during "
            "validation data."
        )

    if not validation_last < test_first:

        raise ValueError(
            "Validation data occurs after or during "
            "testing data."
        )

    print(
        "\nChronological ordering -> PASS"
    )

    # ========================================================
    # 4. TARGET DISTRIBUTIONS
    # ========================================================

    print_section(
        "TARGET DISTRIBUTIONS"
    )

    for name, dataset in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df),
    ]:

        print(
            f"\n{name}:"
        )

        counts = (
            dataset[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        percentages = (
            dataset[TARGET_COLUMN]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
            .round(2)
        )

        print(
            "Counts:"
        )

        print(
            counts
        )

        print(
            "Percentages:"
        )

        print(
            percentages
        )

    # ========================================================
    # 5. CREATE FEATURE MATRICES
    # ========================================================

    print_section(
        "PREPARING FEATURE MATRICES"
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
    # 6. MISSING VALUE ANALYSIS
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    train_missing = (
        X_train
        .isna()
        .sum()
    )

    train_missing = train_missing[
        train_missing > 0
    ]

    if train_missing.empty:

        print(
            "No missing values in training features."
        )

    else:

        print(
            "Training missing values:"
        )

        print(
            train_missing
        )

    validation_missing = (
        X_validation
        .isna()
        .sum()
    )

    validation_missing = (
        validation_missing[
            validation_missing > 0
        ]
    )

    if validation_missing.empty:

        print(
            "\nNo missing values in validation features."
        )

    else:

        print(
            "\nValidation missing values:"
        )

        print(
            validation_missing
        )

    test_missing = (
        X_test
        .isna()
        .sum()
    )

    test_missing = (
        test_missing[
            test_missing > 0
        ]
    )

    if test_missing.empty:

        print(
            "\nNo missing values in testing features."
        )

    else:

        print(
            "\nTesting missing values:"
        )

        print(
            test_missing
        )

    # ========================================================
    # 7. BUILD PIPELINE
    # ========================================================

    print_section(
        "BUILDING TEMPORAL RANDOM FOREST PIPELINE"
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

    model_pipeline = Pipeline(
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
        "\nTemporal Random Forest pipeline -> PASS"
    )

    # ========================================================
    # 8. TRAIN
    # ========================================================

    print_section(
        "TRAINING TEMPORAL RANDOM FOREST"
    )

    print(
        "Fitting model on TRAINING data only..."
    )

    model_pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Model training complete."
    )

    # ========================================================
    # 9. TRAINING PERFORMANCE
    # ========================================================

    print_section(
        "TRAINING PERFORMANCE"
    )

    train_predictions = (
        model_pipeline.predict(
            X_train
        )
    )

    train_probabilities = (
        model_pipeline
        .predict_proba(
            X_train
        )[:, 1]
    )

    train_metrics = (
        evaluate_predictions(
            y_train,
            train_predictions,
            train_probabilities,
        )
    )

    print(
        f"Accuracy:  "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{train_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{train_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{train_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{train_metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # 10. VALIDATION PERFORMANCE
    # ========================================================

    print_section(
        "VALIDATION PERFORMANCE"
    )

    validation_predictions = (
        model_pipeline.predict(
            X_validation
        )
    )

    validation_probabilities = (
        model_pipeline
        .predict_proba(
            X_validation
        )[:, 1]
    )

    validation_metrics = (
        evaluate_predictions(
            y_validation,
            validation_predictions,
            validation_probabilities,
        )
    )

    print(
        f"Accuracy:  "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{validation_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # 11. TEST PERFORMANCE
    # ========================================================

    print_section(
        "FINAL TEST PERFORMANCE"
    )

    print(
        "Evaluating on the later, untouched test period..."
    )

    test_predictions = (
        model_pipeline.predict(
            X_test
        )
    )

    test_probabilities = (
        model_pipeline
        .predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = (
        evaluate_predictions(
            y_test,
            test_predictions,
            test_probabilities,
        )
    )

    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{test_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{test_metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # 12. GENERALIZATION GAP
    # ========================================================

    print_section(
        "TEMPORAL GENERALIZATION ANALYSIS"
    )

    validation_roc_gap = (
        train_metrics["roc_auc"]
        - validation_metrics["roc_auc"]
    )

    test_roc_gap = (
        train_metrics["roc_auc"]
        - test_metrics["roc_auc"]
    )

    validation_f1_gap = (
        train_metrics["f1_score"]
        - validation_metrics["f1_score"]
    )

    test_f1_gap = (
        train_metrics["f1_score"]
        - test_metrics["f1_score"]
    )

    print(
        f"Train → Validation ROC-AUC gap: "
        f"{validation_roc_gap:.4f}"
    )

    print(
        f"Train → Test ROC-AUC gap: "
        f"{test_roc_gap:.4f}"
    )

    print(
        f"Train → Validation F1 gap: "
        f"{validation_f1_gap:.4f}"
    )

    print(
        f"Train → Test F1 gap: "
        f"{test_f1_gap:.4f}"
    )

    if test_roc_gap > 0.10:

        print(
            "\nWARNING:"
        )

        print(
            "The test ROC-AUC is substantially lower than "
            "training ROC-AUC."
        )

        print(
            "This may indicate overfitting, temporal drift, "
            "or changing seller behavior."
        )

    else:

        print(
            "\nTemporal ROC-AUC generalization gap "
            "is below 0.10."
        )

    # ========================================================
    # 13. TEST CONFUSION MATRIX
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

    tn, fp, fn, tp = (
        test_cm.ravel()
    )

    print()

    print(
        f"True Negatives:  {tn}"
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
    # 14. TEST CLASSIFICATION REPORT
    # ========================================================

    print_section(
        "FINAL TEST CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Not Future At Risk",
                "Future At Risk",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # 15. FEATURE IMPORTANCE
    # ========================================================

    print_section(
        "TEMPORAL MODEL FEATURE IMPORTANCE"
    )

    forest = model_pipeline.named_steps[
        "model"
    ]

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                forest.feature_importances_
            ),
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # ========================================================
    # 16. SAVE MODEL
    # ========================================================

    print_section(
        "SAVING TEMPORAL MODEL"
    )

    joblib.dump(
        model_pipeline,
        MODEL_FILE,
    )

    print(
        f"Temporal Random Forest saved to:\n"
        f"{MODEL_FILE}"
    )

    # ========================================================
    # 17. SAVE VALIDATION RESULTS
    # ========================================================

    validation_results = build_results_dataframe(
        validation_metrics,
        "validation",
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
    # 18. SAVE TEST RESULTS
    # ========================================================

    test_results = build_results_dataframe(
        test_metrics,
        "test",
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
    # 19. SAVE FEATURE IMPORTANCE
    # ========================================================

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    # ========================================================
    # 20. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL RANDOM FOREST COMPLETE"
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

    print(
        f"\nValidation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"\nValidation F1: "
        f"{validation_metrics['f1_score']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1_score']:.4f}"
    )

    print(
        "\nImportant:"
    )

    print(
        "The model was trained only on the historical "
        "training period."
    )

    print(
        "Validation data was used for model assessment."
    )

    print(
        "The final test period was not used for fitting."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()