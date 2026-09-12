"""
Train a Random Forest classifier for seller risk prediction.

This script:
1. Loads the existing train/test datasets.
2. Validates the datasets.
3. Handles missing values.
4. Trains a Random Forest classifier.
5. Evaluates the model on the untouched test set.
6. Calculates classification metrics.
7. Displays the confusion matrix.
8. Calculates feature importance.
9. Saves the trained model.
10. Saves model results and feature importance.

Important:
- The test set is NEVER used during training.
- Missing-value imputation is fitted using training data only.
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

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DATA_DIR / "seller_train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "seller_test.csv"

MODEL_FILE = (
    PROCESSED_DATA_DIR /
    "random_forest_model.joblib"
)

RESULTS_FILE = (
    PROCESSED_DATA_DIR /
    "random_forest_results.csv"
)

FEATURE_IMPORTANCE_FILE = (
    PROCESSED_DATA_DIR /
    "random_forest_feature_importance.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "at_risk"

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 2


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print_section("TRAINING RANDOM FOREST MODEL")

print("Loading training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

print(f"Training rows: {len(train_df):,}")
print(f"Training columns: {len(train_df.columns):,}")

print()
print("Loading testing dataset...")

test_df = pd.read_csv(TEST_FILE)

print(f"Testing rows: {len(test_df):,}")
print(f"Testing columns: {len(test_df.columns):,}")


# ============================================================
# DATA VALIDATION
# ============================================================

print_section("DATA VALIDATION")

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        "is missing from training data."
    )

if TARGET_COLUMN not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        "is missing from testing data."
    )

print("Target column: at_risk -> PASS")


# ------------------------------------------------------------
# Feature consistency
# ------------------------------------------------------------

train_features = [
    column
    for column in train_df.columns
    if column != TARGET_COLUMN
]

test_features = [
    column
    for column in test_df.columns
    if column != TARGET_COLUMN
]

if train_features != test_features:
    raise ValueError(
        "Training and testing feature columns do not match."
    )

print("Training/testing feature consistency -> PASS")


# ------------------------------------------------------------
# Target validation
# ------------------------------------------------------------

invalid_train_targets = ~train_df[TARGET_COLUMN].isin([0, 1])

invalid_test_targets = ~test_df[TARGET_COLUMN].isin([0, 1])

if invalid_train_targets.sum() > 0:
    raise ValueError(
        "Invalid target values found in training data."
    )

if invalid_test_targets.sum() > 0:
    raise ValueError(
        "Invalid target values found in testing data."
    )

print("Target values -> PASS")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

print_section("SEPARATING FEATURES AND TARGET")

X_train = train_df.drop(
    columns=[TARGET_COLUMN]
)

y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(
    columns=[TARGET_COLUMN]
)

y_test = test_df[TARGET_COLUMN]

print(
    f"Training feature matrix: {X_train.shape}"
)

print(
    f"Training target: {y_train.shape}"
)

print(
    f"Testing feature matrix: {X_test.shape}"
)

print(
    f"Testing target: {y_test.shape}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print_section("TRAINING TARGET DISTRIBUTION")

print(y_train.value_counts().sort_index())

print()

print("Percentages:")

print(
    y_train
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print_section("TESTING TARGET DISTRIBUTION")

print(y_test.value_counts().sort_index())

print()

print("Percentages:")

print(
    y_test
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print_section("MISSING VALUE ANALYSIS")

missing_train = X_train.isna().sum()

missing_train = missing_train[
    missing_train > 0
]

if missing_train.empty:
    print("No missing values in training features.")

else:
    print(
        "Missing values in training features:"
    )

    print(missing_train)


print()

missing_test = X_test.isna().sum()

missing_test = missing_test[
    missing_test > 0
]

if missing_test.empty:
    print("No missing values in testing features.")

else:
    print(
        "Missing values in testing features:"
    )

    print(missing_test)


# ============================================================
# BUILD RANDOM FOREST PIPELINE
# ============================================================

print_section(
    "BUILDING RANDOM FOREST PIPELINE"
)

print("Pipeline steps:")

print("1. Median imputation")

print("2. Random Forest classifier")

print()

print("Random Forest configuration:")

print(
    f"Number of trees: {N_ESTIMATORS}"
)

print(
    f"Maximum tree depth: {MAX_DEPTH}"
)

print(
    f"Minimum samples per leaf: "
    f"{MIN_SAMPLES_LEAF}"
)

print(
    f"Random state: {RANDOM_STATE}"
)


pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
    ]
)


print()

print(
    "Random Forest pipeline created successfully."
)


# ============================================================
# TRAIN MODEL
# ============================================================

print_section("TRAINING RANDOM FOREST MODEL")

print(
    "Fitting pipeline on training data..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Random Forest training complete."
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print_section(
    "GENERATING TEST PREDICTIONS"
)

y_pred = pipeline.predict(X_test)

y_probability = (
    pipeline.predict_proba(X_test)[:, 1]
)


print(
    f"Predictions generated: "
    f"{len(y_pred):,}"
)

print()

print("First 10 predictions:")

print(y_pred[:10])

print()

print(
    "First 10 predicted probabilities:"
)

print(
    y_probability[:10].round(4)
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print_section(
    "RANDOM FOREST MODEL EVALUATION"
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print_section(
    "RANDOM FOREST CONFUSION MATRIX"
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion matrix:")

print(cm)

print()

print("Interpretation:")

print(
    f"True Negatives:  {cm[0, 0]}"
)

print(
    f"False Positives: {cm[0, 1]}"
)

print(
    f"False Negatives: {cm[1, 0]}"
)

print(
    f"True Positives:  {cm[1, 1]}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print_section(
    "RANDOM FOREST CLASSIFICATION REPORT"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not At Risk",
            "At Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print_section(
    "RANDOM FOREST FEATURE IMPORTANCE"
)

model = pipeline.named_steps["model"]

importance = model.feature_importances_

feature_importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": importance,
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print()

print(
    "Top 15 most important features:"
)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

print_section(
    "SAVING FEATURE IMPORTANCE"
)

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print(
    "Feature importance saved to:"
)

print(
    FEATURE_IMPORTANCE_FILE
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

print_section(
    "SAVING RANDOM FOREST RESULTS"
)

results = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ],
        "value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
        ],
    }
)

results.to_csv(
    RESULTS_FILE,
    index=False
)

print(
    "Evaluation results saved to:"
)

print(
    RESULTS_FILE
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print()

print("Saving trained Random Forest model...")

joblib.dump(
    pipeline,
    MODEL_FILE
)

print()

print("Model saved to:")

print(MODEL_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section(
    "RANDOM FOREST MODEL COMPLETE"
)

print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Testing rows: {len(X_test):,}"
)

print(
    f"Predictor features: "
    f"{X_train.shape[1]:,}"
)

print()

print(
    "Model: Random Forest"
)

print()

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print()

print(
    "The test set was not used during model fitting."
)

print()

print(
    "======================================================================"
)