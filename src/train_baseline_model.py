"""
Train a baseline Logistic Regression model for seller risk prediction.

This script:
1. Loads the training and testing datasets.
2. Separates predictors and target.
3. Handles missing values using median imputation.
4. Standardizes numerical features.
5. Trains a Logistic Regression classifier.
6. Evaluates the model on the unseen test set.
7. Reports classification metrics.
8. Saves the trained model and evaluation results.

Important:
- Imputation is fitted only on training data.
- Scaling is fitted only on training data.
- The test set remains completely unseen during training.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DATA_DIR / "seller_train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "seller_test.csv"

MODEL_FILE = PROCESSED_DATA_DIR / "baseline_logistic_regression.joblib"
RESULTS_FILE = PROCESSED_DATA_DIR / "baseline_model_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "at_risk"

RANDOM_STATE = 42


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

print_section("TRAINING BASELINE LOGISTIC REGRESSION MODEL")

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
        f"Target column '{TARGET_COLUMN}' is missing from training data."
    )

if TARGET_COLUMN not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' is missing from testing data."
    )

print("Target column: at_risk -> PASS")


# Check feature consistency

train_features = [
    column for column in train_df.columns
    if column != TARGET_COLUMN
]

test_features = [
    column for column in test_df.columns
    if column != TARGET_COLUMN
]

if train_features != test_features:
    raise ValueError(
        "Training and testing feature columns do not match."
    )

print("Training/testing feature consistency -> PASS")


# Check target values

invalid_train_targets = ~train_df[TARGET_COLUMN].isin([0, 1])
invalid_test_targets = ~test_df[TARGET_COLUMN].isin([0, 1])

if invalid_train_targets.sum() > 0:
    raise ValueError("Invalid target values found in training dataset.")

if invalid_test_targets.sum() > 0:
    raise ValueError("Invalid target values found in testing dataset.")

print("Target values -> PASS")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

print_section("SEPARATING FEATURES AND TARGET")

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

print(f"Training feature matrix: {X_train.shape}")
print(f"Training target: {y_train.shape}")

print(f"Testing feature matrix: {X_test.shape}")
print(f"Testing target: {y_test.shape}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print_section("TRAINING TARGET DISTRIBUTION")

print(y_train.value_counts().sort_index())

print()
print("Percentages:")

print(
    (y_train.value_counts(normalize=True)
     .sort_index()
     .mul(100)
     .round(2))
)


print_section("TESTING TARGET DISTRIBUTION")

print(y_test.value_counts().sort_index())

print()
print("Percentages:")

print(
    (y_test.value_counts(normalize=True)
     .sort_index()
     .mul(100)
     .round(2))
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print_section("MISSING VALUE ANALYSIS")

missing_train = X_train.isna().sum()
missing_train = missing_train[missing_train > 0]

if missing_train.empty:
    print("No missing values in training features.")
else:
    print("Missing values in training features:")
    print(missing_train)

missing_test = X_test.isna().sum()
missing_test = missing_test[missing_test > 0]

print()

if missing_test.empty:
    print("No missing values in testing features.")
else:
    print("Missing values in testing features:")
    print(missing_test)


# ============================================================
# BUILD MACHINE LEARNING PIPELINE
# ============================================================

print_section("BUILDING MACHINE LEARNING PIPELINE")

print("Pipeline steps:")
print("1. Median imputation")
print("2. Standard feature scaling")
print("3. Logistic Regression")

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=2000
            )
        ),
    ]
)

print()
print("Pipeline created successfully.")


# ============================================================
# TRAIN MODEL
# ============================================================

print_section("TRAINING BASELINE MODEL")

print("Fitting pipeline on training data...")

pipeline.fit(X_train, y_train)

print("Model training complete.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print_section("GENERATING TEST PREDICTIONS")

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

print(f"Predictions generated: {len(y_pred):,}")

print()
print("First 10 predictions:")
print(y_pred[:10])

print()
print("First 10 predicted probabilities:")
print(y_probability[:10].round(4))


# ============================================================
# MODEL EVALUATION
# ============================================================

print_section("MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)

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


print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print_section("CONFUSION MATRIX")

cm = confusion_matrix(y_test, y_pred)

print("Confusion matrix:")
print(cm)

print()
print("Interpretation:")

print(f"True Negatives:  {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives:  {cm[1, 1]}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print_section("CLASSIFICATION REPORT")

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
# MODEL INTERPRETATION
# ============================================================

print_section("MODEL COEFFICIENTS")

model = pipeline.named_steps["model"]

coefficients = pd.DataFrame(
    {
        "feature": X_train.columns,
        "coefficient": model.coef_[0],
    }
)

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
)

print()
print("Most influential features:")

print(
    coefficients[
        [
            "feature",
            "coefficient"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

print_section("SAVING MODEL RESULTS")

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

print(f"Evaluation results saved to:")
print(RESULTS_FILE)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print()
print("Saving trained model...")

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

print_section("BASELINE MODEL COMPLETE")

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Predictor features: {X_train.shape[1]:,}")

print()
print("Baseline model: Logistic Regression")

print()
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print()
print("The test set was not used during model fitting.")

print()
print("======================================================================")