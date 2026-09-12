"""
Train Production Candidate Random Forest

Purpose:
    Train a Random Forest using only the audited production
    feature set and compare it against the original benchmark.

This is an ablation-style experiment.

Benchmark:
    18 features

Production candidate:
    10 audited features

The model algorithm and hyperparameters remain the same.

Outputs:
    - production_random_forest_model.joblib
    - production_model_results.csv
    - production_model_feature_importance.csv
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

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = (
    DATA_DIR / "seller_production_train.csv"
)

TEST_FILE = (
    DATA_DIR / "seller_production_test.csv"
)

PRODUCTION_DATASET = (
    DATA_DIR / "seller_production_ml_dataset.csv"
)

FEATURE_LIST_FILE = (
    DATA_DIR / "production_feature_list.csv"
)

MODEL_FILE = (
    DATA_DIR / "production_random_forest_model.joblib"
)

RESULTS_FILE = (
    DATA_DIR / "production_model_results.csv"
)

FEATURE_IMPORTANCE_FILE = (
    DATA_DIR
    / "production_model_feature_importance.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "at_risk"

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 2

TEST_SIZE = 0.20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a consistent section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_production_features():
    """
    Load the audited production feature list.
    """

    if not FEATURE_LIST_FILE.exists():

        raise FileNotFoundError(
            f"Production feature list not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    feature_df = pd.read_csv(
        FEATURE_LIST_FILE
    )

    if "feature" not in feature_df.columns:

        raise ValueError(
            "Production feature list does not contain "
            "a 'feature' column."
        )

    features = (
        feature_df["feature"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not features:

        raise ValueError(
            "Production feature list is empty."
        )

    return features


def evaluate_model(
    y_true,
    y_pred,
    y_probability
):
    """
    Calculate standard classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING PRODUCTION CANDIDATE RANDOM FOREST")
    print("=" * 70)

    # ========================================================
    # 1. LOAD PRODUCTION DATASET
    # ========================================================

    print("\nLoading production ML dataset...")

    if not PRODUCTION_DATASET.exists():

        raise FileNotFoundError(
            f"Production ML dataset not found:\n"
            f"{PRODUCTION_DATASET}"
        )

    df = pd.read_csv(
        PRODUCTION_DATASET
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # ========================================================
    # 2. LOAD FEATURE LIST
    # ========================================================

    print_section(
        "LOADING PRODUCTION FEATURES"
    )

    production_features = (
        load_production_features()
    )

    print(
        f"Production feature count: "
        f"{len(production_features)}"
    )

    print("\nFeatures:")

    for feature in production_features:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 3. VALIDATE TARGET
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' missing."
        )

    print(
        f"Target column: "
        f"{TARGET_COLUMN} -> PASS"
    )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Target contains missing values."
        )

    print(
        "Missing target values -> PASS"
    )

    # ========================================================
    # 4. VALIDATE FEATURES
    # ========================================================

    print_section(
        "FEATURE VALIDATION"
    )

    missing_features = [
        feature
        for feature in production_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing production features:\n"
            + "\n".join(missing_features)
        )

    print(
        "All production features exist -> PASS"
    )

    # ========================================================
    # 5. SEPARATE X AND y
    # ========================================================

    print_section(
        "PREPARING FEATURES AND TARGET"
    )

    X = df[
        production_features
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    print(
        f"Feature matrix: {X.shape}"
    )

    print(
        f"Target vector: {y.shape}"
    )

    # ========================================================
    # 6. TRAIN / TEST SPLIT
    # ========================================================

    print_section(
        "CREATING TRAIN / TEST SPLIT"
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows: {len(X_test):,}"
    )

    print(
        "Stratification: ENABLED"
    )

    # ========================================================
    # 7. CHECK TARGET DISTRIBUTIONS
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION"
    )

    print("\nTraining:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    print("\nTraining percentages:")

    print(
        (
            y_train.value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
            .round(2)
        )
    )

    print("\nTesting:")

    print(
        y_test.value_counts()
        .sort_index()
    )

    print("\nTesting percentages:")

    print(
        (
            y_test.value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
            .round(2)
        )
    )

    # ========================================================
    # 8. BUILD PIPELINE
    # ========================================================

    print_section(
        "BUILDING PRODUCTION MODEL PIPELINE"
    )

    print(
        "Pipeline:"
    )

    print(
        "1. Median imputation"
    )

    print(
        "2. Random Forest"
    )

    print("\nHyperparameters:")

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

    production_pipeline = Pipeline(
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
            )
        ]
    )

    print(
        "\nProduction pipeline created -> PASS"
    )

    # ========================================================
    # 9. TRAIN MODEL
    # ========================================================

    print_section(
        "TRAINING PRODUCTION MODEL"
    )

    print(
        "Fitting production Random Forest..."
    )

    production_pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Production model training complete."
    )

    # ========================================================
    # 10. PREDICT
    # ========================================================

    print_section(
        "GENERATING TEST PREDICTIONS"
    )

    y_pred = production_pipeline.predict(
        X_test
    )

    y_probability = (
        production_pipeline
        .predict_proba(X_test)[:, 1]
    )

    print(
        f"Predictions generated: "
        f"{len(y_pred):,}"
    )

    # ========================================================
    # 11. EVALUATE
    # ========================================================

    print_section(
        "PRODUCTION MODEL EVALUATION"
    )

    metrics = evaluate_model(
        y_test,
        y_pred,
        y_probability
    )

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # 12. CONFUSION MATRIX
    # ========================================================

    print_section(
        "PRODUCTION MODEL CONFUSION MATRIX"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        cm
    )

    tn, fp, fn, tp = cm.ravel()

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
    # 13. CLASSIFICATION REPORT
    # ========================================================

    print_section(
        "PRODUCTION MODEL CLASSIFICATION REPORT"
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

    # ========================================================
    # 14. FEATURE IMPORTANCE
    # ========================================================

    print_section(
        "PRODUCTION MODEL FEATURE IMPORTANCE"
    )

    forest = production_pipeline.named_steps[
        "model"
    ]

    feature_importance = pd.DataFrame(
        {
            "feature": production_features,
            "importance": (
                forest.feature_importances_
            )
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

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # ========================================================
    # 15. SAVE MODEL
    # ========================================================

    print_section(
        "SAVING PRODUCTION MODEL"
    )

    joblib.dump(
        production_pipeline,
        MODEL_FILE
    )

    print(
        f"Production model saved to:\n"
        f"{MODEL_FILE}"
    )

    # ========================================================
    # 16. SAVE RESULTS
    # ========================================================

    print("\nSaving production model metrics...")

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
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["roc_auc"],
            ]
        }
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"Results saved to:\n"
        f"{RESULTS_FILE}"
    )

    # ========================================================
    # 17. SAVE FEATURE IMPORTANCE
    # ========================================================

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print(
        f"Feature importance saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    # ========================================================
    # 18. FINAL SUMMARY
    # ========================================================

    print_section(
        "PRODUCTION MODEL COMPLETE"
    )

    print(
        f"Features used: "
        f"{len(production_features)}"
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "\nThis is the production candidate model."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()