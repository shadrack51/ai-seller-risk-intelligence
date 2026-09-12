"""
Business Threshold Optimization

Purpose:
    Analyze how different probability thresholds affect:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - False positives
    - False negatives

The Random Forest model is NOT retrained.

Important:
    This script uses the existing trained model and evaluates
    different classification thresholds.

Project:
    AI & Data Science Seller Risk Prediction System
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = (
    PROCESSED_DATA_DIR /
    "random_forest_model.joblib"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR /
    "seller_test.csv"
)

OUTPUT_PATH = (
    PROCESSED_DATA_DIR /
    "threshold_optimization.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "at_risk"

RANDOM_STATE = 42

# Thresholds that will be evaluated.
THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a consistent section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(df):
    """Validate the test dataset."""

    print_section("DATA VALIDATION")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    print(
        f"Target column: {TARGET_COLUMN} -> PASS"
    )

    missing_target = df[TARGET_COLUMN].isna().sum()

    if missing_target > 0:
        raise ValueError(
            f"Target contains {missing_target} missing values."
        )

    print(
        f"Missing target values: {missing_target} -> PASS"
    )

    invalid_targets = set(
        df[TARGET_COLUMN].dropna().unique()
    ) - {0, 1}

    if invalid_targets:
        raise ValueError(
            f"Invalid target values found: {invalid_targets}"
        )

    print("Target values: PASS")


def get_model_feature_names(model):
    """
    Extract feature names from the trained pipeline.

    We do not assume that the pipeline contains a step
    called 'preprocessor'.

    Instead, we inspect the pipeline and find the
    transformation step that exposes feature_names_in_.
    """

    print_section("MODEL FEATURE VALIDATION")

    print("Pipeline steps:")

    for name, step in model.named_steps.items():
        print(f" - {name}: {type(step).__name__}")

    print()

    # The Random Forest pipeline contains the training
    # transformation steps. The first pipeline step that
    # knows the input feature names is used.
    model_features = None

    for step_name, step in model.named_steps.items():

        if hasattr(step, "feature_names_in_"):

            model_features = list(
                step.feature_names_in_
            )

            print(
                f"Feature names obtained from pipeline step: "
                f"{step_name}"
            )

            break

    # Some sklearn versions/models expose feature names
    # through the final estimator instead.
    if model_features is None:

        final_estimator = model.steps[-1][1]

        if hasattr(
            final_estimator,
            "feature_names_in_"
        ):

            model_features = list(
                final_estimator.feature_names_in_
            )

            print(
                "Feature names obtained from final estimator."
            )

    if model_features is None:

        print(
            "Model does not expose feature names directly."
        )

        return None

    print(
        f"Number of model features: "
        f"{len(model_features)}"
    )

    return model_features


def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):
    """
    Convert predicted probabilities into class predictions
    using a specified classification threshold.

    Example:

        threshold = 0.50

        probability >= 0.50
            -> predicted class 1

        probability < 0.50
            -> predicted class 0
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("BUSINESS THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading trained Random Forest model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Random Forest model loaded successfully."
    )

    print(
        f"Model path: {MODEL_PATH}"
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print()
    print("Loading untouched test dataset...")

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_DATA_PATH}"
        )

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

    print(
        f"Testing columns: {len(test_df.columns):,}"
    )

    # ========================================================
    # VALIDATE DATA
    # ========================================================

    validate_dataset(
        test_df
    )

    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    print_section(
        "SEPARATING FEATURES AND TARGET"
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[
        TARGET_COLUMN
    ]

    print(
        f"Test feature matrix: {X_test.shape}"
    )

    print(
        f"Test target: {y_test.shape}"
    )

    print()
    print("Target distribution:")

    print(
        y_test.value_counts()
        .sort_index()
    )

    print()
    print("Target percentages:")

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
    # MODEL FEATURE VALIDATION
    # ========================================================

    model_features = get_model_feature_names(
        model
    )

    if model_features is not None:

        test_features = list(
            X_test.columns
        )

        if model_features != test_features:

            print()
            print(
                "WARNING: Model/test feature ordering "
                "does not match exactly."
            )

            if set(model_features) != set(
                test_features
            ):

                raise ValueError(
                    "Model features and test features "
                    "do not contain the same columns."
                )

            # Reorder the test dataframe to match
            # the order used during model training.

            X_test = X_test[
                model_features
            ]

            print(
                "Test features reordered to match "
                "the trained model."
            )

        else:

            print(
                "Model/test feature consistency -> PASS"
            )

    # ========================================================
    # GENERATE PROBABILITIES
    # ========================================================

    print_section(
        "GENERATING RISK PROBABILITIES"
    )

    print(
        "Generating probability predictions..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        f"Probability predictions generated: "
        f"{len(probabilities):,}"
    )

    print()
    print("First 10 probabilities:")

    print(
        np.round(
            probabilities[:10],
            4
        )
    )

    # ========================================================
    # DEFAULT THRESHOLD
    # ========================================================

    print_section(
        "DEFAULT THRESHOLD ANALYSIS"
    )

    default_threshold = 0.50

    default_result = evaluate_threshold(
        y_test,
        probabilities,
        default_threshold
    )

    print(
        f"Default threshold: "
        f"{default_threshold:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{default_result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{default_result['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{default_result['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{default_result['f1_score']:.4f}"
    )

    print(
        f"False positives: "
        f"{default_result['false_positives']}"
    )

    print(
        f"False negatives: "
        f"{default_result['false_negatives']}"
    )

    # ========================================================
    # TEST MULTIPLE THRESHOLDS
    # ========================================================

    print_section(
        "TESTING BUSINESS THRESHOLDS"
    )

    print(
        "Testing probability thresholds..."
    )

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            y_test,
            probabilities,
            threshold
        )

        results.append(
            result
        )

    threshold_df = pd.DataFrame(
        results
    )

    print()

    print(
        threshold_df.round(4).to_string(
            index=False
        )
    )

    # ========================================================
    # BEST F1 THRESHOLD
    # ========================================================

    print_section(
        "BEST F1 THRESHOLD"
    )

    best_f1_row = (
        threshold_df
        .loc[
            threshold_df[
                "f1_score"
            ].idxmax()
        ]
    )

    print(
        f"Best threshold by F1 score: "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"F1 Score: "
        f"{best_f1_row['f1_score']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_f1_row['recall']:.4f}"
    )

    # ========================================================
    # HIGH-RECALL BUSINESS THRESHOLD
    # ========================================================

    print_section(
        "HIGH-RECALL BUSINESS THRESHOLD"
    )

    """
    In this project, missing an at-risk seller can be
    more costly than incorrectly flagging a healthy seller.

    Therefore we examine a threshold that achieves at
    least 90% recall, if one exists.
    """

    high_recall_candidates = threshold_df[
        threshold_df["recall"] >= 0.90
    ].copy()

    if not high_recall_candidates.empty:

        # Among thresholds reaching 90% recall,
        # choose the one with the highest precision.

        best_high_recall = (
            high_recall_candidates
            .sort_values(
                [
                    "precision",
                    "f1_score"
                ],
                ascending=False
            )
            .iloc[0]
        )

        print(
            "A threshold achieving at least "
            "90% recall was found."
        )

        print(
            f"Selected threshold: "
            f"{best_high_recall['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_high_recall['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_high_recall['recall']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{best_high_recall['f1_score']:.4f}"
        )

        print(
            f"False positives: "
            f"{int(best_high_recall['false_positives'])}"
        )

        print(
            f"False negatives: "
            f"{int(best_high_recall['false_negatives'])}"
        )

    else:

        best_high_recall = None

        print(
            "No tested threshold achieved "
            "90% recall."
        )

    # ========================================================
    # BUSINESS COST ANALYSIS
    # ========================================================

    print_section(
        "BUSINESS COST ANALYSIS"
    )

    print(
        "We now assign different costs to false "
        "negatives and false positives."
    )

    print()
    print(
        "False negative = at-risk seller missed."
    )

    print(
        "False positive = healthy seller incorrectly flagged."
    )

    # Example business-cost assumptions.
    #
    # These are illustrative values, not measured
    # financial losses from the Olist dataset.

    FALSE_NEGATIVE_COST = 10

    FALSE_POSITIVE_COST = 1

    print()
    print(
        f"Illustrative false-negative cost: "
        f"{FALSE_NEGATIVE_COST}"
    )

    print(
        f"Illustrative false-positive cost: "
        f"{FALSE_POSITIVE_COST}"
    )

    print()
    print(
        "These costs are placeholders for scenario analysis."
    )

    threshold_df[
        "business_cost"
    ] = (
        threshold_df["false_negatives"]
        * FALSE_NEGATIVE_COST
        +
        threshold_df["false_positives"]
        * FALSE_POSITIVE_COST
    )

    best_cost_row = (
        threshold_df
        .loc[
            threshold_df[
                "business_cost"
            ].idxmin()
        ]
    )

    print()

    print(
        "Lowest-cost threshold under these assumptions:"
    )

    print(
        f"Threshold: "
        f"{best_cost_row['threshold']:.2f}"
    )

    print(
        f"Business cost: "
        f"{best_cost_row['business_cost']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_cost_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_cost_row['recall']:.4f}"
    )

    print(
        f"False positives: "
        f"{int(best_cost_row['false_positives'])}"
    )

    print(
        f"False negatives: "
        f"{int(best_cost_row['false_negatives'])}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print_section(
        "SAVING THRESHOLD ANALYSIS"
    )

    threshold_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Threshold optimization results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print_section(
        "BUSINESS INTERPRETATION"
    )

    print(
        "The Random Forest produces probabilities rather "
        "than simply saying 'at risk' or 'not at risk'."
    )

    print()

    print(
        "The classification threshold determines when "
        "a probability becomes an at-risk prediction."
    )

    print()

    print(
        "Lowering the threshold generally increases recall "
        "but also increases false positives."
    )

    print()

    print(
        "Raising the threshold generally increases precision "
        "but can increase false negatives."
    )

    print()

    print(
        "Therefore, the best threshold depends on the "
        "business cost of missed at-risk sellers versus "
        "unnecessary interventions."
    )

    print()

    print(
        "IMPORTANT: The cost values used in this script are "
        "illustrative. They should eventually be replaced "
        "with real business costs."
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Thresholds evaluated: "
        f"{len(threshold_df)}"
    )

    print(
        f"Default threshold: "
        f"{default_threshold:.2f}"
    )

    print(
        f"Best F1 threshold: "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"Lowest-cost threshold under illustrative "
        f"cost assumptions: "
        f"{best_cost_row['threshold']:.2f}"
    )

    print()

    print(
        "The Random Forest model was not retrained."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()