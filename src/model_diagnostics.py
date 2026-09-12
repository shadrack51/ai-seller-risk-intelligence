"""
Seller Risk Prediction - Model Diagnostics

This script evaluates the already-trained Random Forest model
using the untouched test dataset.

Diagnostics include:
1. Confusion matrix
2. ROC curve
3. Precision-Recall curve
4. ROC-AUC
5. Average Precision
6. Prediction probability distribution
7. Threshold analysis
8. False positive / false negative analysis
9. Diagnostic summary

The model is NOT retrained in this script.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = DATA_DIR / "random_forest_model.joblib"
TEST_DATA_PATH = DATA_DIR / "seller_test.csv"

OUTPUT_RESULTS_PATH = DATA_DIR / "model_diagnostics.csv"
THRESHOLD_RESULTS_PATH = DATA_DIR / "threshold_analysis.csv"

ROC_CURVE_PATH = DATA_DIR / "random_forest_roc_curve.png"
PR_CURVE_PATH = DATA_DIR / "random_forest_precision_recall_curve.png"
CONFUSION_MATRIX_PATH = DATA_DIR / "random_forest_confusion_matrix.png"
PROBABILITY_PATH = DATA_DIR / "random_forest_probability_distribution.png"


TARGET_COLUMN = "at_risk"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("RANDOM FOREST MODEL DIAGNOSTICS")

    # --------------------------------------------------------
    # 1. LOAD MODEL
    # --------------------------------------------------------

    print("Loading trained Random Forest model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Random Forest model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")

    # --------------------------------------------------------
    # 2. LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading untouched test dataset...")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Testing rows: {len(test_df):,}")
    print(f"Testing columns: {len(test_df.columns)}")

    # --------------------------------------------------------
    # 3. VALIDATE TEST DATA
    # --------------------------------------------------------

    print_section("DATA VALIDATION")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    print("Target column: at_risk -> PASS")

    missing_target = test_df[TARGET_COLUMN].isna().sum()

    if missing_target > 0:
        raise ValueError(
            f"Missing target values: {missing_target}"
        )

    print("Missing target values: 0 -> PASS")

    # --------------------------------------------------------
    # 4. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    print_section("SEPARATING FEATURES AND TARGET")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print(f"Test feature matrix: {X_test.shape}")
    print(f"Test target: {y_test.shape}")

    print("\nTarget distribution:")
    print(y_test.value_counts())

    print("\nTarget percentages:")
    print(
        (y_test.value_counts(normalize=True) * 100).round(2)
    )

    # --------------------------------------------------------
    # 5. CHECK FEATURE CONSISTENCY
    # --------------------------------------------------------

    print_section("MODEL FEATURE VALIDATION")

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)
        test_features = list(X_test.columns)

        if model_features != test_features:

            missing_features = [
                feature
                for feature in model_features
                if feature not in test_features
            ]

            extra_features = [
                feature
                for feature in test_features
                if feature not in model_features
            ]

            print("Model/test feature consistency -> FAIL")

            print("\nMissing features:")
            print(missing_features)

            print("\nExtra features:")
            print(extra_features)

            raise ValueError(
                "Test features do not match the features used by the model."
            )

        print("Model/test feature consistency -> PASS")

    else:
        print(
            "Model does not expose feature_names_in_. "
            "Feature-name validation skipped."
        )

    # --------------------------------------------------------
    # 6. GENERATE PREDICTIONS
    # --------------------------------------------------------

    print_section("GENERATING MODEL PREDICTIONS")

    print("Generating class predictions...")

    y_pred = model.predict(X_test)

    print(f"Predictions generated: {len(y_pred):,}")

    print("\nFirst 10 predictions:")
    print(y_pred[:10])

    # --------------------------------------------------------
    # 7. GENERATE PROBABILITIES
    # --------------------------------------------------------

    print("\nGenerating probability predictions...")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The loaded model does not support probability predictions."
        )

    y_probability = model.predict_proba(X_test)[:, 1]

    print(f"Probability predictions generated: {len(y_probability):,}")

    print("\nFirst 10 probabilities:")
    print(np.round(y_probability[:10], 4))

    # --------------------------------------------------------
    # 8. BASIC PERFORMANCE METRICS
    # --------------------------------------------------------

    print_section("MODEL PERFORMANCE")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)
    average_precision = average_precision_score(
        y_test,
        y_probability
    )

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Average Precision:  {average_precision:.4f}")

    # --------------------------------------------------------
    # 9. CONFUSION MATRIX
    # --------------------------------------------------------

    print_section("CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("Confusion matrix:")
    print(cm)

    print("\nInterpretation:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\nBusiness interpretation:")

    print(
        f"- {tp} at-risk sellers were correctly identified."
    )

    print(
        f"- {fn} at-risk sellers were missed by the model."
    )

    print(
        f"- {fp} sellers were incorrectly flagged as at risk."
    )

    print(
        f"- {tn} sellers were correctly classified as not at risk."
    )

    # --------------------------------------------------------
    # 10. FALSE NEGATIVE RATE
    # --------------------------------------------------------

    false_negative_rate = fn / (fn + tp)

    false_positive_rate = fp / (fp + tn)

    print("\nFalse negative rate:")
    print(f"{false_negative_rate:.4f}")

    print("\nFalse positive rate:")
    print(f"{false_positive_rate:.4f}")

    # --------------------------------------------------------
    # 11. ROC CURVE
    # --------------------------------------------------------

    print_section("ROC CURVE ANALYSIS")

    fpr, tpr, roc_thresholds = roc_curve(
        y_test,
        y_probability
    )

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(
        "The ROC curve shows how recall changes as the "
        "classification threshold changes."
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_PATH,
        dpi=150
    )

    plt.close()

    print(f"ROC curve saved to:\n{ROC_CURVE_PATH}")

    # --------------------------------------------------------
    # 12. PRECISION-RECALL CURVE
    # --------------------------------------------------------

    print_section("PRECISION-RECALL CURVE ANALYSIS")

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(
            y_test,
            y_probability
        )
    )

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print(
        "The Precision-Recall curve is especially useful "
        "because only 28.01% of sellers are at risk."
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"Random Forest (AP = {average_precision:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Random Forest Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PR_CURVE_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Precision-Recall curve saved to:\n"
        f"{PR_CURVE_PATH}"
    )

    # --------------------------------------------------------
    # 13. CONFUSION MATRIX VISUALIZATION
    # --------------------------------------------------------

    print_section("CONFUSION MATRIX VISUALIZATION")

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        [0, 1],
        ["Not At Risk", "At Risk"]
    )

    plt.yticks(
        [0, 1],
        ["Not At Risk", "At Risk"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Confusion matrix plot saved to:\n"
        f"{CONFUSION_MATRIX_PATH}"
    )

    # --------------------------------------------------------
    # 14. PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    print_section("PREDICTED PROBABILITY DISTRIBUTION")

    print(
        "Probability statistics:"
    )

    print(
        pd.Series(y_probability).describe()
    )

    plt.figure(figsize=(8, 6))

    plt.hist(
        y_probability[y_test == 0],
        bins=20,
        alpha=0.6,
        label="Actual: Not At Risk"
    )

    plt.hist(
        y_probability[y_test == 1],
        bins=20,
        alpha=0.6,
        label="Actual: At Risk"
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Default threshold = 0.50"
    )

    plt.xlabel("Predicted Probability of Being At Risk")
    plt.ylabel("Number of Sellers")
    plt.title("Random Forest Risk Probability Distribution")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PROBABILITY_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Probability distribution saved to:\n"
        f"{PROBABILITY_PATH}"
    )

    # --------------------------------------------------------
    # 15. THRESHOLD ANALYSIS
    # --------------------------------------------------------

    print_section("CLASSIFICATION THRESHOLD ANALYSIS")

    print(
        "Testing different probability thresholds..."
    )

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    threshold_results = []

    for threshold in thresholds:

        threshold_predictions = (
            y_probability >= threshold
        ).astype(int)

        threshold_accuracy = accuracy_score(
            y_test,
            threshold_predictions
        )

        threshold_precision = precision_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        threshold_recall = recall_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        threshold_f1 = f1_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        threshold_cm = confusion_matrix(
            y_test,
            threshold_predictions
        )

        threshold_tn, threshold_fp, threshold_fn, threshold_tp = (
            threshold_cm.ravel()
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "accuracy": threshold_accuracy,
                "precision": threshold_precision,
                "recall": threshold_recall,
                "f1_score": threshold_f1,
                "true_negatives": threshold_tn,
                "false_positives": threshold_fp,
                "false_negatives": threshold_fn,
                "true_positives": threshold_tp,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    print(
        threshold_df.round(4).to_string(index=False)
    )

    threshold_df.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False
    )

    print(
        f"\nThreshold analysis saved to:\n"
        f"{THRESHOLD_RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # 16. FIND BEST F1 THRESHOLD
    # --------------------------------------------------------

    print_section("BEST THRESHOLD ANALYSIS")

    best_threshold_row = threshold_df.loc[
        threshold_df["f1_score"].idxmax()
    ]

    best_threshold = best_threshold_row[
        "threshold"
    ]

    best_threshold_f1 = best_threshold_row[
        "f1_score"
    ]

    best_threshold_precision = best_threshold_row[
        "precision"
    ]

    best_threshold_recall = best_threshold_row[
        "recall"
    ]

    print(
        f"Best threshold by F1 score: "
        f"{best_threshold:.2f}"
    )

    print(
        f"F1 Score at this threshold: "
        f"{best_threshold_f1:.4f}"
    )

    print(
        f"Precision at this threshold: "
        f"{best_threshold_precision:.4f}"
    )

    print(
        f"Recall at this threshold: "
        f"{best_threshold_recall:.4f}"
    )

    print(
        "\nImportant:"
    )

    print(
        "The threshold producing the best F1 score is "
        "not automatically the final business threshold."
    )

    print(
        "The final threshold should depend on the business "
        "cost of missing an at-risk seller versus incorrectly "
        "flagging a healthy seller."
    )

    # --------------------------------------------------------
    # 17. DIAGNOSTIC SUMMARY
    # --------------------------------------------------------

    print_section("DIAGNOSTIC SUMMARY")

    diagnostic_results = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "value": accuracy
            },
            {
                "metric": "precision",
                "value": precision
            },
            {
                "metric": "recall",
                "value": recall
            },
            {
                "metric": "f1_score",
                "value": f1
            },
            {
                "metric": "roc_auc",
                "value": roc_auc
            },
            {
                "metric": "average_precision",
                "value": average_precision
            },
            {
                "metric": "false_positive_rate",
                "value": false_positive_rate
            },
            {
                "metric": "false_negative_rate",
                "value": false_negative_rate
            },
            {
                "metric": "best_f1_threshold",
                "value": best_threshold
            },
            {
                "metric": "best_threshold_f1",
                "value": best_threshold_f1
            },
        ]
    )

    print(
        diagnostic_results.round(4).to_string(index=False)
    )

    diagnostic_results.to_csv(
        OUTPUT_RESULTS_PATH,
        index=False
    )

    print(
        f"\nDiagnostic results saved to:\n"
        f"{OUTPUT_RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # 18. FINAL BUSINESS INTERPRETATION
    # --------------------------------------------------------

    print_section("BUSINESS INTERPRETATION")

    print(
        "The Random Forest model identifies at-risk sellers "
        "with strong overall performance."
    )

    print(
        f"At the default 0.50 threshold, recall is "
        f"{recall:.2%}."
    )

    print(
        f"This means the model identifies approximately "
        f"{recall:.2%} of the sellers who are actually at risk."
    )

    print(
        f"The model misses approximately "
        f"{false_negative_rate:.2%} of at-risk sellers."
    )

    print(
        "\nThis is important because a false negative means:"
    )

    print(
        "A seller is actually at risk, but the system fails "
        "to warn the business."
    )

    print(
        "\nThe next stage will therefore examine how changing "
        "the classification threshold changes the trade-off "
        "between precision and recall."
    )

    # --------------------------------------------------------
    # 19. COMPLETE
    # --------------------------------------------------------

    print_section("MODEL DIAGNOSTICS COMPLETE")

    print(
        f"Test rows analyzed: {len(test_df):,}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print(
        f"Default threshold: 0.50"
    )

    print(
        f"Best F1 threshold: {best_threshold:.2f}"
    )

    print(
        "\nThe test set was used only for model diagnostics "
        "and evaluation. The Random Forest model was not "
        "retrained."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()