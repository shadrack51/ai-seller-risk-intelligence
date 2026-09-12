
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DELIVERY_FILE = (
    PROCESSED_DIR / "temporal_delivery_ml_dataset.csv"
)

REVIEW_FILE = (
    PROCESSED_DIR / "temporal_review_ml_dataset.csv"
)


# =========================================================
# 2. TARGETS
# =========================================================

DELIVERY_TARGET = "future_delivery_risk"
REVIEW_TARGET = "future_review_risk"


# =========================================================
# 3. DELIVERY FEATURES
# =========================================================

DELIVERY_FEATURES = [
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
    "seller_history_orders",
    "seller_history_reviews",
    "recent_order_activity",
    "older_order_activity",
    "low_history_orders",
    "low_history_reviews",
    "sufficient_order_history",
    "sufficient_review_history",
    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",
    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",
    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",
    "recent_delivered_orders",
    "older_delivered_orders",
    "recent_late_orders",
    "older_late_orders",
    "recent_delivery_activity",
    "older_delivery_activity",
    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",
]


# =========================================================
# 4. REVIEW FEATURES
# =========================================================

REVIEW_FEATURES = [
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
    "seller_history_orders",
    "seller_history_reviews",
    "recent_order_activity",
    "older_order_activity",
    "low_history_orders",
    "low_history_reviews",
    "sufficient_order_history",
    "sufficient_review_history",
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
    "recent_review_activity",
    "older_review_activity",
    "review_score_declining",
    "low_review_rate_increasing",
]


# =========================================================
# 5. ESTABLISHED TEMPORAL SPLIT
# =========================================================
#
# This reproduces the exact population used in our previous
# temporal Random Forest experiments.
#
# TRAIN:
#     Includes all snapshots through 2018-02-01.
#
# VALIDATION:
#     2018-03-01 through 2018-05-01.
#
# TEST:
#     2018-06-01 through 2018-08-01.
#
# Expected populations:
#
#     Train       = 6,046
#     Validation  = 2,272
#     Test        = 2,348
#
# Total = 10,666
# =========================================================

def create_temporal_split(data):
    """
    Reproduce the established temporal split.
    """

    train_df = data[
        data["snapshot_date"]
        <= pd.Timestamp("2018-02-01")
    ].copy()

    validation_df = data[
        (data["snapshot_date"]
         >= pd.Timestamp("2018-03-01"))
        &
        (data["snapshot_date"]
         <= pd.Timestamp("2018-05-01"))
    ].copy()

    test_df = data[
        (data["snapshot_date"]
         >= pd.Timestamp("2018-06-01"))
        &
        (data["snapshot_date"]
         <= pd.Timestamp("2018-08-01"))
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


# =========================================================
# 6. CHECK TEMPORAL SPLIT
# =========================================================

def check_split_sizes(
    train_df,
    validation_df,
    test_df,
    dataset_name,
):
    """
    Ensure we reproduced the population used by the
    established experiment.
    """

    expected_train = 6046
    expected_validation = 2272
    expected_test = 2348

    actual_train = len(train_df)
    actual_validation = len(validation_df)
    actual_test = len(test_df)

    print(
        f"\n{dataset_name} temporal split"
    )

    print(
        "-" * 70
    )

    print(
        f"Train rows:       {actual_train:,}"
    )

    print(
        f"Validation rows:  {actual_validation:,}"
    )

    print(
        f"Test rows:        {actual_test:,}"
    )

    if actual_train != expected_train:
        raise ValueError(
            "\nTRAIN split mismatch.\n"
            f"Expected: {expected_train:,}\n"
            f"Actual:   {actual_train:,}\n"
        )

    if actual_validation != expected_validation:
        raise ValueError(
            "\nVALIDATION split mismatch.\n"
            f"Expected: {expected_validation:,}\n"
            f"Actual:   {actual_validation:,}\n"
        )

    if actual_test != expected_test:
        raise ValueError(
            "\nTEST split mismatch.\n"
            f"Expected: {expected_test:,}\n"
            f"Actual:   {actual_test:,}\n"
        )

    print(
        "\nSplit validation: PASS"
    )


# =========================================================
# 7. LOAD TARGET-SPECIFIC DATASET
# =========================================================

def load_dataset(
    file_path,
    target,
    features,
):
    """
    Load one target-specific dataset and validate
    its required columns.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{file_path}"
        )

    print(
        f"\nLoading:"
    )

    print(
        f"{file_path.name}"
    )

    data = pd.read_csv(
        file_path
    )

    print(
        f"Rows:    {len(data):,}"
    )

    print(
        f"Columns: {len(data.columns):,}"
    )

    missing_required = []

    required_columns = [
        "snapshot_date",
        target,
    ] + features

    for column in required_columns:
        if column not in data.columns:
            missing_required.append(
                column
            )

    if missing_required:

        print(
            "\nMissing required columns:"
        )

        for column in missing_required:
            print(
                f"  - {column}"
            )

        raise KeyError(
            "Dataset structure validation failed."
        )

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    # Confirm binary target.
    unique_values = sorted(
        data[target]
        .dropna()
        .unique()
        .tolist()
    )

    if not set(unique_values).issubset(
        {0, 1}
    ):
        raise ValueError(
            f"{target} contains unexpected "
            f"values: {unique_values}"
        )

    print(
        f"Predictors: {len(features)}"
    )

    print(
        "Feature validation: PASS"
    )

    print(
        "Target validation:  PASS"
    )

    return data


# =========================================================
# 8. BUILD LOGISTIC REGRESSION PIPELINE
# =========================================================

def build_logistic_model():
    """
    Build the baseline model.

    Pipeline:
        1. Median imputation
        2. Standardization
        3. L2-regularized Logistic Regression
    """

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# =========================================================
# 9. EVALUATE MODEL
# =========================================================

def evaluate_predictions(
    y_true,
    probabilities,
    threshold=0.50,
):
    """
    Evaluate classification performance at a given
    probability threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "average_precision": (
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# 10. VALIDATION THRESHOLD SEARCH
# =========================================================

def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Find the probability threshold that maximizes F1
    on VALIDATION data only.
    """

    thresholds = np.arange(
        0.05,
        0.91,
        0.01,
    )

    results = []

    for threshold in thresholds:

        metrics = evaluate_predictions(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    best_index = (
        results_df["f1"].idxmax()
    )

    best_row = results_df.loc[
        best_index
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    return (
        best_threshold,
        best_row,
        results_df,
    )


# =========================================================
# 11. PRINT METRICS
# =========================================================

def print_metrics(
    title,
    metrics,
):
    """
    Print model evaluation metrics.
    """

    print(
        f"\n{title}"
    )

    print(
        "-" * 55
    )

    print(
        f"Threshold:          "
        f"{metrics['threshold']:.2f}"
    )

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
        f"F1:                 "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"TN:                 "
        f"{metrics['tn']:,}"
    )

    print(
        f"FP:                 "
        f"{metrics['fp']:,}"
    )

    print(
        f"FN:                 "
        f"{metrics['fn']:,}"
    )

    print(
        f"TP:                 "
        f"{metrics['tp']:,}"
    )


# =========================================================
# 12. RUN ONE TARGET
# =========================================================

def run_target(
    dataset_name,
    data,
    target,
    features,
):
    """
    Train and evaluate Logistic Regression for one
    risk target.
    """

    print("\n")
    print("=" * 70)

    print(
        f"{dataset_name.upper()} "
        "— LOGISTIC REGRESSION BASELINE"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Create established temporal split
    # -----------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_temporal_split(
        data
    )

    check_split_sizes(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        dataset_name=dataset_name,
    )

    # -----------------------------------------------------
    # Separate X and y
    # -----------------------------------------------------

    X_train = train_df[
        features
    ]

    y_train = train_df[
        target
    ]

    X_validation = validation_df[
        features
    ]

    y_validation = validation_df[
        target
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
        target
    ]

    # -----------------------------------------------------
    # Target prevalence
    # -----------------------------------------------------

    print(
        "\nTarget prevalence"
    )

    print(
        "-" * 55
    )

    print(
        f"Train:       "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Validation:  "
        f"{y_validation.mean():.2%}"
    )

    print(
        f"Test:        "
        f"{y_test.mean():.2%}"
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    model = build_logistic_model()

    print(
        "\nTraining Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # -----------------------------------------------------
    # Generate probability predictions
    # -----------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            X_train
        )[:, 1]
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    train_metrics = evaluate_predictions(
        y_true=y_train,
        probabilities=train_probabilities,
        threshold=0.50,
    )

    # -----------------------------------------------------
    # VALIDATION @ 0.50
    # -----------------------------------------------------

    validation_metrics = evaluate_predictions(
        y_true=y_validation,
        probabilities=validation_probabilities,
        threshold=0.50,
    )

    # -----------------------------------------------------
    # FIND BEST VALIDATION THRESHOLD
    # -----------------------------------------------------

    (
        best_threshold,
        best_validation_row,
        threshold_results,
    ) = find_best_threshold(
        y_true=y_validation,
        probabilities=validation_probabilities,
    )

    # -----------------------------------------------------
    # TEST @ 0.50
    # -----------------------------------------------------

    test_metrics_050 = evaluate_predictions(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=0.50,
    )

    # -----------------------------------------------------
    # TEST @ FROZEN VALIDATION THRESHOLD
    # -----------------------------------------------------

    test_metrics_frozen = evaluate_predictions(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=best_threshold,
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print_metrics(
        "TRAIN @ 0.50",
        train_metrics,
    )

    print_metrics(
        "VALIDATION @ 0.50",
        validation_metrics,
    )

    print(
        "\nBEST VALIDATION THRESHOLD"
    )

    print(
        "-" * 55
    )

    print(
        f"Threshold:  "
        f"{best_threshold:.2f}"
    )

    print(
        f"Precision:  "
        f"{best_validation_row['precision']:.4f}"
    )

    print(
        f"Recall:     "
        f"{best_validation_row['recall']:.4f}"
    )

    print(
        f"F1:         "
        f"{best_validation_row['f1']:.4f}"
    )

    print_metrics(
        "TEST @ 0.50",
        test_metrics_050,
    )

    print_metrics(
        (
            "TEST @ FROZEN VALIDATION "
            f"THRESHOLD ({best_threshold:.2f})"
        ),
        test_metrics_frozen,
    )

    # =====================================================
    # LOGISTIC REGRESSION COEFFICIENTS
    # =====================================================

    logistic_model = (
        model.named_steps[
            "logistic_regression"
        ]
    )

    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient": (
                logistic_model.coef_[0]
            ),
        }
    )

    coefficients[
        "absolute_coefficient"
    ] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print(
        "\nTOP LOGISTIC REGRESSION COEFFICIENTS"
    )

    print(
        "-" * 70
    )

    print(
        coefficients[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # PROBABILITY SUMMARY
    # =====================================================

    print(
        "\nPREDICTED PROBABILITY SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Train mean probability:       "
        f"{train_probabilities.mean():.4f}"
    )

    print(
        f"Validation mean probability:  "
        f"{validation_probabilities.mean():.4f}"
    )

    print(
        f"Test mean probability:        "
        f"{test_probabilities.mean():.4f}"
    )

    return {
        "model": model,
        "best_threshold": best_threshold,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics_050": test_metrics_050,
        "test_metrics_frozen": test_metrics_frozen,
        "coefficients": coefficients,
        "threshold_results": threshold_results,
    }


# =========================================================
# 13. LOAD DELIVERY DATA
# =========================================================

delivery_df = load_dataset(
    file_path=DELIVERY_FILE,
    target=DELIVERY_TARGET,
    features=DELIVERY_FEATURES,
)


# =========================================================
# 14. LOAD REVIEW DATA
# =========================================================

review_df = load_dataset(
    file_path=REVIEW_FILE,
    target=REVIEW_TARGET,
    features=REVIEW_FEATURES,
)


# =========================================================
# 15. RUN DELIVERY MODEL
# =========================================================

delivery_results = run_target(
    dataset_name="Delivery risk",
    data=delivery_df,
    target=DELIVERY_TARGET,
    features=DELIVERY_FEATURES,
)


# =========================================================
# 16. RUN REVIEW MODEL
# =========================================================

review_results = run_target(
    dataset_name="Review risk",
    data=review_df,
    target=REVIEW_TARGET,
    features=REVIEW_FEATURES,
)


# =========================================================
# 17. FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print(
    "LOGISTIC REGRESSION BASELINE COMPLETE"
)
print("=" * 70)


print(
    "\nDELIVERY-RISK BASELINE"
)

print(
    "-" * 55
)

print(
    f"Validation ROC-AUC: "
    f"{delivery_results['validation_metrics']['roc_auc']:.4f}"
)

print(
    f"Test ROC-AUC:       "
    f"{delivery_results['test_metrics_050']['roc_auc']:.4f}"
)

print(
    f"Validation AP:      "
    f"{delivery_results['validation_metrics']['average_precision']:.4f}"
)

print(
    f"Test AP:            "
    f"{delivery_results['test_metrics_050']['average_precision']:.4f}"
)

print(
    f"Frozen threshold:   "
    f"{delivery_results['best_threshold']:.2f}"
)


print(
    "\nREVIEW-RISK BASELINE"
)

print(
    "-" * 55
)

print(
    f"Validation ROC-AUC: "
    f"{review_results['validation_metrics']['roc_auc']:.4f}"
)

print(
    f"Test ROC-AUC:       "
    f"{review_results['test_metrics_050']['roc_auc']:.4f}"
)

print(
    f"Validation AP:      "
    f"{review_results['validation_metrics']['average_precision']:.4f}"
)

print(
    f"Test AP:            "
    f"{review_results['test_metrics_050']['average_precision']:.4f}"
)

print(
    f"Frozen threshold:   "
    f"{review_results['best_threshold']:.2f}"
)


print(
    "\nEXPERIMENT INTEGRITY"
)

print(
    "-" * 55
)

print(
    "Same target-specific datasets were used."
)

print(
    "Same temporal populations were used."
)

print(
    "Threshold selection used validation only."
)

print(
    "The test set was not used for tuning."
)

