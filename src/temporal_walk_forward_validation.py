"""
Temporal Walk-Forward Validation

Purpose
-------
Evaluate the delivery-risk prediction problem across multiple
historical time periods using expanding-window validation.

The objective is to determine whether predictive performance
is stable across time.

This script does NOT use the final June-August 2018 test period.

Expanding validation windows:

Fold 1
    Train: 2017-01 through 2017-06
    Validate: 2017-07 through 2017-09

Fold 2
    Train: 2017-01 through 2017-09
    Validate: 2017-10 through 2017-12

Fold 3
    Train: 2017-01 through 2017-12
    Validate: 2018-01 through 2018-03

Fold 4
    Train: 2017-01 through 2018-02
    Validate: 2018-03 through 2018-05

The final test period remains:

    2018-06 through 2018-08

Input
-----
data/processed/target_delivery_features.csv

Outputs
-------
data/processed/delivery_walk_forward_results.csv
data/processed/delivery_walk_forward_summary.csv
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

INPUT_FILE = (
    DATA_DIR
    / "target_delivery_features.csv"
)

RESULTS_FILE = (
    DATA_DIR
    / "delivery_walk_forward_results.csv"
)

SUMMARY_FILE = (
    DATA_DIR
    / "delivery_walk_forward_summary.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_delivery_risk"


# ============================================================
# FINAL TEST PROTECTION
# ============================================================

FINAL_TEST_START = pd.Timestamp(
    "2018-06-01"
)

FINAL_TEST_END = pd.Timestamp(
    "2018-08-01"
)


# ============================================================
# WALK-FORWARD FOLD DEFINITIONS
# ============================================================

FOLDS = [
    {
        "fold": 1,
        "train_start": "2017-01-01",
        "train_end": "2017-06-01",
        "validation_start": "2017-07-01",
        "validation_end": "2017-09-01",
    },
    {
        "fold": 2,
        "train_start": "2017-01-01",
        "train_end": "2017-09-01",
        "validation_start": "2017-10-01",
        "validation_end": "2017-12-01",
    },
    {
        "fold": 3,
        "train_start": "2017-01-01",
        "train_end": "2017-12-01",
        "validation_start": "2018-01-01",
        "validation_end": "2018-03-01",
    },
    {
        "fold": 4,
        "train_start": "2017-01-01",
        "train_end": "2018-02-01",
        "validation_start": "2018-03-01",
        "validation_end": "2018-05-01",
    },
]


# ============================================================
# MODEL CONFIGURATION
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


def validate_dataset(df):
    """
    Validate the input delivery dataset.
    """

    required_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                missing_columns
            )
        )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates detected."
        )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Missing target values detected."
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
            f"Invalid target values detected: "
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
            "Duplicate seller/snapshot observations detected."
        )


def validate_features(feature_columns):
    """
    Ensure predictors do not contain forbidden future variables
    or seller identifiers.
    """

    forbidden = [
        SELLER_COLUMN,
        TARGET_COLUMN,
    ]

    leakage = [
        column
        for column in feature_columns
        if column in forbidden
        or column.startswith("future_")
    ]

    if leakage:

        raise ValueError(
            "Forbidden predictor columns detected:\n"
            + "\n".join(leakage)
        )


def build_pipeline():
    """
    Build the same Random Forest architecture used in the
    previous delivery model.

    No hyperparameter tuning is performed here.
    """

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


def calculate_metrics(
    model,
    X,
    y,
):
    """
    Generate predictions and calculate evaluation metrics.
    """

    predictions = model.predict(
        X
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

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
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    return metrics


def prepare_split(
    df,
    start_date,
    end_date,
):
    """
    Return a chronological subset using inclusive boundaries.
    """

    return df[
        (
            df[DATE_COLUMN]
            >= start_date
        )
        &
        (
            df[DATE_COLUMN]
            <= end_date
        )
    ].copy()


def print_distribution(
    df,
    label,
):
    """
    Print binary target distribution.
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DELIVERY MODEL WALK-FORWARD VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING DELIVERY FEATURE DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_target_specific_features.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # ========================================================
    # 2. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING SNAPSHOT DATES"
    )

    df[
        DATE_COLUMN
    ] = pd.to_datetime(
        df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    print(
        "Snapshot date conversion -> PASS"
    )

    # ========================================================
    # 3. VALIDATE DATA
    # ========================================================

    print_section(
        "DATA VALIDATION"
    )

    validate_dataset(
        df
    )

    print(
        "Dataset validation -> PASS"
    )

    # ========================================================
    # 4. PROTECT FINAL TEST PERIOD
    # ========================================================

    print_section(
        "FINAL TEST PERIOD PROTECTION"
    )

    print(
        f"Final test starts: "
        f"{FINAL_TEST_START.date()}"
    )

    print(
        f"Final test ends: "
        f"{FINAL_TEST_END.date()}"
    )

    walk_forward_max_date = max(
        pd.Timestamp(
            fold["validation_end"]
        )
        for fold in FOLDS
    )

    print(
        f"Latest walk-forward validation date: "
        f"{walk_forward_max_date.date()}"
    )

    if (
        walk_forward_max_date
        >= FINAL_TEST_START
    ):

        raise ValueError(
            "A walk-forward fold overlaps the final test period."
        )

    print(
        "Final test period remains untouched -> PASS"
    )

    # ========================================================
    # 5. DETERMINE FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING DELIVERY MODEL FEATURES"
    )

    excluded_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    validate_features(
        feature_columns
    )

    print(
        f"Predictor features: "
        f"{len(feature_columns)}"
    )

    for number, feature in enumerate(
        feature_columns,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 6. BUILD FOLDS
    # ========================================================

    print_section(
        "WALK-FORWARD VALIDATION DESIGN"
    )

    for fold in FOLDS:

        print(
            f"Fold {fold['fold']}: "
            f"Train "
            f"{fold['train_start']} → "
            f"{fold['train_end']} | "
            f"Validation "
            f"{fold['validation_start']} → "
            f"{fold['validation_end']}"
        )

    print(
        "\nTraining windows expand forward in time."
    )

    print(
        "Validation periods always occur after their "
        "corresponding training periods."
    )

    # ========================================================
    # 7. RUN FOLDS
    # ========================================================

    print_section(
        "RUNNING WALK-FORWARD VALIDATION"
    )

    all_results = []

    for fold_definition in FOLDS:

        fold_number = (
            fold_definition[
                "fold"
            ]
        )

        train_start = pd.Timestamp(
            fold_definition[
                "train_start"
            ]
        )

        train_end = pd.Timestamp(
            fold_definition[
                "train_end"
            ]
        )

        validation_start = pd.Timestamp(
            fold_definition[
                "validation_start"
            ]
        )

        validation_end = pd.Timestamp(
            fold_definition[
                "validation_end"
            ]
        )

        print()
        print("-" * 70)
        print(
            f"FOLD {fold_number}"
        )
        print("-" * 70)

        print(
            f"Training period: "
            f"{train_start.date()} → "
            f"{train_end.date()}"
        )

        print(
            f"Validation period: "
            f"{validation_start.date()} → "
            f"{validation_end.date()}"
        )

        # ----------------------------------------------------
        # Create chronological subsets
        # ----------------------------------------------------

        train_df = prepare_split(
            df,
            train_start,
            train_end,
        )

        validation_df = prepare_split(
            df,
            validation_start,
            validation_end,
        )

        if train_df.empty:

            raise ValueError(
                f"Fold {fold_number}: "
                "training dataset is empty."
            )

        if validation_df.empty:

            raise ValueError(
                f"Fold {fold_number}: "
                "validation dataset is empty."
            )

        # ----------------------------------------------------
        # Verify chronology
        # ----------------------------------------------------

        train_last = (
            train_df[
                DATE_COLUMN
            ].max()
        )

        validation_first = (
            validation_df[
                DATE_COLUMN
            ].min()
        )

        if not (
            train_last
            < validation_first
        ):

            raise ValueError(
                f"Fold {fold_number}: "
                "training and validation overlap."
            )

        print(
            f"Training rows: "
            f"{len(train_df):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation_df):,}"
        )

        # ----------------------------------------------------
        # Target distributions
        # ----------------------------------------------------

        print()

        print_distribution(
            train_df,
            "Training target",
        )

        print()

        print_distribution(
            validation_df,
            "Validation target",
        )

        # ----------------------------------------------------
        # Build matrices
        # ----------------------------------------------------

        X_train = train_df[
            feature_columns
        ].copy()

        y_train = train_df[
            TARGET_COLUMN
        ].copy()

        X_validation = validation_df[
            feature_columns
        ].copy()

        y_validation = validation_df[
            TARGET_COLUMN
        ].copy()

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        print(
            "\nBuilding Random Forest pipeline..."
        )

        model = build_pipeline()

        # ----------------------------------------------------
        # Train only on fold training data
        # ----------------------------------------------------

        print(
            "Training on fold training data only..."
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Training complete."
        )

        # ----------------------------------------------------
        # Evaluate training
        # ----------------------------------------------------

        train_metrics = calculate_metrics(
            model,
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Evaluate validation
        # ----------------------------------------------------

        validation_metrics = calculate_metrics(
            model,
            X_validation,
            y_validation,
        )

        # ----------------------------------------------------
        # Calculate generalization gaps
        # ----------------------------------------------------

        roc_auc_gap = (
            train_metrics["roc_auc"]
            - validation_metrics["roc_auc"]
        )

        f1_gap = (
            train_metrics["f1_score"]
            - validation_metrics["f1_score"]
        )

        # ----------------------------------------------------
        # Print fold performance
        # ----------------------------------------------------

        print(
            "\nTRAINING PERFORMANCE"
        )

        print(
            f"Accuracy:           "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision:          "
            f"{train_metrics['precision']:.4f}"
        )

        print(
            f"Recall:             "
            f"{train_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:           "
            f"{train_metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC:            "
            f"{train_metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision:  "
            f"{train_metrics['average_precision']:.4f}"
        )

        print(
            "\nVALIDATION PERFORMANCE"
        )

        print(
            f"Accuracy:           "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision:          "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"Recall:             "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:           "
            f"{validation_metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC:            "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision:  "
            f"{validation_metrics['average_precision']:.4f}"
        )

        print(
            "\nGENERALIZATION GAPS"
        )

        print(
            f"ROC-AUC gap: "
            f"{roc_auc_gap:.4f}"
        )

        print(
            f"F1 gap:     "
            f"{f1_gap:.4f}"
        )

        # ----------------------------------------------------
        # Store fold results
        # ----------------------------------------------------

        result_row = {
            "fold": fold_number,
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "train_sellers": (
                train_df[
                    SELLER_COLUMN
                ].nunique()
            ),
            "validation_sellers": (
                validation_df[
                    SELLER_COLUMN
                ].nunique()
            ),
            "train_positive_rate": (
                y_train.mean()
            ),
            "validation_positive_rate": (
                y_validation.mean()
            ),
            "train_accuracy": (
                train_metrics["accuracy"]
            ),
            "train_precision": (
                train_metrics["precision"]
            ),
            "train_recall": (
                train_metrics["recall"]
            ),
            "train_f1": (
                train_metrics["f1_score"]
            ),
            "train_roc_auc": (
                train_metrics["roc_auc"]
            ),
            "train_average_precision": (
                train_metrics[
                    "average_precision"
                ]
            ),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "validation_precision": (
                validation_metrics["precision"]
            ),
            "validation_recall": (
                validation_metrics["recall"]
            ),
            "validation_f1": (
                validation_metrics["f1_score"]
            ),
            "validation_roc_auc": (
                validation_metrics["roc_auc"]
            ),
            "validation_average_precision": (
                validation_metrics[
                    "average_precision"
                ]
            ),
            "roc_auc_gap": roc_auc_gap,
            "f1_gap": f1_gap,
            "validation_true_negatives": (
                validation_metrics[
                    "true_negatives"
                ]
            ),
            "validation_false_positives": (
                validation_metrics[
                    "false_positives"
                ]
            ),
            "validation_false_negatives": (
                validation_metrics[
                    "false_negatives"
                ]
            ),
            "validation_true_positives": (
                validation_metrics[
                    "true_positives"
                ]
            ),
        }

        all_results.append(
            result_row
        )

    # ========================================================
    # 8. BUILD RESULTS TABLE
    # ========================================================

    print_section(
        "WALK-FORWARD RESULTS"
    )

    results_df = pd.DataFrame(
        all_results
    )

    print(
        results_df[
            [
                "fold",
                "train_rows",
                "validation_rows",
                "validation_positive_rate",
                "validation_roc_auc",
                "validation_average_precision",
                "validation_recall",
                "validation_f1",
                "roc_auc_gap",
                "f1_gap",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # 9. PERFORMANCE STABILITY SUMMARY
    # ========================================================

    print_section(
        "PERFORMANCE STABILITY SUMMARY"
    )

    validation_metrics = [
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
        "validation_average_precision",
    ]

    summary_rows = []

    for metric in validation_metrics:

        values = (
            results_df[
                metric
            ]
            .astype(float)
        )

        summary_rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(
                    ddof=0
                ),
                "minimum": values.min(),
                "maximum": values.max(),
                "range": (
                    values.max()
                    - values.min()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        summary_df
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # 10. AVERAGE GENERALIZATION GAP
    # ========================================================

    print_section(
        "GENERALIZATION GAP SUMMARY"
    )

    print(
        f"Average ROC-AUC gap: "
        f"{results_df['roc_auc_gap'].mean():.4f}"
    )

    print(
        f"Maximum ROC-AUC gap: "
        f"{results_df['roc_auc_gap'].max():.4f}"
    )

    print(
        f"Average F1 gap: "
        f"{results_df['f1_gap'].mean():.4f}"
    )

    print(
        f"Maximum F1 gap: "
        f"{results_df['f1_gap'].max():.4f}"
    )

    # ========================================================
    # 11. STABILITY INTERPRETATION
    # ========================================================

    print_section(
        "TEMPORAL STABILITY INTERPRETATION"
    )

    mean_auc = (
        results_df[
            "validation_roc_auc"
        ].mean()
    )

    min_auc = (
        results_df[
            "validation_roc_auc"
        ].min()
    )

    auc_std = (
        results_df[
            "validation_roc_auc"
        ].std(
            ddof=0
        )
    )

    print(
        f"Mean validation ROC-AUC: "
        f"{mean_auc:.4f}"
    )

    print(
        f"Worst validation ROC-AUC: "
        f"{min_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC standard deviation: "
        f"{auc_std:.4f}"
    )

    if mean_auc <= 0.55:

        print(
            "\nWARNING:"
        )

        print(
            "The delivery model shows little useful "
            "predictive discrimination across historical periods."
        )

    elif min_auc <= 0.55:

        print(
            "\nWARNING:"
        )

        print(
            "At least one historical validation period "
            "shows near-random discrimination."
        )

    else:

        print(
            "\nThe model shows some predictive discrimination "
            "across the evaluated periods."
        )

    if auc_std > 0.10:

        print(
            "WARNING:"
        )

        print(
            "Performance varies substantially across time."
        )

    else:

        print(
            "Validation ROC-AUC variation is relatively limited."
        )

    # ========================================================
    # 12. SAVE RESULTS
    # ========================================================

    print_section(
        "SAVING WALK-FORWARD RESULTS"
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        f"Fold results saved to:\n"
        f"{RESULTS_FILE}"
    )

    print(
        f"\nPerformance summary saved to:\n"
        f"{SUMMARY_FILE}"
    )

    # ========================================================
    # 13. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL WALK-FORWARD VALIDATION COMPLETE"
    )

    print(
        f"Folds evaluated: "
        f"{len(FOLDS)}"
    )

    print(
        f"Predictor features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Mean validation ROC-AUC: "
        f"{mean_auc:.4f}"
    )

    print(
        f"Minimum validation ROC-AUC: "
        f"{min_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC standard deviation: "
        f"{auc_std:.4f}"
    )

    print(
        "\nThe June-August 2018 final test period "
        "was not used."
    )

    print(
        "No final production model was created by this script."
    )

    print(
        "\nThe purpose of this experiment is to determine "
        "whether delivery-risk prediction is temporally stable."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()