"""
Temporal Walk-Forward Validation with Distribution-Aware Features

Purpose
-------
Test whether the new distribution-aware delivery features improve
temporal generalization.

Experiment:

BASE MODEL
    39 predictors
    Mean walk-forward ROC-AUC = 0.6005

NEW MODEL
    69 predictors
    39 base predictors
    +
    30 distribution-aware predictors

The model architecture remains unchanged so that the feature-set
addition is the main experimental difference.

Walk-forward folds:

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

Final test period:

    2018-06 through 2018-08

The final test period is NOT used in this experiment.

Input
-----
data/processed/delivery_distribution_ml_features.csv

Previous baseline results
-----
data/processed/delivery_walk_forward_results.csv

Outputs
-------
data/processed/delivery_distribution_walk_forward_results.csv
data/processed/delivery_distribution_walk_forward_summary.csv
data/processed/delivery_feature_experiment_comparison.csv
"""


from pathlib import Path

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
    / "delivery_distribution_ml_features.csv"
)

BASELINE_RESULTS_FILE = (
    DATA_DIR
    / "delivery_walk_forward_results.csv"
)

RESULTS_FILE = (
    DATA_DIR
    / "delivery_distribution_walk_forward_results.csv"
)

SUMMARY_FILE = (
    DATA_DIR
    / "delivery_distribution_walk_forward_summary.csv"
)

COMPARISON_FILE = (
    DATA_DIR
    / "delivery_feature_experiment_comparison.csv"
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
# WALK-FORWARD FOLDS
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


def validate_dataset(df):
    """
    Validate the distribution-aware delivery dataset.
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
            "Required columns missing:\n"
            + "\n".join(
                missing_columns
            )
        )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates found."
        )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Missing target values found."
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
            f"Invalid target values: "
            f"{invalid_targets}"
        )

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

    if duplicates > 0:

        raise ValueError(
            "Duplicate seller/snapshot observations found."
        )


def get_features(df):
    """
    Return predictor columns.

    seller_id, snapshot_date, target, and all future-prefixed
    columns are excluded.
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

    return features


def validate_features(
    df,
    features,
):
    """
    Validate predictor columns.
    """

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
            + "\n".join(
                non_numeric
            )
        )

    leakage = [
        feature
        for feature in features
        if feature.startswith(
            "future_"
        )
    ]

    if leakage:

        raise ValueError(
            "Future predictors detected:\n"
            + "\n".join(
                leakage
            )
        )

    forbidden = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    forbidden_found = [
        feature
        for feature in features
        if feature in forbidden
    ]

    if forbidden_found:

        raise ValueError(
            "Forbidden predictor columns detected:\n"
            + "\n".join(
                forbidden_found
            )
        )


def build_model():
    """
    Build the fixed Random Forest pipeline.

    The architecture is intentionally identical to the
    previous walk-forward baseline.
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


def evaluate(
    model,
    X,
    y,
):
    """
    Calculate classification metrics.
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

    tn, fp, fn, tp = (
        cm.ravel()
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
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    return metrics


def select_period(
    df,
    start_date,
    end_date,
):
    """
    Select one chronological period.
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


def print_target_distribution(
    df,
    label,
):
    """Print target counts and percentages."""

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
        "DISTRIBUTION-AWARE DELIVERY WALK-FORWARD VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD INPUT
    # ========================================================

    print_section(
        "LOADING DISTRIBUTION-AWARE DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Distribution-aware dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_delivery_distribution_ml_features.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns):,}"
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
    # 3. DATA VALIDATION
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
    # 4. FINAL TEST PROTECTION
    # ========================================================

    print_section(
        "PROTECTING FINAL TEST PERIOD"
    )

    latest_validation_date = max(
        pd.Timestamp(
            fold[
                "validation_end"
            ]
        )
        for fold in FOLDS
    )

    print(
        f"Final test starts: "
        f"{FINAL_TEST_START.date()}"
    )

    print(
        f"Final test ends: "
        f"{FINAL_TEST_END.date()}"
    )

    print(
        f"Latest walk-forward validation date: "
        f"{latest_validation_date.date()}"
    )

    if (
        latest_validation_date
        >= FINAL_TEST_START
    ):

        raise ValueError(
            "Walk-forward validation overlaps final test."
        )

    print(
        "Final test period remains untouched -> PASS"
    )

    # ========================================================
    # 5. IDENTIFY FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING DISTRIBUTION-AWARE FEATURES"
    )

    features = get_features(
        df
    )

    validate_features(
        df,
        features,
    )

    print(
        f"Total predictors: "
        f"{len(features)}"
    )

    distribution_features = [
        feature
        for feature in features
        if (
            feature.startswith(
                "recent_distribution_"
            )
            or feature.startswith(
                "older_distribution_"
            )
            or feature
            in [
                "delivery_median_change",
                "delivery_std_change",
                "delivery_p90_change",
                "delivery_max_change",
                "delay_median_change",
                "delay_std_change",
                "delay_p90_change",
                "delay_max_change",
            ]
        )
    ]

    print(
        f"Distribution-aware predictors: "
        f"{len(distribution_features)}"
    )

    print(
        f"Existing base predictors: "
        f"{len(features) - len(distribution_features)}"
    )

    # ========================================================
    # 6. MISSING VALUE OVERVIEW
    # ========================================================

    print_section(
        "MISSING VALUE OVERVIEW"
    )

    missing = (
        df[
            features
        ]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = (
        missing[
            missing > 0
        ]
    )

    if missing.empty:

        print(
            "No missing predictor values."
        )

    else:

        print(
            missing.head(20)
            .to_string()
        )

    # ========================================================
    # 7. FOLD DESIGN
    # ========================================================

    print_section(
        "WALK-FORWARD FOLD DESIGN"
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

    # ========================================================
    # 8. RUN FOLDS
    # ========================================================

    print_section(
        "RUNNING DISTRIBUTION-AWARE WALK-FORWARD VALIDATION"
    )

    results = []

    for fold in FOLDS:

        fold_number = (
            fold["fold"]
        )

        train_start = pd.Timestamp(
            fold[
                "train_start"
            ]
        )

        train_end = pd.Timestamp(
            fold[
                "train_end"
            ]
        )

        validation_start = pd.Timestamp(
            fold[
                "validation_start"
            ]
        )

        validation_end = pd.Timestamp(
            fold[
                "validation_end"
            ]
        )

        print()
        print("-" * 70)
        print(
            f"FOLD {fold_number}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Select periods
        # ----------------------------------------------------

        train_df = select_period(
            df,
            train_start,
            train_end,
        )

        validation_df = select_period(
            df,
            validation_start,
            validation_end,
        )

        if train_df.empty:

            raise ValueError(
                f"Fold {fold_number}: "
                "training data is empty."
            )

        if validation_df.empty:

            raise ValueError(
                f"Fold {fold_number}: "
                "validation data is empty."
            )

        # ----------------------------------------------------
        # Validate chronological order
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
                "chronological ordering failed."
            )

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

        print_target_distribution(
            train_df,
            "Training target",
        )

        print()

        print_target_distribution(
            validation_df,
            "Validation target",
        )

        # ----------------------------------------------------
        # Build matrices
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Build fixed model
        # ----------------------------------------------------

        model = build_model()

        print(
            "\nTraining distribution-aware model..."
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Training complete."
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        train_metrics = evaluate(
            model,
            X_train,
            y_train,
        )

        validation_metrics = evaluate(
            model,
            X_validation,
            y_validation,
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            "\nTRAINING PERFORMANCE"
        )

        print(
            f"Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{train_metrics['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{train_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{train_metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{train_metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision: "
            f"{train_metrics['average_precision']:.4f}"
        )

        print(
            "\nVALIDATION PERFORMANCE"
        )

        print(
            f"Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{validation_metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision: "
            f"{validation_metrics['average_precision']:.4f}"
        )

        # ----------------------------------------------------
        # Generalization gaps
        # ----------------------------------------------------

        roc_auc_gap = (
            train_metrics[
                "roc_auc"
            ]
            -
            validation_metrics[
                "roc_auc"
            ]
        )

        f1_gap = (
            train_metrics[
                "f1_score"
            ]
            -
            validation_metrics[
                "f1_score"
            ]
        )

        print(
            "\nGENERALIZATION GAPS"
        )

        print(
            f"ROC-AUC gap: "
            f"{roc_auc_gap:.4f}"
        )

        print(
            f"F1 gap: "
            f"{f1_gap:.4f}"
        )

        # ----------------------------------------------------
        # Save fold row
        # ----------------------------------------------------

        results.append(
            {
                "fold": fold_number,

                "train_start": train_start,
                "train_end": train_end,

                "validation_start":
                    validation_start,
                "validation_end":
                    validation_end,

                "train_rows":
                    len(train_df),

                "validation_rows":
                    len(validation_df),

                "train_sellers":
                    train_df[
                        SELLER_COLUMN
                    ].nunique(),

                "validation_sellers":
                    validation_df[
                        SELLER_COLUMN
                    ].nunique(),

                "train_positive_rate":
                    y_train.mean(),

                "validation_positive_rate":
                    y_validation.mean(),

                "train_accuracy":
                    train_metrics[
                        "accuracy"
                    ],

                "train_precision":
                    train_metrics[
                        "precision"
                    ],

                "train_recall":
                    train_metrics[
                        "recall"
                    ],

                "train_f1":
                    train_metrics[
                        "f1_score"
                    ],

                "train_roc_auc":
                    train_metrics[
                        "roc_auc"
                    ],

                "train_average_precision":
                    train_metrics[
                        "average_precision"
                    ],

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_precision":
                    validation_metrics[
                        "precision"
                    ],

                "validation_recall":
                    validation_metrics[
                        "recall"
                    ],

                "validation_f1":
                    validation_metrics[
                        "f1_score"
                    ],

                "validation_roc_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "validation_average_precision":
                    validation_metrics[
                        "average_precision"
                    ],

                "roc_auc_gap":
                    roc_auc_gap,

                "f1_gap":
                    f1_gap,

                "validation_true_negatives":
                    validation_metrics[
                        "true_negatives"
                    ],

                "validation_false_positives":
                    validation_metrics[
                        "false_positives"
                    ],

                "validation_false_negatives":
                    validation_metrics[
                        "false_negatives"
                    ],

                "validation_true_positives":
                    validation_metrics[
                        "true_positives"
                    ],
            }
        )

    # ========================================================
    # 9. BUILD RESULTS
    # ========================================================

    print_section(
        "DISTRIBUTION-AWARE WALK-FORWARD RESULTS"
    )

    results_df = pd.DataFrame(
        results
    )

    display_columns = [
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

    print(
        results_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 10. PERFORMANCE SUMMARY
    # ========================================================

    print_section(
        "DISTRIBUTION-AWARE PERFORMANCE SUMMARY"
    )

    metric_columns = [
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
        "validation_average_precision",
    ]

    summary_rows = []

    for column in metric_columns:

        values = (
            results_df[
                column
            ]
            .astype(float)
        )

        summary_rows.append(
            {
                "metric": column,
                "mean": values.mean(),
                "std": values.std(
                    ddof=0
                ),
                "minimum": values.min(),
                "maximum": values.max(),
                "range": (
                    values.max()
                    -
                    values.min()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        summary_df
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 11. COMPARE AGAINST BASELINE
    # ========================================================

    print_section(
        "COMPARING AGAINST BASELINE FEATURE SET"
    )

    if not BASELINE_RESULTS_FILE.exists():

        print(
            "Baseline walk-forward results file not found."
        )

        print(
            "Feature comparison will be skipped."
        )

        comparison_df = pd.DataFrame()

    else:

        baseline_df = pd.read_csv(
            BASELINE_RESULTS_FILE
        )

        required_baseline_columns = [
            "fold",
            "validation_roc_auc",
            "validation_average_precision",
            "validation_recall",
            "validation_f1",
        ]

        missing_baseline_columns = [
            column
            for column
            in required_baseline_columns
            if column
            not in baseline_df.columns
        ]

        if missing_baseline_columns:

            raise ValueError(
                "Baseline results missing columns:\n"
                +
                "\n".join(
                    missing_baseline_columns
                )
            )

        comparison_df = (
            results_df[
                [
                    "fold",
                    "validation_roc_auc",
                    "validation_average_precision",
                    "validation_recall",
                    "validation_f1",
                ]
            ]
            .rename(
                columns={
                    "validation_roc_auc":
                        "distribution_roc_auc",
                    "validation_average_precision":
                        "distribution_average_precision",
                    "validation_recall":
                        "distribution_recall",
                    "validation_f1":
                        "distribution_f1",
                }
            )
            .merge(
                baseline_df[
                    [
                        "fold",
                        "validation_roc_auc",
                        "validation_average_precision",
                        "validation_recall",
                        "validation_f1",
                    ]
                ]
                .rename(
                    columns={
                        "validation_roc_auc":
                            "baseline_roc_auc",
                        "validation_average_precision":
                            "baseline_average_precision",
                        "validation_recall":
                            "baseline_recall",
                        "validation_f1":
                            "baseline_f1",
                    }
                ),
                on="fold",
                how="inner",
                validate="one_to_one",
            )
        )

        comparison_df[
            "roc_auc_change"
        ] = (
            comparison_df[
                "distribution_roc_auc"
            ]
            -
            comparison_df[
                "baseline_roc_auc"
            ]
        )

        comparison_df[
            "average_precision_change"
        ] = (
            comparison_df[
                "distribution_average_precision"
            ]
            -
            comparison_df[
                "baseline_average_precision"
            ]
        )

        comparison_df[
            "recall_change"
        ] = (
            comparison_df[
                "distribution_recall"
            ]
            -
            comparison_df[
                "baseline_recall"
            ]
        )

        comparison_df[
            "f1_change"
        ] = (
            comparison_df[
                "distribution_f1"
            ]
            -
            comparison_df[
                "baseline_f1"
            ]
        )

        print(
            comparison_df
            .round(4)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Average comparison
        # ----------------------------------------------------

        print(
            "\nMEAN PERFORMANCE COMPARISON"
        )

        baseline_mean_auc = (
            comparison_df[
                "baseline_roc_auc"
            ]
            .mean()
        )

        distribution_mean_auc = (
            comparison_df[
                "distribution_roc_auc"
            ]
            .mean()
        )

        auc_change = (
            distribution_mean_auc
            -
            baseline_mean_auc
        )

        baseline_mean_ap = (
            comparison_df[
                "baseline_average_precision"
            ]
            .mean()
        )

        distribution_mean_ap = (
            comparison_df[
                "distribution_average_precision"
            ]
            .mean()
        )

        ap_change = (
            distribution_mean_ap
            -
            baseline_mean_ap
        )

        baseline_mean_recall = (
            comparison_df[
                "baseline_recall"
            ]
            .mean()
        )

        distribution_mean_recall = (
            comparison_df[
                "distribution_recall"
            ]
            .mean()
        )

        recall_change = (
            distribution_mean_recall
            -
            baseline_mean_recall
        )

        baseline_mean_f1 = (
            comparison_df[
                "baseline_f1"
            ]
            .mean()
        )

        distribution_mean_f1 = (
            comparison_df[
                "distribution_f1"
            ]
            .mean()
        )

        f1_change = (
            distribution_mean_f1
            -
            baseline_mean_f1
        )

        print(
            f"Baseline mean ROC-AUC: "
            f"{baseline_mean_auc:.4f}"
        )

        print(
            f"Distribution mean ROC-AUC: "
            f"{distribution_mean_auc:.4f}"
        )

        print(
            f"ROC-AUC change: "
            f"{auc_change:+.4f}"
        )

        print(
            f"\nBaseline mean Average Precision: "
            f"{baseline_mean_ap:.4f}"
        )

        print(
            f"Distribution mean Average Precision: "
            f"{distribution_mean_ap:.4f}"
        )

        print(
            f"Average Precision change: "
            f"{ap_change:+.4f}"
        )

        print(
            f"\nBaseline mean Recall: "
            f"{baseline_mean_recall:.4f}"
        )

        print(
            f"Distribution mean Recall: "
            f"{distribution_mean_recall:.4f}"
        )

        print(
            f"Recall change: "
            f"{recall_change:+.4f}"
        )

        print(
            f"\nBaseline mean F1: "
            f"{baseline_mean_f1:.4f}"
        )

        print(
            f"Distribution mean F1: "
            f"{distribution_mean_f1:.4f}"
        )

        print(
            f"F1 change: "
            f"{f1_change:+.4f}"
        )

    # ========================================================
    # 12. INTERPRETATION
    # ========================================================

    print_section(
        "FEATURE EXPERIMENT INTERPRETATION"
    )

    mean_validation_auc = (
        results_df[
            "validation_roc_auc"
        ]
        .mean()
    )

    minimum_validation_auc = (
        results_df[
            "validation_roc_auc"
        ]
        .min()
    )

    auc_std = (
        results_df[
            "validation_roc_auc"
        ]
        .std(
            ddof=0
        )
    )

    print(
        f"Mean validation ROC-AUC: "
        f"{mean_validation_auc:.4f}"
    )

    print(
        f"Worst validation ROC-AUC: "
        f"{minimum_validation_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC standard deviation: "
        f"{auc_std:.4f}"
    )

    if not comparison_df.empty:

        if auc_change > 0:

            print(
                "\nThe distribution-aware feature set "
                "improved mean ROC-AUC."
            )

        elif auc_change < 0:

            print(
                "\nThe distribution-aware feature set "
                "reduced mean ROC-AUC."
            )

        else:

            print(
                "\nThe distribution-aware feature set "
                "did not change mean ROC-AUC."
            )

        if auc_std < 0.05:

            print(
                "Validation ROC-AUC variation remains "
                "relatively low across folds."
            )

        else:

            print(
                "Validation ROC-AUC varies substantially "
                "across folds."
            )

    print(
        "\nThis is a feature-set experiment, not a "
        "hyperparameter-tuning experiment."
    )

    # ========================================================
    # 13. SAVE RESULTS
    # ========================================================

    print_section(
        "SAVING EXPERIMENT RESULTS"
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
        f"Walk-forward results saved to:\n"
        f"{RESULTS_FILE}"
    )

    print(
        f"\nWalk-forward summary saved to:\n"
        f"{SUMMARY_FILE}"
    )

    if not comparison_df.empty:

        comparison_df.to_csv(
            COMPARISON_FILE,
            index=False,
        )

        print(
            f"\nFeature experiment comparison saved to:\n"
            f"{COMPARISON_FILE}"
        )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print_section(
        "DISTRIBUTION-AWARE WALK-FORWARD EXPERIMENT COMPLETE"
    )

    print(
        f"Folds evaluated: "
        f"{len(FOLDS)}"
    )

    print(
        f"Total predictors: "
        f"{len(features)}"
    )

    print(
        f"Distribution-aware predictors: "
        f"{len(distribution_features)}"
    )

    print(
        f"Mean validation ROC-AUC: "
        f"{mean_validation_auc:.4f}"
    )

    print(
        f"Minimum validation ROC-AUC: "
        f"{minimum_validation_auc:.4f}"
    )

    print(
        f"ROC-AUC standard deviation: "
        f"{auc_std:.4f}"
    )

    print(
        "\nFinal June-August 2018 test period "
        "was not used."
    )

    print(
        "No final production model was created."
    )

    print(
        "\nThe purpose of this experiment was to determine "
        "whether distribution-aware historical delivery "
        "features improve temporal generalization."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()