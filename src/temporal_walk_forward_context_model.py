"""
Temporal Walk-Forward Validation with Context-Aware Features

Purpose
-------
Evaluate the 105-feature context-aware delivery dataset using
the same walk-forward validation design used for the baseline
delivery model.

Feature sets:

BASELINE
--------
39 base delivery predictors

DISTRIBUTION-AWARE
------------------
39 base
+
30 distribution predictors
=
69 predictors

CONTEXT-AWARE
-------------
39 base
+
30 distribution predictors
+
36 context predictors
=
105 predictors

The model architecture remains unchanged.

Random Forest:
    n_estimators = 300
    max_depth = None
    min_samples_leaf = 2
    random_state = 42

Walk-forward folds:

Fold 1
    Train: 2017-01-01 -> 2017-06-01
    Validation: 2017-07-01 -> 2017-09-01

Fold 2
    Train: 2017-01-01 -> 2017-09-01
    Validation: 2017-10-01 -> 2017-12-01

Fold 3
    Train: 2017-01-01 -> 2017-12-01
    Validation: 2018-01-01 -> 2018-03-01

Fold 4
    Train: 2017-01-01 -> 2018-02-01
    Validation: 2018-03-01 -> 2018-05-01

Final test period:

    2018-06-01 -> 2018-08-01

The final test period is NOT used.

Input
-----
data/processed/delivery_context_ml_features.csv

Previous baseline results
-----
data/processed/delivery_walk_forward_results.csv

Previous distribution experiment
-----
data/processed/delivery_distribution_walk_forward_results.csv

Outputs
-------
data/processed/delivery_context_walk_forward_results.csv
data/processed/delivery_context_walk_forward_summary.csv
data/processed/delivery_feature_experiment_full_comparison.csv
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
    / "delivery_context_ml_features.csv"
)

BASELINE_FILE = (
    DATA_DIR
    / "delivery_walk_forward_results.csv"
)

DISTRIBUTION_FILE = (
    DATA_DIR
    / "delivery_distribution_walk_forward_results.csv"
)

RESULTS_FILE = (
    DATA_DIR
    / "delivery_context_walk_forward_results.csv"
)

SUMMARY_FILE = (
    DATA_DIR
    / "delivery_context_walk_forward_summary.csv"
)

COMPARISON_FILE = (
    DATA_DIR
    / "delivery_feature_experiment_full_comparison.csv"
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
# HELPERS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_input_dataset(df):
    """Validate the context-aware temporal dataset."""

    required = [
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(missing)
        )

    if df[
        DATE_COLUMN
    ].isna().any():

        raise ValueError(
            "Invalid snapshot dates detected."
        )

    if df[
        TARGET_COLUMN
    ].isna().any():

        raise ValueError(
            "Missing target values detected."
        )

    invalid_targets = (
        set(
            df[
                TARGET_COLUMN
            ].unique()
        )
        -
        {0, 1}
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
            "Duplicate seller/snapshot observations detected."
        )


def get_features(df):
    """
    Return predictor columns.

    Identifiers, dates, target, and future-prefixed columns
    are excluded.
    """

    excluded = {
        SELLER_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    }

    return [
        column
        for column in df.columns
        if (
            column not in excluded
            and not column.startswith(
                "future_"
            )
        )
    ]


def identify_feature_blocks(
    features,
):
    """Separate predictors into the three feature blocks."""

    distribution_features = [
        feature
        for feature in features
        if (
            feature.startswith(
                "recent_distribution_"
            )
            or
            feature.startswith(
                "older_distribution_"
            )
            or
            feature in [
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

    context_features = [
        feature
        for feature in features
        if (
            feature.startswith(
                "recent_context_"
            )
            or
            feature.startswith(
                "older_context_"
            )
            or
            feature.startswith(
                "context_"
            )
        )
    ]

    base_features = [
        feature
        for feature in features
        if (
            feature not in distribution_features
            and
            feature not in context_features
        )
    ]

    return (
        base_features,
        distribution_features,
        context_features,
    )


def build_model():
    """Create the fixed Random Forest pipeline."""

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


def select_period(
    df,
    start_date,
    end_date,
):
    """Select a chronological period."""

    return df[
        (
            df[
                DATE_COLUMN
            ]
            >= start_date
        )
        &
        (
            df[
                DATE_COLUMN
            ]
            <= end_date
        )
    ].copy()


def evaluate_model(
    model,
    X,
    y,
):
    """Calculate model evaluation metrics."""

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

    return {
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
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "average_precision":
            average_precision_score(
                y,
                probabilities,
            ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def print_target_distribution(
    df,
    label,
):
    """Print target counts and rates."""

    counts = (
        df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    rates = (
        df[
            TARGET_COLUMN
        ]
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
        rates
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CONTEXT-AWARE DELIVERY WALK-FORWARD VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING CONTEXT-AWARE DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_delivery_context_ml_features.py first."
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
    # 2. DATES
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
    # 3. VALIDATION
    # ========================================================

    print_section(
        "DATA VALIDATION"
    )

    validate_input_dataset(
        df
    )

    print(
        "Dataset validation -> PASS"
    )

    # ========================================================
    # 4. FINAL TEST PROTECTION
    # ========================================================

    print_section(
        "FINAL TEST PERIOD PROTECTION"
    )

    latest_validation_end = max(
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
        f"{latest_validation_end.date()}"
    )

    if (
        latest_validation_end
        >= FINAL_TEST_START
    ):

        raise ValueError(
            "Walk-forward validation overlaps "
            "the final test period."
        )

    print(
        "Final test remains untouched -> PASS"
    )

    # ========================================================
    # 5. FEATURE BLOCKS
    # ========================================================

    print_section(
        "IDENTIFYING FEATURE BLOCKS"
    )

    features = get_features(
        df
    )

    (
        base_features,
        distribution_features,
        context_features,
    ) = identify_feature_blocks(
        features
    )

    print(
        f"Base predictors: "
        f"{len(base_features)}"
    )

    print(
        f"Distribution predictors: "
        f"{len(distribution_features)}"
    )

    print(
        f"Context predictors: "
        f"{len(context_features)}"
    )

    print(
        f"Total predictors: "
        f"{len(features)}"
    )

    expected_count = (
        len(base_features)
        +
        len(distribution_features)
        +
        len(context_features)
    )

    if expected_count != len(
        features
    ):

        raise ValueError(
            "Feature block counts do not reconcile."
        )

    print(
        "Feature block reconciliation -> PASS"
    )

    # ========================================================
    # 6. MISSING VALUES
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

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print(
            "No missing predictors."
        )

    else:

        print(
            missing.head(25)
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
            f"{fold['train_start']} -> "
            f"{fold['train_end']} | "
            f"Validation "
            f"{fold['validation_start']} -> "
            f"{fold['validation_end']}"
        )

    # ========================================================
    # 8. RUN WALK-FORWARD VALIDATION
    # ========================================================

    print_section(
        "RUNNING CONTEXT-AWARE WALK-FORWARD VALIDATION"
    )

    results = []

    for fold in FOLDS:

        fold_number = fold[
            "fold"
        ]

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
        # Select train and validation data
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
                f"Fold {fold_number} training set is empty."
            )

        if validation_df.empty:

            raise ValueError(
                f"Fold {fold_number} validation set is empty."
            )

        # ----------------------------------------------------
        # Chronological check
        # ----------------------------------------------------

        train_last = (
            train_df[
                DATE_COLUMN
            ]
            .max()
        )

        validation_first = (
            validation_df[
                DATE_COLUMN
            ]
            .min()
        )

        if not (
            train_last
            <
            validation_first
        ):

            raise ValueError(
                f"Fold {fold_number} "
                "violates chronological ordering."
            )

        print(
            f"Training period: "
            f"{train_start.date()} -> "
            f"{train_end.date()}"
        )

        print(
            f"Validation period: "
            f"{validation_start.date()} -> "
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
        # Feature matrices
        # ----------------------------------------------------

        X_train = (
            train_df[
                features
            ]
        )

        y_train = (
            train_df[
                TARGET_COLUMN
            ]
        )

        X_validation = (
            validation_df[
                features
            ]
        )

        y_validation = (
            validation_df[
                TARGET_COLUMN
            ]
        )

        print(
            f"\nTraining feature matrix: "
            f"{X_train.shape}"
        )

        print(
            f"Validation feature matrix: "
            f"{X_validation.shape}"
        )

        # ----------------------------------------------------
        # Build and fit model
        # ----------------------------------------------------

        print(
            "\nBuilding Random Forest pipeline..."
        )

        model = build_model()

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
        # Evaluate train
        # ----------------------------------------------------

        train_metrics = evaluate_model(
            model,
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Evaluate validation
        # ----------------------------------------------------

        validation_metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        # ----------------------------------------------------
        # Training performance
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
            f"{train_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:            "
            f"{train_metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision:  "
            f"{train_metrics['average_precision']:.4f}"
        )

        # ----------------------------------------------------
        # Validation performance
        # ----------------------------------------------------

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
            f"{validation_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:            "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision:  "
            f"{validation_metrics['average_precision']:.4f}"
        )

        # ----------------------------------------------------
        # Generalization gaps
        # ----------------------------------------------------

        roc_gap = (
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
                "f1"
            ]
            -
            validation_metrics[
                "f1"
            ]
        )

        print(
            "\nGENERALIZATION GAPS"
        )

        print(
            f"ROC-AUC gap: "
            f"{roc_gap:.4f}"
        )

        print(
            f"F1 gap:      "
            f"{f1_gap:.4f}"
        )

        # ----------------------------------------------------
        # Save fold result
        # ----------------------------------------------------

        results.append(
            {
                "fold": fold_number,

                "train_start":
                    train_start,

                "train_end":
                    train_end,

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
                        "f1"
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
                        "f1"
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
                    roc_gap,

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
    # 9. RESULTS TABLE
    # ========================================================

    print_section(
        "CONTEXT-AWARE WALK-FORWARD RESULTS"
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
    # 10. SUMMARY
    # ========================================================

    print_section(
        "CONTEXT-AWARE PERFORMANCE SUMMARY"
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

    for metric in metric_columns:

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
    # 11. FULL FEATURE EXPERIMENT COMPARISON
    # ========================================================

    print_section(
        "FULL FEATURE EXPERIMENT COMPARISON"
    )

    comparison_available = (
        BASELINE_FILE.exists()
        and
        DISTRIBUTION_FILE.exists()
    )

    if not comparison_available:

        print(
            "Previous experiment result files "
            "were not both found."
        )

        print(
            "The current context-aware results "
            "will still be saved."
        )

    else:

        baseline_df = pd.read_csv(
            BASELINE_FILE
        )

        distribution_df = pd.read_csv(
            DISTRIBUTION_FILE
        )

        required_columns = [
            "fold",
            "validation_roc_auc",
            "validation_average_precision",
            "validation_recall",
            "validation_f1",
        ]

        for label, source_df in [
            (
                "Baseline",
                baseline_df,
            ),
            (
                "Distribution-aware",
                distribution_df,
            ),
            (
                "Context-aware",
                results_df,
            ),
        ]:

            missing = [
                column
                for column
                in required_columns
                if column
                not in source_df.columns
            ]

            if missing:

                raise ValueError(
                    f"{label} results missing:\n"
                    +
                    "\n".join(
                        missing
                    )
                )

        baseline_compare = (
            baseline_df[
                required_columns
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
            )
        )

        distribution_compare = (
            distribution_df[
                required_columns
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
        )

        context_compare = (
            results_df[
                required_columns
            ]
            .rename(
                columns={
                    "validation_roc_auc":
                        "context_roc_auc",
                    "validation_average_precision":
                        "context_average_precision",
                    "validation_recall":
                        "context_recall",
                    "validation_f1":
                        "context_f1",
                }
            )
        )

        comparison_df = (
            baseline_compare
            .merge(
                distribution_compare,
                on="fold",
                how="inner",
                validate="one_to_one",
            )
            .merge(
                context_compare,
                on="fold",
                how="inner",
                validate="one_to_one",
            )
        )

        comparison_df[
            "distribution_vs_baseline_auc_change"
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
            "context_vs_baseline_auc_change"
        ] = (
            comparison_df[
                "context_roc_auc"
            ]
            -
            comparison_df[
                "baseline_roc_auc"
            ]
        )

        comparison_df[
            "context_vs_distribution_auc_change"
        ] = (
            comparison_df[
                "context_roc_auc"
            ]
            -
            comparison_df[
                "distribution_roc_auc"
            ]
        )

        comparison_df[
            "context_vs_baseline_recall_change"
        ] = (
            comparison_df[
                "context_recall"
            ]
            -
            comparison_df[
                "baseline_recall"
            ]
        )

        comparison_df[
            "context_vs_baseline_f1_change"
        ] = (
            comparison_df[
                "context_f1"
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
        # Mean comparison
        # ----------------------------------------------------

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

        context_mean_auc = (
            comparison_df[
                "context_roc_auc"
            ]
            .mean()
        )

        print(
            "\nMEAN ROC-AUC COMPARISON"
        )

        print(
            f"Base:         "
            f"{baseline_mean_auc:.4f}"
        )

        print(
            f"Distribution: "
            f"{distribution_mean_auc:.4f}"
        )

        print(
            f"Context:      "
            f"{context_mean_auc:.4f}"
        )

        print(
            f"\nDistribution vs baseline: "
            f"{distribution_mean_auc - baseline_mean_auc:+.4f}"
        )

        print(
            f"Context vs baseline:      "
            f"{context_mean_auc - baseline_mean_auc:+.4f}"
        )

        print(
            f"Context vs distribution:   "
            f"{context_mean_auc - distribution_mean_auc:+.4f}"
        )

        # ----------------------------------------------------
        # Mean recall
        # ----------------------------------------------------

        baseline_mean_recall = (
            comparison_df[
                "baseline_recall"
            ]
            .mean()
        )

        context_mean_recall = (
            comparison_df[
                "context_recall"
            ]
            .mean()
        )

        print(
            "\nMEAN RECALL COMPARISON"
        )

        print(
            f"Base:    "
            f"{baseline_mean_recall:.4f}"
        )

        print(
            f"Context: "
            f"{context_mean_recall:.4f}"
        )

        print(
            f"Context vs baseline: "
            f"{context_mean_recall - baseline_mean_recall:+.4f}"
        )

        # ----------------------------------------------------
        # Mean F1
        # ----------------------------------------------------

        baseline_mean_f1 = (
            comparison_df[
                "baseline_f1"
            ]
            .mean()
        )

        context_mean_f1 = (
            comparison_df[
                "context_f1"
            ]
            .mean()
        )

        print(
            "\nMEAN F1 COMPARISON"
        )

        print(
            f"Base:    "
            f"{baseline_mean_f1:.4f}"
        )

        print(
            f"Context: "
            f"{context_mean_f1:.4f}"
        )

        print(
            f"Context vs baseline: "
            f"{context_mean_f1 - baseline_mean_f1:+.4f}"
        )

    # ========================================================
    # 12. INTERPRETATION
    # ========================================================

    print_section(
        "TEMPORAL FEATURE EXPERIMENT INTERPRETATION"
    )

    context_mean_auc = (
        results_df[
            "validation_roc_auc"
        ]
        .mean()
    )

    context_min_auc = (
        results_df[
            "validation_roc_auc"
        ]
        .min()
    )

    context_std_auc = (
        results_df[
            "validation_roc_auc"
        ]
        .std(
            ddof=0
        )
    )

    print(
        f"Context-aware mean ROC-AUC: "
        f"{context_mean_auc:.4f}"
    )

    print(
        f"Context-aware minimum ROC-AUC: "
        f"{context_min_auc:.4f}"
    )

    print(
        f"Context-aware ROC-AUC standard deviation: "
        f"{context_std_auc:.4f}"
    )

    if comparison_available:

        auc_change = (
            context_mean_auc
            -
            baseline_mean_auc
        )

        if auc_change > 0:

            print(
                "\nContext features improved "
                "mean ROC-AUC relative to baseline."
            )

        elif auc_change < 0:

            print(
                "\nContext features reduced "
                "mean ROC-AUC relative to baseline."
            )

        else:

            print(
                "\nContext features did not change "
                "mean ROC-AUC relative to baseline."
            )

        if abs(auc_change) < 0.01:

            print(
                "The absolute ROC-AUC change is small; "
                "this should not automatically be treated "
                "as a meaningful improvement."
            )

    print(
        "\nThe final test period remains untouched."
    )

    print(
        "No production model was created."
    )

    # ========================================================
    # 13. SAVE
    # ========================================================

    print_section(
        "SAVING CONTEXT-AWARE EXPERIMENT RESULTS"
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

    if comparison_available:

        comparison_df.to_csv(
            COMPARISON_FILE,
            index=False,
        )

        print(
            f"\nFull feature comparison saved to:\n"
            f"{COMPARISON_FILE}"
        )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print_section(
        "CONTEXT-AWARE WALK-FORWARD VALIDATION COMPLETE"
    )

    print(
        f"Folds evaluated: "
        f"{len(FOLDS)}"
    )

    print(
        f"Base predictors: "
        f"{len(base_features)}"
    )

    print(
        f"Distribution predictors: "
        f"{len(distribution_features)}"
    )

    print(
        f"Context predictors: "
        f"{len(context_features)}"
    )

    print(
        f"Total predictors: "
        f"{len(features)}"
    )

    print(
        f"Mean validation ROC-AUC: "
        f"{context_mean_auc:.4f}"
    )

    print(
        f"Minimum validation ROC-AUC: "
        f"{context_min_auc:.4f}"
    )

    print(
        "\nJune-August 2018 final test period "
        "was not used."
    )

    print(
        "No final production model was created."
    )

    print(
        "\nThe purpose of this experiment was to determine "
        "whether the additional context feature layer "
        "improves temporal generalization."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()