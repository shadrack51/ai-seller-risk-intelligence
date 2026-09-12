"""
Build Risk-Specific Temporal Train / Validation / Test Splits

Purpose
-------
Apply the SAME chronological split boundaries to the two
target-specific prediction problems:

    1. Future delivery-risk prediction
    2. Future review-risk prediction

Temporal boundaries:

    TRAIN:
        2017-01-01 through 2018-02-01

    VALIDATION:
        2018-03-01 through 2018-05-01

    TEST:
        2018-06-01 through 2018-08-01

The same dates are used for both targets so that model
comparisons are performed under identical temporal conditions.

Outputs
-------
data/processed/target_delivery_train.csv
data/processed/target_delivery_validation.csv
data/processed/target_delivery_test.csv

data/processed/target_review_train.csv
data/processed/target_review_validation.csv
data/processed/target_review_test.csv

data/processed/risk_specific_split_summary.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

DELIVERY_INPUT = (
    DATA_DIR
    / "target_delivery_features.csv"
)

REVIEW_INPUT = (
    DATA_DIR
    / "target_review_features.csv"
)


# ------------------------------------------------------------
# Delivery outputs
# ------------------------------------------------------------

DELIVERY_TRAIN_OUTPUT = (
    DATA_DIR
    / "target_delivery_train.csv"
)

DELIVERY_VALIDATION_OUTPUT = (
    DATA_DIR
    / "target_delivery_validation.csv"
)

DELIVERY_TEST_OUTPUT = (
    DATA_DIR
    / "target_delivery_test.csv"
)


# ------------------------------------------------------------
# Review outputs
# ------------------------------------------------------------

REVIEW_TRAIN_OUTPUT = (
    DATA_DIR
    / "target_review_train.csv"
)

REVIEW_VALIDATION_OUTPUT = (
    DATA_DIR
    / "target_review_validation.csv"
)

REVIEW_TEST_OUTPUT = (
    DATA_DIR
    / "target_review_test.csv"
)


# ------------------------------------------------------------
# Summary output
# ------------------------------------------------------------

SUMMARY_OUTPUT = (
    DATA_DIR
    / "risk_specific_split_summary.csv"
)


# ============================================================
# CORE CONFIGURATION
# ============================================================

DATE_COLUMN = "snapshot_date"

SELLER_COLUMN = "seller_id"

DELIVERY_TARGET = "future_delivery_risk"

REVIEW_TARGET = "future_review_risk"


# ============================================================
# TEMPORAL BOUNDARIES
# ============================================================

TRAIN_START = pd.Timestamp(
    "2017-01-01"
)

TRAIN_END = pd.Timestamp(
    "2018-02-01"
)

VALIDATION_START = pd.Timestamp(
    "2018-03-01"
)

VALIDATION_END = pd.Timestamp(
    "2018-05-01"
)

TEST_START = pd.Timestamp(
    "2018-06-01"
)

TEST_END = pd.Timestamp(
    "2018-08-01"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(
    df,
    target,
    label,
):
    """
    Validate one target-specific dataset.
    """

    required_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        target,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{label} dataset is missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    if df[DATE_COLUMN].isna().any():
        raise ValueError(
            f"{label} contains invalid snapshot dates."
        )

    missing_target = (
        df[target]
        .isna()
        .sum()
    )

    if missing_target > 0:
        raise ValueError(
            f"{label} contains "
            f"{missing_target} missing target values."
        )

    invalid_target = (
        set(
            df[target]
            .unique()
        )
        - {0, 1}
    )

    if invalid_target:
        raise ValueError(
            f"{label} contains invalid target values:\n"
            f"{invalid_target}"
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
            f"{label} contains duplicate "
            "seller/snapshot observations."
        )


def split_dataset(df):
    """
    Split dataset using fixed chronological boundaries.
    """

    train = df[
        (
            df[DATE_COLUMN]
            >= TRAIN_START
        )
        &
        (
            df[DATE_COLUMN]
            <= TRAIN_END
        )
    ].copy()

    validation = df[
        (
            df[DATE_COLUMN]
            >= VALIDATION_START
        )
        &
        (
            df[DATE_COLUMN]
            <= VALIDATION_END
        )
    ].copy()

    test = df[
        (
            df[DATE_COLUMN]
            >= TEST_START
        )
        &
        (
            df[DATE_COLUMN]
            <= TEST_END
        )
    ].copy()

    return (
        train,
        validation,
        test,
    )


def summarize_split(
    df,
    target,
    target_name,
    split_name,
):
    """
    Create one summary row for a target/split combination.
    """

    return {
        "target": target_name,
        "split": split_name,
        "rows": len(df),
        "unique_sellers": (
            df[SELLER_COLUMN]
            .nunique()
        ),
        "snapshots": (
            df[DATE_COLUMN]
            .nunique()
        ),
        "first_snapshot": (
            df[DATE_COLUMN]
            .min()
        ),
        "last_snapshot": (
            df[DATE_COLUMN]
            .max()
        ),
        "positive_count": (
            df[target]
            .sum()
        ),
        "positive_rate_percent": (
            df[target]
            .mean()
            * 100
        ),
    }


def validate_split_boundaries(
    train,
    validation,
    test,
    label,
):
    """
    Confirm strict chronological ordering.
    """

    train_last = (
        train[DATE_COLUMN]
        .max()
    )

    validation_first = (
        validation[DATE_COLUMN]
        .min()
    )

    validation_last = (
        validation[DATE_COLUMN]
        .max()
    )

    test_first = (
        test[DATE_COLUMN]
        .min()
    )

    if not (
        train_last
        < validation_first
    ):
        raise ValueError(
            f"{label}: training and validation "
            "periods overlap or are out of order."
        )

    if not (
        validation_last
        < test_first
    ):
        raise ValueError(
            f"{label}: validation and test "
            "periods overlap or are out of order."
        )


def validate_snapshot_coverage(
    original,
    train,
    validation,
    test,
    label,
):
    """
    Confirm that every original snapshot belongs to exactly
    one of the three temporal partitions.
    """

    original_snapshots = set(
        original[
            DATE_COLUMN
        ].unique()
    )

    train_snapshots = set(
        train[
            DATE_COLUMN
        ].unique()
    )

    validation_snapshots = set(
        validation[
            DATE_COLUMN
        ].unique()
    )

    test_snapshots = set(
        test[
            DATE_COLUMN
        ].unique()
    )

    all_split_snapshots = (
        train_snapshots
        | validation_snapshots
        | test_snapshots
    )

    if (
        all_split_snapshots
        != original_snapshots
    ):

        missing_snapshots = (
            original_snapshots
            - all_split_snapshots
        )

        extra_snapshots = (
            all_split_snapshots
            - original_snapshots
        )

        raise ValueError(
            f"{label}: snapshot coverage mismatch.\n"
            f"Missing: {missing_snapshots}\n"
            f"Extra: {extra_snapshots}"
        )

    if (
        train_snapshots
        & validation_snapshots
    ):
        raise ValueError(
            f"{label}: train/validation snapshot overlap."
        )

    if (
        validation_snapshots
        & test_snapshots
    ):
        raise ValueError(
            f"{label}: validation/test snapshot overlap."
        )

    if (
        train_snapshots
        & test_snapshots
    ):
        raise ValueError(
            f"{label}: train/test snapshot overlap."
        )


def compare_feature_columns(
    delivery_df,
    review_df,
):
    """
    Compare predictor columns between delivery and review
    datasets.
    """

    delivery_features = set(
        delivery_df.columns
    ) - {
        SELLER_COLUMN,
        DATE_COLUMN,
        DELIVERY_TARGET,
    }

    review_features = set(
        review_df.columns
    ) - {
        SELLER_COLUMN,
        DATE_COLUMN,
        REVIEW_TARGET,
    }

    common = (
        delivery_features
        & review_features
    )

    delivery_only = (
        delivery_features
        - review_features
    )

    review_only = (
        review_features
        - delivery_features
    )

    return (
        common,
        delivery_only,
        review_only,
    )


def print_target_summary(
    df,
    target,
    label,
):
    """
    Print target counts and percentages for one split.

    This function was missing from the previous version of
    the script. It is required by the main workflow.
    """

    counts = (
        df[target]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[target]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)

    print(
        f"{label} target counts:"
    )

    print(
        counts
    )

    print(
        f"\n{label} target percentages:"
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
        "BUILDING RISK-SPECIFIC TEMPORAL SPLITS"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD
    # ========================================================

    print_section(
        "LOADING TARGET-SPECIFIC DATASETS"
    )

    if not DELIVERY_INPUT.exists():
        raise FileNotFoundError(
            f"Delivery dataset not found:\n"
            f"{DELIVERY_INPUT}"
        )

    if not REVIEW_INPUT.exists():
        raise FileNotFoundError(
            f"Review dataset not found:\n"
            f"{REVIEW_INPUT}"
        )

    delivery_df = pd.read_csv(
        DELIVERY_INPUT
    )

    review_df = pd.read_csv(
        REVIEW_INPUT
    )

    print(
        f"Delivery rows: "
        f"{len(delivery_df):,}"
    )

    print(
        f"Review rows: "
        f"{len(review_df):,}"
    )

    # ========================================================
    # 2. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING SNAPSHOT DATES"
    )

    delivery_df[
        DATE_COLUMN
    ] = pd.to_datetime(
        delivery_df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    review_df[
        DATE_COLUMN
    ] = pd.to_datetime(
        review_df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    print(
        "Delivery dates converted -> PASS"
    )

    print(
        "Review dates converted -> PASS"
    )

    # ========================================================
    # 3. VALIDATE SOURCE DATA
    # ========================================================

    print_section(
        "SOURCE DATA VALIDATION"
    )

    validate_dataset(
        delivery_df,
        DELIVERY_TARGET,
        "Delivery",
    )

    validate_dataset(
        review_df,
        REVIEW_TARGET,
        "Review",
    )

    print(
        "Delivery source validation -> PASS"
    )

    print(
        "Review source validation -> PASS"
    )

    # ========================================================
    # 4. DISPLAY FIXED BOUNDARIES
    # ========================================================

    print_section(
        "TEMPORAL SPLIT BOUNDARIES"
    )

    print(
        f"Training:      "
        f"{TRAIN_START.date()} → "
        f"{TRAIN_END.date()}"
    )

    print(
        f"Validation:    "
        f"{VALIDATION_START.date()} → "
        f"{VALIDATION_END.date()}"
    )

    print(
        f"Testing:       "
        f"{TEST_START.date()} → "
        f"{TEST_END.date()}"
    )

    print(
        "\nThe SAME boundaries will be used for "
        "both prediction targets."
    )

    # ========================================================
    # 5. SPLIT DELIVERY DATA
    # ========================================================

    print_section(
        "SPLITTING DELIVERY-RISK DATA"
    )

    (
        delivery_train,
        delivery_validation,
        delivery_test,
    ) = split_dataset(
        delivery_df
    )

    print(
        f"Training rows:   "
        f"{len(delivery_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(delivery_validation):,}"
    )

    print(
        f"Testing rows:    "
        f"{len(delivery_test):,}"
    )

    # ========================================================
    # 6. SPLIT REVIEW DATA
    # ========================================================

    print_section(
        "SPLITTING REVIEW-RISK DATA"
    )

    (
        review_train,
        review_validation,
        review_test,
    ) = split_dataset(
        review_df
    )

    print(
        f"Training rows:   "
        f"{len(review_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(review_validation):,}"
    )

    print(
        f"Testing rows:    "
        f"{len(review_test):,}"
    )

    # ========================================================
    # 7. CHRONOLOGICAL VALIDATION
    # ========================================================

    print_section(
        "CHRONOLOGICAL VALIDATION"
    )

    validate_split_boundaries(
        delivery_train,
        delivery_validation,
        delivery_test,
        "Delivery",
    )

    validate_split_boundaries(
        review_train,
        review_validation,
        review_test,
        "Review",
    )

    print(
        "Delivery chronological ordering -> PASS"
    )

    print(
        "Review chronological ordering -> PASS"
    )

    # ========================================================
    # 8. SNAPSHOT COVERAGE
    # ========================================================

    print_section(
        "SNAPSHOT COVERAGE VALIDATION"
    )

    validate_snapshot_coverage(
        delivery_df,
        delivery_train,
        delivery_validation,
        delivery_test,
        "Delivery",
    )

    validate_snapshot_coverage(
        review_df,
        review_train,
        review_validation,
        review_test,
        "Review",
    )

    print(
        "Delivery snapshot coverage -> PASS"
    )

    print(
        "Review snapshot coverage -> PASS"
    )

    # ========================================================
    # 9. ROW CONSERVATION
    # ========================================================

    print_section(
        "ROW CONSERVATION"
    )

    delivery_split_rows = (
        len(delivery_train)
        + len(delivery_validation)
        + len(delivery_test)
    )

    review_split_rows = (
        len(review_train)
        + len(review_validation)
        + len(review_test)
    )

    print(
        f"Delivery original: "
        f"{len(delivery_df):,}"
    )

    print(
        f"Delivery split total: "
        f"{delivery_split_rows:,}"
    )

    print(
        f"Review original: "
        f"{len(review_df):,}"
    )

    print(
        f"Review split total: "
        f"{review_split_rows:,}"
    )

    if (
        delivery_split_rows
        != len(delivery_df)
    ):
        raise ValueError(
            "Delivery rows were lost or duplicated."
        )

    if (
        review_split_rows
        != len(review_df)
    ):
        raise ValueError(
            "Review rows were lost or duplicated."
        )

    print(
        "\nDelivery row conservation -> PASS"
    )

    print(
        "Review row conservation -> PASS"
    )

    # ========================================================
    # 10. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "DELIVERY TARGET DISTRIBUTION"
    )

    print_target_summary(
        delivery_train,
        DELIVERY_TARGET,
        "Delivery training",
    )

    print_target_summary(
        delivery_validation,
        DELIVERY_TARGET,
        "Delivery validation",
    )

    print_target_summary(
        delivery_test,
        DELIVERY_TARGET,
        "Delivery testing",
    )

    print_section(
        "REVIEW TARGET DISTRIBUTION"
    )

    print_target_summary(
        review_train,
        REVIEW_TARGET,
        "Review training",
    )

    print_target_summary(
        review_validation,
        REVIEW_TARGET,
        "Review validation",
    )

    print_target_summary(
        review_test,
        REVIEW_TARGET,
        "Review testing",
    )

    # ========================================================
    # 11. TARGET RATE SHIFT
    # ========================================================

    print_section(
        "TARGET RATE SHIFT ANALYSIS"
    )

    delivery_train_rate = (
        delivery_train[
            DELIVERY_TARGET
        ].mean()
        * 100
    )

    delivery_validation_rate = (
        delivery_validation[
            DELIVERY_TARGET
        ].mean()
        * 100
    )

    delivery_test_rate = (
        delivery_test[
            DELIVERY_TARGET
        ].mean()
        * 100
    )

    review_train_rate = (
        review_train[
            REVIEW_TARGET
        ].mean()
        * 100
    )

    review_validation_rate = (
        review_validation[
            REVIEW_TARGET
        ].mean()
        * 100
    )

    review_test_rate = (
        review_test[
            REVIEW_TARGET
        ].mean()
        * 100
    )

    print(
        "DELIVERY RISK:"
    )

    print(
        f"Training:     "
        f"{delivery_train_rate:.2f}%"
    )

    print(
        f"Validation:   "
        f"{delivery_validation_rate:.2f}%"
    )

    print(
        f"Testing:      "
        f"{delivery_test_rate:.2f}%"
    )

    print(
        "\nREVIEW RISK:"
    )

    print(
        f"Training:     "
        f"{review_train_rate:.2f}%"
    )

    print(
        f"Validation:   "
        f"{review_validation_rate:.2f}%"
    )

    print(
        f"Testing:      "
        f"{review_test_rate:.2f}%"
    )

    # ========================================================
    # 12. TARGET-SPECIFIC FEATURE COMPARISON
    # ========================================================

    print_section(
        "TARGET-SPECIFIC FEATURE COMPARISON"
    )

    (
        common_features,
        delivery_only_features,
        review_only_features,
    ) = compare_feature_columns(
        delivery_df,
        review_df,
    )

    print(
        f"Common predictors: "
        f"{len(common_features)}"
    )

    print(
        f"Delivery-specific predictors: "
        f"{len(delivery_only_features)}"
    )

    print(
        f"Review-specific predictors: "
        f"{len(review_only_features)}"
    )

    print(
        "\nThe two models intentionally do not "
        "have identical feature sets."
    )

    # ========================================================
    # 13. SELLER OVERLAP
    # ========================================================

    print_section(
        "SELLER OVERLAP BY SPLIT"
    )

    for split_name, delivery_split, review_split in [
        (
            "Training",
            delivery_train,
            review_train,
        ),
        (
            "Validation",
            delivery_validation,
            review_validation,
        ),
        (
            "Testing",
            delivery_test,
            review_test,
        ),
    ]:

        delivery_sellers = set(
            delivery_split[
                SELLER_COLUMN
            ]
        )

        review_sellers = set(
            review_split[
                SELLER_COLUMN
            ]
        )

        overlap = (
            delivery_sellers
            & review_sellers
        )

        print(
            f"{split_name}: "
            f"delivery sellers = "
            f"{len(delivery_sellers):,}, "
            f"review sellers = "
            f"{len(review_sellers):,}, "
            f"overlap = "
            f"{len(overlap):,}"
        )

    # ========================================================
    # 14. BUILD SUMMARY
    # ========================================================

    print_section(
        "BUILDING SPLIT SUMMARY"
    )

    summary_rows = [
        summarize_split(
            delivery_train,
            DELIVERY_TARGET,
            "delivery",
            "train",
        ),
        summarize_split(
            delivery_validation,
            DELIVERY_TARGET,
            "delivery",
            "validation",
        ),
        summarize_split(
            delivery_test,
            DELIVERY_TARGET,
            "delivery",
            "test",
        ),
        summarize_split(
            review_train,
            REVIEW_TARGET,
            "review",
            "train",
        ),
        summarize_split(
            review_validation,
            REVIEW_TARGET,
            "review",
            "validation",
        ),
        summarize_split(
            review_test,
            REVIEW_TARGET,
            "review",
            "test",
        ),
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df[
        "positive_rate_percent"
    ] = (
        summary_df[
            "positive_rate_percent"
        ]
        .round(2)
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 15. SAVE DELIVERY SPLITS
    # ========================================================

    print_section(
        "SAVING DELIVERY SPLITS"
    )

    delivery_train.to_csv(
        DELIVERY_TRAIN_OUTPUT,
        index=False,
    )

    delivery_validation.to_csv(
        DELIVERY_VALIDATION_OUTPUT,
        index=False,
    )

    delivery_test.to_csv(
        DELIVERY_TEST_OUTPUT,
        index=False,
    )

    print(
        f"Delivery train:\n"
        f"{DELIVERY_TRAIN_OUTPUT}"
    )

    print(
        f"\nDelivery validation:\n"
        f"{DELIVERY_VALIDATION_OUTPUT}"
    )

    print(
        f"\nDelivery test:\n"
        f"{DELIVERY_TEST_OUTPUT}"
    )

    # ========================================================
    # 16. SAVE REVIEW SPLITS
    # ========================================================

    print_section(
        "SAVING REVIEW SPLITS"
    )

    review_train.to_csv(
        REVIEW_TRAIN_OUTPUT,
        index=False,
    )

    review_validation.to_csv(
        REVIEW_VALIDATION_OUTPUT,
        index=False,
    )

    review_test.to_csv(
        REVIEW_TEST_OUTPUT,
        index=False,
    )

    print(
        f"Review train:\n"
        f"{REVIEW_TRAIN_OUTPUT}"
    )

    print(
        f"\nReview validation:\n"
        f"{REVIEW_VALIDATION_OUTPUT}"
    )

    print(
        f"\nReview test:\n"
        f"{REVIEW_TEST_OUTPUT}"
    )

    # ========================================================
    # 17. SAVE SUMMARY
    # ========================================================

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        f"\nRisk-specific split summary:\n"
        f"{SUMMARY_OUTPUT}"
    )

    # ========================================================
    # 18. FINAL SUMMARY
    # ========================================================

    print_section(
        "RISK-SPECIFIC TEMPORAL SPLIT COMPLETE"
    )

    print(
        "Delivery:"
    )

    print(
        f"  Train:      {len(delivery_train):,}"
    )

    print(
        f"  Validation: {len(delivery_validation):,}"
    )

    print(
        f"  Test:       {len(delivery_test):,}"
    )

    print(
        "\nReview:"
    )

    print(
        f"  Train:      {len(review_train):,}"
    )

    print(
        f"  Validation: {len(review_validation):,}"
    )

    print(
        f"  Test:       {len(review_test):,}"
    )

    print(
        "\nBoth targets use identical chronological "
        "train/validation/test boundaries."
    )

    print(
        "No models were trained."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()