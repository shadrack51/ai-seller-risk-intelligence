"""
Temporal Train / Validation / Test Split

Purpose
-------
Create an out-of-time train/validation/test split for the
forward-looking seller risk prediction problem.

The split is based on snapshot_date rather than random rows.

Structure:
    Earlier snapshots  -> TRAIN
    Middle snapshots   -> VALIDATION
    Latest snapshots   -> TEST

This simulates the real-world use case:

    Learn from the past
          ↓
    tune on later data
          ↓
    evaluate on the future

Outputs
-------
data/processed/seller_temporal_train.csv
data/processed/seller_temporal_validation.csv
data/processed/seller_temporal_test.csv
data/processed/temporal_split_summary.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_ml_dataset.csv"
)

TRAIN_OUTPUT = (
    PROCESSED_DATA_DIR
    / "seller_temporal_train.csv"
)

VALIDATION_OUTPUT = (
    PROCESSED_DATA_DIR
    / "seller_temporal_validation.csv"
)

TEST_OUTPUT = (
    PROCESSED_DATA_DIR
    / "seller_temporal_test.csv"
)

SUMMARY_OUTPUT = (
    PROCESSED_DATA_DIR
    / "temporal_split_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "future_at_risk"

DATE_COLUMN = "snapshot_date"

# Approximate split proportions over TIME, not rows.
TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a consistent section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_split_sizes(snapshot_count):
    """
    Calculate the number of chronological snapshots allocated
    to train, validation, and test.

    The test set is guaranteed to contain at least one snapshot.
    """

    train_count = int(
        snapshot_count * TRAIN_RATIO
    )

    validation_count = int(
        snapshot_count * VALIDATION_RATIO
    )

    test_count = (
        snapshot_count
        - train_count
        - validation_count
    )

    # Guard against very small datasets.
    if train_count < 1:
        train_count = 1

    if validation_count < 1:
        validation_count = 1

    if test_count < 1:
        test_count = 1

    # If rounding caused the total to exceed the available
    # snapshots, reduce training first.
    while (
        train_count
        + validation_count
        + test_count
        > snapshot_count
    ):

        if train_count > 1:
            train_count -= 1

        elif validation_count > 1:
            validation_count -= 1

        else:
            test_count -= 1

    return (
        train_count,
        validation_count,
        test_count,
    )


def summarize_split(
    df,
    split_name,
):
    """
    Create a summary row for one temporal split.
    """

    target_rate = (
        df[TARGET_COLUMN]
        .mean()
        * 100
    )

    return {
        "split": split_name,
        "rows": len(df),
        "unique_sellers": (
            df["seller_id"].nunique()
        ),
        "snapshots": (
            df[DATE_COLUMN].nunique()
        ),
        "first_snapshot": (
            df[DATE_COLUMN].min()
        ),
        "last_snapshot": (
            df[DATE_COLUMN].max()
        ),
        "at_risk_count": (
            df[TARGET_COLUMN].sum()
        ),
        "at_risk_rate_percent": (
            target_rate
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING TEMPORAL ML DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Temporal ML dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_temporal_ml_dataset.py first."
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

    if DATE_COLUMN not in df.columns:

        raise ValueError(
            f"Missing date column: {DATE_COLUMN}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates detected."
        )

    print(
        "Snapshot date conversion -> PASS"
    )

    # ========================================================
    # 3. VALIDATE TARGET
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Missing target column: "
            f"{TARGET_COLUMN}"
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
            "Invalid target values found:\n"
            f"{invalid_targets}"
        )

    print(
        f"Target column: "
        f"{TARGET_COLUMN} -> PASS"
    )

    print(
        "Target values -> PASS"
    )

    # ========================================================
    # 4. GET CHRONOLOGICAL SNAPSHOTS
    # ========================================================

    print_section(
        "IDENTIFYING CHRONOLOGICAL SNAPSHOTS"
    )

    snapshots = sorted(
        df[DATE_COLUMN]
        .drop_duplicates()
        .tolist()
    )

    snapshot_count = len(
        snapshots
    )

    print(
        f"Total snapshots: "
        f"{snapshot_count}"
    )

    print(
        "\nSnapshots:"
    )

    for snapshot in snapshots:

        print(
            f" - {snapshot.date()}"
        )

    if snapshot_count < 6:

        raise ValueError(
            "At least six snapshots are recommended "
            "for temporal train/validation/test splitting."
        )

    # ========================================================
    # 5. CALCULATE SPLIT SIZES
    # ========================================================

    print_section(
        "CALCULATING TEMPORAL SPLIT"
    )

    (
        train_snapshot_count,
        validation_snapshot_count,
        test_snapshot_count,
    ) = calculate_split_sizes(
        snapshot_count
    )

    print(
        f"Training snapshots: "
        f"{train_snapshot_count}"
    )

    print(
        f"Validation snapshots: "
        f"{validation_snapshot_count}"
    )

    print(
        f"Testing snapshots: "
        f"{test_snapshot_count}"
    )

    print(
        f"\nRequested proportions:"
    )

    print(
        f"Training:     {TRAIN_RATIO:.0%}"
    )

    print(
        f"Validation:   {VALIDATION_RATIO:.0%}"
    )

    print(
        f"Testing:      {TEST_RATIO:.0%}"
    )

    # ========================================================
    # 6. DEFINE CHRONOLOGICAL BOUNDARIES
    # ========================================================

    train_snapshots = (
        snapshots[
            :train_snapshot_count
        ]
    )

    validation_start_index = (
        train_snapshot_count
    )

    validation_end_index = (
        validation_start_index
        + validation_snapshot_count
    )

    validation_snapshots = (
        snapshots[
            validation_start_index:
            validation_end_index
        ]
    )

    test_snapshots = (
        snapshots[
            validation_end_index:
        ]
    )

    # ========================================================
    # 7. DISPLAY SPLIT BOUNDARIES
    # ========================================================

    print_section(
        "TEMPORAL SPLIT BOUNDARIES"
    )

    print(
        "TRAIN:"
    )

    print(
        f"First: "
        f"{train_snapshots[0].date()}"
    )

    print(
        f"Last:  "
        f"{train_snapshots[-1].date()}"
    )

    print(
        "\nVALIDATION:"
    )

    print(
        f"First: "
        f"{validation_snapshots[0].date()}"
    )

    print(
        f"Last:  "
        f"{validation_snapshots[-1].date()}"
    )

    print(
        "\nTEST:"
    )

    print(
        f"First: "
        f"{test_snapshots[0].date()}"
    )

    print(
        f"Last:  "
        f"{test_snapshots[-1].date()}"
    )

    # ========================================================
    # 8. CREATE DATA SPLITS
    # ========================================================

    print_section(
        "CREATING TEMPORAL DATA SPLITS"
    )

    train_df = df[
        df[DATE_COLUMN].isin(
            train_snapshots
        )
    ].copy()

    validation_df = df[
        df[DATE_COLUMN].isin(
            validation_snapshots
        )
    ].copy()

    test_df = df[
        df[DATE_COLUMN].isin(
            test_snapshots
        )
    ].copy()

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Testing rows:    {len(test_df):,}"
    )

    # ========================================================
    # 9. CHRONOLOGICAL ORDER VALIDATION
    # ========================================================

    print_section(
        "CHRONOLOGICAL ORDER VALIDATION"
    )

    train_last = (
        train_df[DATE_COLUMN].max()
    )

    validation_first = (
        validation_df[DATE_COLUMN].min()
    )

    validation_last = (
        validation_df[DATE_COLUMN].max()
    )

    test_first = (
        test_df[DATE_COLUMN].min()
    )

    chronological_train_validation = (
        train_last < validation_first
    )

    chronological_validation_test = (
        validation_last < test_first
    )

    print(
        "Training ends before validation -> "
        f"{chronological_train_validation}"
    )

    print(
        "Validation ends before testing -> "
        f"{chronological_validation_test}"
    )

    if not (
        chronological_train_validation
        and chronological_validation_test
    ):

        raise ValueError(
            "Temporal ordering violation detected."
        )

    print(
        "\nChronological ordering -> PASS"
    )

    # ========================================================
    # 10. ROW CONSERVATION
    # ========================================================

    print_section(
        "ROW CONSERVATION CHECK"
    )

    combined_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Original rows: "
        f"{len(df):,}"
    )

    print(
        f"Split rows:    "
        f"{combined_rows:,}"
    )

    if combined_rows != len(df):

        raise ValueError(
            "Rows were lost or duplicated during splitting."
        )

    print(
        "Row conservation -> PASS"
    )

    # ========================================================
    # 11. SNAPSHOT CONSERVATION
    # ========================================================

    split_snapshot_count = (
        train_df[DATE_COLUMN].nunique()
        +
        validation_df[DATE_COLUMN].nunique()
        +
        test_df[DATE_COLUMN].nunique()
    )

    print(
        f"Original snapshots: "
        f"{snapshot_count}"
    )

    print(
        f"Split snapshots:    "
        f"{split_snapshot_count}"
    )

    if split_snapshot_count != snapshot_count:

        raise ValueError(
            "Snapshots were lost during splitting."
        )

    print(
        "Snapshot conservation -> PASS"
    )

    # ========================================================
    # 12. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION BY SPLIT"
    )

    print(
        "TRAIN target distribution:"
    )

    print(
        train_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nVALIDATION target distribution:"
    )

    print(
        validation_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTEST target distribution:"
    )

    print(
        test_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # 13. TARGET RATE COMPARISON
    # ========================================================

    print_section(
        "FUTURE-RISK RATE COMPARISON"
    )

    train_rate = (
        train_df[
            TARGET_COLUMN
        ]
        .mean()
        * 100
    )

    validation_rate = (
        validation_df[
            TARGET_COLUMN
        ]
        .mean()
        * 100
    )

    test_rate = (
        test_df[
            TARGET_COLUMN
        ]
        .mean()
        * 100
    )

    overall_rate = (
        df[
            TARGET_COLUMN
        ]
        .mean()
        * 100
    )

    print(
        f"Overall future-risk rate:      "
        f"{overall_rate:.2f}%"
    )

    print(
        f"Training future-risk rate:     "
        f"{train_rate:.2f}%"
    )

    print(
        f"Validation future-risk rate:   "
        f"{validation_rate:.2f}%"
    )

    print(
        f"Testing future-risk rate:      "
        f"{test_rate:.2f}%"
    )

    print(
        "\nDifferences from overall rate:"
    )

    print(
        f"Training:     "
        f"{train_rate - overall_rate:+.2f} pp"
    )

    print(
        f"Validation:   "
        f"{validation_rate - overall_rate:+.2f} pp"
    )

    print(
        f"Testing:      "
        f"{test_rate - overall_rate:+.2f} pp"
    )

    # ========================================================
    # 14. SELLER OVERLAP ANALYSIS
    # ========================================================

    print_section(
        "SELLER OVERLAP ANALYSIS"
    )

    train_sellers = set(
        train_df["seller_id"]
    )

    validation_sellers = set(
        validation_df["seller_id"]
    )

    test_sellers = set(
        test_df["seller_id"]
    )

    train_validation_overlap = (
        train_sellers
        & validation_sellers
    )

    validation_test_overlap = (
        validation_sellers
        & test_sellers
    )

    train_test_overlap = (
        train_sellers
        & test_sellers
    )

    print(
        f"Train sellers: "
        f"{len(train_sellers):,}"
    )

    print(
        f"Validation sellers: "
        f"{len(validation_sellers):,}"
    )

    print(
        f"Test sellers: "
        f"{len(test_sellers):,}"
    )

    print(
        f"\nTrain/validation seller overlap: "
        f"{len(train_validation_overlap):,}"
    )

    print(
        f"Validation/test seller overlap: "
        f"{len(validation_test_overlap):,}"
    )

    print(
        f"Train/test seller overlap: "
        f"{len(train_test_overlap):,}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Seller overlap is expected because our product is designed "
        "to monitor existing sellers over time."
    )

    print(
        "We therefore preserve seller history across periods."
    )

    # ========================================================
    # 15. BUILD SUMMARY TABLE
    # ========================================================

    print_section(
        "BUILDING SPLIT SUMMARY"
    )

    summary_rows = [
        summarize_split(
            train_df,
            "train"
        ),
        summarize_split(
            validation_df,
            "validation"
        ),
        summarize_split(
            test_df,
            "test"
        ),
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 16. SAVE SPLITS
    # ========================================================

    print_section(
        "SAVING TEMPORAL SPLITS"
    )

    train_df.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    test_df.to_csv(
        TEST_OUTPUT,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    print(
        f"Training dataset saved to:\n"
        f"{TRAIN_OUTPUT}"
    )

    print(
        f"\nValidation dataset saved to:\n"
        f"{VALIDATION_OUTPUT}"
    )

    print(
        f"\nTesting dataset saved to:\n"
        f"{TEST_OUTPUT}"
    )

    print(
        f"\nSplit summary saved to:\n"
        f"{SUMMARY_OUTPUT}"
    )

    # ========================================================
    # 17. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL TRAIN / VALIDATION / TEST SPLIT COMPLETE"
    )

    print(
        f"Train snapshots: "
        f"{train_df[DATE_COLUMN].nunique()}"
    )

    print(
        f"Validation snapshots: "
        f"{validation_df[DATE_COLUMN].nunique()}"
    )

    print(
        f"Test snapshots: "
        f"{test_df[DATE_COLUMN].nunique()}"
    )

    print(
        f"\nTrain rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    print(
        "\nThe test set contains only later time periods "
        "than the training and validation sets."
    )

    print(
        "This creates an out-of-time evaluation design."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()