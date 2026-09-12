"""
Build Temporal Machine Learning Dataset

Purpose:
    Convert the temporal seller dataset into a clean ML dataset
    for forward-looking seller risk prediction.

The prediction problem is:

    Historical seller behavior
                ↓
          predict future risk

The temporal dataset contains both historical features and
future outcome columns. This script ensures that future
information cannot enter the predictor matrix.

Outputs:
    data/processed/temporal_ml_dataset.csv
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
    / "temporal_seller_dataset.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_ml_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "future_at_risk"

# ------------------------------------------------------------
# Historical predictors
# ------------------------------------------------------------

HISTORICAL_FEATURES = [
    "total_orders",
    "total_revenue",
    "average_item_price",
    "average_freight",
    "units_sold",
    "delivered_orders",
    "average_delivery_days",
    "average_estimated_delivery_days",
    "average_delivery_delay_days",
    "review_count",
]


# ------------------------------------------------------------
# Identifier / temporal metadata
# ------------------------------------------------------------

IDENTIFIER_COLUMNS = [
    "seller_id",
]

TEMPORAL_METADATA_COLUMNS = [
    "snapshot_date",
    "observation_start",
    "prediction_start",
    "prediction_end",
]


# ------------------------------------------------------------
# Future outcome columns
# ------------------------------------------------------------

FUTURE_COLUMNS = [
    "future_order_count",
    "future_delivered_orders",
    "future_late_delivery_rate",
    "future_average_review_score",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_required_columns(df, columns, group_name):
    """
    Validate that all required columns exist.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {group_name} columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TEMPORAL MACHINE LEARNING DATASET")
    print("=" * 70)

    # ========================================================
    # 1. LOAD TEMPORAL DATASET
    # ========================================================

    print_section(
        "LOADING TEMPORAL DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Temporal dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run build_temporal_dataset.py first."
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

    print(
        "\nColumns:"
    )

    for column in df.columns:

        print(
            f" - {column}"
        )

    # ========================================================
    # 2. CONVERT TEMPORAL COLUMNS
    # ========================================================

    print_section(
        "PREPARING TEMPORAL METADATA"
    )

    for column in TEMPORAL_METADATA_COLUMNS:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    missing_dates = (
        df[
            TEMPORAL_METADATA_COLUMNS
        ]
        .isna()
        .sum()
    )

    print(
        "Missing temporal metadata:"
    )

    print(
        missing_dates
    )

    if missing_dates.sum() > 0:

        raise ValueError(
            "Missing temporal metadata detected."
        )

    print(
        "\nTemporal metadata conversion -> PASS"
    )

    # ========================================================
    # 3. VALIDATE IDENTIFIERS
    # ========================================================

    print_section(
        "IDENTIFIER VALIDATION"
    )

    validate_required_columns(
        df,
        IDENTIFIER_COLUMNS,
        "identifier"
    )

    duplicate_seller_snapshot_rows = (
        df[
            IDENTIFIER_COLUMNS
            + ["snapshot_date"]
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate seller/snapshot rows: "
        f"{duplicate_seller_snapshot_rows}"
    )

    if duplicate_seller_snapshot_rows > 0:

        raise ValueError(
            "Duplicate seller/snapshot records detected."
        )

    print(
        "Seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 4. VALIDATE HISTORICAL FEATURES
    # ========================================================

    print_section(
        "HISTORICAL FEATURE VALIDATION"
    )

    validate_required_columns(
        df,
        HISTORICAL_FEATURES,
        "historical feature"
    )

    print(
        f"Historical predictors: "
        f"{len(HISTORICAL_FEATURES)}"
    )

    print(
        "\nHistorical features:"
    )

    for feature in HISTORICAL_FEATURES:

        print(
            f" - {feature}"
        )

    print(
        "\nHistorical feature validation -> PASS"
    )

    # ========================================================
    # 5. VALIDATE FUTURE COLUMNS
    # ========================================================

    print_section(
        "FUTURE COLUMN VALIDATION"
    )

    validate_required_columns(
        df,
        FUTURE_COLUMNS,
        "future outcome"
    )

    validate_required_columns(
        df,
        [TARGET_COLUMN],
        "target"
    )

    print(
        "Future outcome columns -> PASS"
    )

    print(
        f"Target column: "
        f"{TARGET_COLUMN}"
    )

    # ========================================================
    # 6. TEMPORAL LEAKAGE CHECK
    # ========================================================

    print_section(
        "POINT-IN-TIME LEAKAGE CHECK"
    )

    # --------------------------------------------------------
    # Observation period must end on/before prediction cutoff
    # --------------------------------------------------------

    observation_check = (
        df["observation_start"]
        <
        df["snapshot_date"]
    ).all()

    observation_end_check = (
        df["snapshot_date"]
        >=
        df["observation_start"]
    ).all()

    # --------------------------------------------------------
    # Prediction period must begin AFTER cutoff
    # --------------------------------------------------------

    prediction_start_check = (
        df["prediction_start"]
        >
        df["snapshot_date"]
    ).all()

    # --------------------------------------------------------
    # Prediction period must end after it begins
    # --------------------------------------------------------

    prediction_end_check = (
        df["prediction_end"]
        >=
        df["prediction_start"]
    ).all()

    print(
        "Observation starts before snapshot: "
        f"{observation_check}"
    )

    print(
        "Observation ends at snapshot: "
        f"{observation_end_check}"
    )

    print(
        "Prediction starts after snapshot: "
        f"{prediction_start_check}"
    )

    print(
        "Prediction ends after prediction starts: "
        f"{prediction_end_check}"
    )

    if not all(
        [
            observation_check,
            observation_end_check,
            prediction_start_check,
            prediction_end_check,
        ]
    ):

        raise ValueError(
            "Point-in-time temporal leakage detected."
        )

    print(
        "\nPoint-in-time leakage check -> PASS"
    )

    # ========================================================
    # 7. FUTURE FEATURE EXCLUSION
    # ========================================================

    print_section(
        "EXCLUDING FUTURE INFORMATION"
    )

    print(
        "The following columns will NOT be ML predictors:"
    )

    for column in FUTURE_COLUMNS:

        print(
            f" - {column}"
        )

    print(
        f" - {TARGET_COLUMN}"
    )

    print(
        "\nReason:"
    )

    print(
        "These columns describe outcomes that occur "
        "after the prediction cutoff."
    )

    # ========================================================
    # 8. BUILD ML DATASET
    # ========================================================

    print_section(
        "BUILDING TEMPORAL ML DATASET"
    )

    # --------------------------------------------------------
    # Keep identifiers and temporal metadata for auditability.
    #
    # Keep historical features as predictors.
    #
    # Keep future_at_risk as target.
    # --------------------------------------------------------

    output_columns = (
        IDENTIFIER_COLUMNS
        + TEMPORAL_METADATA_COLUMNS
        + HISTORICAL_FEATURES
        + [TARGET_COLUMN]
    )

    temporal_ml_df = (
        df[
            output_columns
        ]
        .copy()
    )

    print(
        f"Rows: "
        f"{len(temporal_ml_df):,}"
    )

    print(
        f"Columns: "
        f"{len(temporal_ml_df.columns):,}"
    )

    # ========================================================
    # 9. FEATURE TYPE VALIDATION
    # ========================================================

    print_section(
        "FEATURE TYPE VALIDATION"
    )

    non_numeric_features = (
        temporal_ml_df[
            HISTORICAL_FEATURES
        ]
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric_features:

        raise ValueError(
            "Non-numeric historical predictors found:\n"
            + "\n".join(
                non_numeric_features
            )
        )

    print(
        "All historical predictors are numeric -> PASS"
    )

    # ========================================================
    # 10. MISSING VALUE ANALYSIS
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing_features = (
        temporal_ml_df[
            HISTORICAL_FEATURES
        ]
        .isna()
        .sum()
    )

    missing_features = (
        missing_features[
            missing_features > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if missing_features.empty:

        print(
            "No missing predictor values found."
        )

    else:

        print(
            "Missing historical predictor values:"
        )

        print(
            missing_features
        )

        print(
            "\nThese will be handled later inside the "
            "ML training pipeline."
        )

    # ========================================================
    # 11. TARGET VALIDATION
    # ========================================================

    print_section(
        "FUTURE TARGET VALIDATION"
    )

    invalid_target_values = (
        set(
            temporal_ml_df[
                TARGET_COLUMN
            ]
            .dropna()
            .unique()
        )
        - {0, 1}
    )

    if invalid_target_values:

        raise ValueError(
            "Invalid future target values found:\n"
            + str(invalid_target_values)
        )

    print(
        "Future target values -> PASS"
    )

    missing_target = (
        temporal_ml_df[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing future target values: "
        f"{missing_target}"
    )

    if missing_target > 0:

        raise ValueError(
            "Missing future target values detected."
        )

    print(
        "Future target completeness -> PASS"
    )

    print(
        "\nFuture target distribution:"
    )

    print(
        temporal_ml_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nFuture target percentages:"
    )

    print(
        (
            temporal_ml_df[
                TARGET_COLUMN
            ]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
            .round(2)
        )
    )

    # ========================================================
    # 12. DIRECT FUTURE-COLUMN LEAKAGE CHECK
    # ========================================================

    print_section(
        "DIRECT FUTURE-COLUMN LEAKAGE CHECK"
    )

    predictor_columns = (
        IDENTIFIER_COLUMNS
        + TEMPORAL_METADATA_COLUMNS
        + HISTORICAL_FEATURES
    )

    leakage_columns = [
        column
        for column in predictor_columns
        if column in FUTURE_COLUMNS
        or column == TARGET_COLUMN
    ]

    if leakage_columns:

        raise ValueError(
            "Future information entered the predictor dataset:\n"
            + "\n".join(leakage_columns)
        )

    print(
        "No future outcome columns are predictors -> PASS"
    )

    # ========================================================
    # 13. SNAPSHOT-LEVEL ANALYSIS
    # ========================================================

    print_section(
        "TEMPORAL TARGET BY SNAPSHOT"
    )

    snapshot_summary = (
        temporal_ml_df
        .groupby("snapshot_date")
        [TARGET_COLUMN]
        .agg(
            observations="count",
            at_risk="sum",
            at_risk_rate="mean"
        )
        .reset_index()
    )

    snapshot_summary[
        "at_risk_rate"
    ] = (
        snapshot_summary[
            "at_risk_rate"
        ]
        * 100
    ).round(2)

    print(
        snapshot_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 14. FINAL DATASET COLUMNS
    # ========================================================

    print_section(
        "FINAL TEMPORAL ML DATASET"
    )

    print(
        "Identifier columns:"
    )

    for column in IDENTIFIER_COLUMNS:

        print(
            f" - {column}"
        )

    print(
        "\nTemporal metadata:"
    )

    for column in TEMPORAL_METADATA_COLUMNS:

        print(
            f" - {column}"
        )

    print(
        "\nPredictor features:"
    )

    for feature in HISTORICAL_FEATURES:

        print(
            f" - {feature}"
        )

    print(
        f"\nTarget:"
    )

    print(
        f" - {TARGET_COLUMN}"
    )

    # ========================================================
    # 15. SAVE DATASET
    # ========================================================

    print_section(
        "SAVING TEMPORAL ML DATASET"
    )

    temporal_ml_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Temporal ML dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL ML DATASET COMPLETE"
    )

    print(
        f"Rows: "
        f"{len(temporal_ml_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{temporal_ml_df['seller_id'].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{temporal_ml_df['snapshot_date'].nunique():,}"
    )

    print(
        f"Predictor features: "
        f"{len(HISTORICAL_FEATURES)}"
    )

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print(
        "\nThe dataset contains historical predictors "
        "and a future outcome target."
    )

    print(
        "Future outcome columns have been excluded from "
        "the predictor set."
    )

    print(
        "\nReady for temporal train/validation/test splitting."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()