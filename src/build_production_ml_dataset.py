"""
Build Production Machine Learning Dataset

Purpose:
    Build a cleaner machine-learning dataset using only the
    features approved by the feature-lineage audit.

This script does NOT train a model.

It:
1. Loads the seller prediction dataset.
2. Loads the audited production feature list.
3. Validates the feature lineage output.
4. Selects only approved production features.
5. Preserves the target variable.
6. Validates the resulting dataset.
7. Saves the production ML dataset.

This is the first step toward a cleaner production model.
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

PREDICTION_DATASET_PATH = (
    PROCESSED_DATA_DIR
    / "seller_prediction_dataset.csv"
)

PRODUCTION_FEATURE_LIST_PATH = (
    PROCESSED_DATA_DIR
    / "production_feature_list.csv"
)

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "seller_production_ml_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "at_risk"

EXPECTED_PRODUCTION_FEATURE_COUNT = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_production_features():
    """
    Load the feature list created by the feature-lineage audit.
    """

    if not PRODUCTION_FEATURE_LIST_PATH.exists():

        raise FileNotFoundError(
            "Production feature list not found:\n"
            f"{PRODUCTION_FEATURE_LIST_PATH}\n\n"
            "Run feature_lineage_audit.py first."
        )

    feature_df = pd.read_csv(
        PRODUCTION_FEATURE_LIST_PATH
    )

    if "feature" not in feature_df.columns:

        raise ValueError(
            "Production feature list must contain "
            "a 'feature' column."
        )

    features = (
        feature_df["feature"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(features) == 0:

        raise ValueError(
            "Production feature list is empty."
        )

    return features


def validate_feature_list(features):
    """
    Validate the audited production feature list.
    """

    duplicate_features = (
        pd.Series(features)
        .duplicated()
        .sum()
    )

    if duplicate_features > 0:

        raise ValueError(
            "Duplicate features found in production "
            "feature list."
        )

    if len(features) != EXPECTED_PRODUCTION_FEATURE_COUNT:

        raise ValueError(
            "Unexpected number of production features.\n"
            f"Expected: "
            f"{EXPECTED_PRODUCTION_FEATURE_COUNT}\n"
            f"Found: {len(features)}"
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING PRODUCTION MACHINE LEARNING DATASET")
    print("=" * 70)

    # ========================================================
    # 1. LOAD SOURCE DATA
    # ========================================================

    print("\nLoading seller prediction dataset...")

    if not PREDICTION_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Prediction dataset not found:\n"
            f"{PREDICTION_DATASET_PATH}"
        )

    df = pd.read_csv(
        PREDICTION_DATASET_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # ========================================================
    # 2. VALIDATE TARGET
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )

    print(
        f"Target column: "
        f"{TARGET_COLUMN} -> PASS"
    )

    missing_target = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    print(
        f"Missing target values: "
        f"{missing_target}"
    )

    if missing_target > 0:

        raise ValueError(
            "Target contains missing values."
        )

    print(
        "Target completeness -> PASS"
    )

    invalid_target_values = (
        set(
            df[TARGET_COLUMN]
            .dropna()
            .unique()
        )
        - {0, 1}
    )

    if invalid_target_values:

        raise ValueError(
            "Invalid target values found:\n"
            f"{invalid_target_values}"
        )

    print(
        "Target values -> PASS"
    )

    # ========================================================
    # 3. LOAD AUDITED PRODUCTION FEATURE LIST
    # ========================================================

    print_section(
        "LOADING AUDITED PRODUCTION FEATURES"
    )

    production_features = (
        load_production_features()
    )

    validate_feature_list(
        production_features
    )

    print(
        f"Production features loaded: "
        f"{len(production_features)}"
    )

    print(
        "\nApproved production features:"
    )

    for position, feature in enumerate(
        production_features,
        start=1
    ):

        print(
            f" {position:2d}. {feature}"
        )

    print(
        "\nProduction feature list validation -> PASS"
    )

    # ========================================================
    # 4. CHECK SOURCE DATA FEATURES
    # ========================================================

    print_section(
        "SOURCE FEATURE VALIDATION"
    )

    missing_features = [
        feature
        for feature in production_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Production features missing from source dataset:\n"
            + "\n".join(missing_features)
        )

    print(
        "All production features exist in source dataset -> PASS"
    )

    # ========================================================
    # 5. CHECK THAT KNOWN EXCLUDED FEATURES ARE ABSENT
    # ========================================================

    print_section(
        "EXCLUDED FEATURE CHECK"
    )

    excluded_features = [
        "catalog_risk",
        "seller_risk_score",
    ]

    excluded_present = [
        feature
        for feature in excluded_features
        if feature in production_features
    ]

    if excluded_present:

        raise ValueError(
            "Excluded features were found in the "
            "production feature list:\n"
            + "\n".join(excluded_present)
        )

    print(
        "Known potential leakage features excluded -> PASS"
    )

    # ========================================================
    # 6. BUILD PRODUCTION DATASET
    # ========================================================

    print_section(
        "BUILDING PRODUCTION ML DATASET"
    )

    production_df = df[
        production_features
        + [TARGET_COLUMN]
    ].copy()

    print(
        f"Production dataset rows: "
        f"{len(production_df):,}"
    )

    print(
        f"Predictor features: "
        f"{len(production_features)}"
    )

    print(
        f"Total columns including target: "
        f"{len(production_df.columns)}"
    )

    # ========================================================
    # 7. FEATURE TYPE VALIDATION
    # ========================================================

    print_section(
        "FEATURE TYPE VALIDATION"
    )

    non_numeric_features = (
        production_df[
            production_features
        ]
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric_features:

        raise ValueError(
            "Non-numeric production features found:\n"
            + "\n".join(non_numeric_features)
        )

    print(
        "All production features are numeric -> PASS"
    )

    # ========================================================
    # 8. MISSING VALUE ANALYSIS
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing_values = (
        production_df[
            production_features
        ]
        .isna()
        .sum()
    )

    missing_values = (
        missing_values[
            missing_values > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if missing_values.empty:

        print(
            "No missing values found."
        )

    else:

        print(
            "Missing values in production features:"
        )

        print(
            missing_values
        )

        print(
            "\nThese will be handled later by the "
            "training pipeline using training-data-only "
            "imputation."
        )

    # ========================================================
    # 9. DUPLICATE CHECK
    # ========================================================

    print_section(
        "DUPLICATE CHECK"
    )

    duplicate_rows = (
        production_df[
            production_features
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate predictor rows: "
        f"{duplicate_rows}"
    )

    # Duplicate feature combinations are not automatically
    # errors because multiple sellers can legitimately have
    # identical numerical profiles.

    print(
        "Duplicate predictor rows are reported for "
        "inspection, not automatically removed."
    )

    # ========================================================
    # 10. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION"
    )

    target_counts = (
        production_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    target_percentages = (
        production_df[
            TARGET_COLUMN
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(
        "Counts:"
    )

    print(
        target_counts
    )

    print(
        "\nPercentages:"
    )

    print(
        target_percentages
    )

    # ========================================================
    # 11. FINAL FEATURE LIST
    # ========================================================

    print_section(
        "FINAL PRODUCTION FEATURE SET"
    )

    for feature in production_features:

        print(
            f" - {feature}"
        )

    # ========================================================
    # 12. FINAL VALIDATION
    # ========================================================

    print_section(
        "FINAL PRODUCTION DATASET VALIDATION"
    )

    if (
        len(production_df)
        != len(df)
    ):

        raise ValueError(
            "Production dataset row count does not match "
            "source dataset."
        )

    print(
        "Row count consistency -> PASS"
    )

    if (
        list(
            production_df.columns
        )
        != production_features
        + [TARGET_COLUMN]
    ):

        raise ValueError(
            "Production dataset column order is incorrect."
        )

    print(
        "Column order consistency -> PASS"
    )

    leakage_columns_in_output = [
        feature
        for feature in excluded_features
        if feature in production_df.columns
    ]

    if leakage_columns_in_output:

        raise ValueError(
            "Excluded leakage features remain in "
            "production dataset:\n"
            + "\n".join(
                leakage_columns_in_output
            )
        )

    print(
        "Excluded-feature check -> PASS"
    )

    # ========================================================
    # 13. SAVE DATASET
    # ========================================================

    print_section(
        "SAVING PRODUCTION ML DATASET"
    )

    production_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Production ML dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print_section(
        "PRODUCTION DATASET COMPLETE"
    )

    print(
        f"Rows: {len(production_df):,}"
    )

    print(
        f"Predictor features: "
        f"{len(production_features)}"
    )

    print(
        "Target: at_risk"
    )

    print(
        "\nThis dataset is ready for the "
        "production-model training stage."
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()