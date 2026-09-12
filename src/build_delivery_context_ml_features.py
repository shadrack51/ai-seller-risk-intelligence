"""
Build Context-Aware Delivery ML Features

Purpose
-------
Combine three audited feature layers:

    1. Base target-specific delivery features
    2. Distribution-aware delivery features
    3. Contextual delivery features

Final prediction unit:

    seller_id + snapshot_date

Final target:

    future_delivery_risk

Important design rule
---------------------
The BASE delivery dataset is the authoritative source of
the target.

Distribution and context datasets are treated as feature
layers. They do not need to contain the target.

Expected feature counts:

    Base delivery predictors:
        39

    Distribution predictors:
        30

    Context predictors:
        36

    Total:
        105

No model is trained by this script.

Inputs
------
data/processed/target_delivery_features.csv
data/processed/delivery_distribution_features.csv
data/processed/delivery_context_features.csv

Output
------
data/processed/delivery_context_ml_features.csv

Lineage
-------
data/processed/delivery_context_ml_feature_lineage.csv
"""


from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

BASE_FILE = (
    DATA_DIR
    / "target_delivery_features.csv"
)

DISTRIBUTION_FILE = (
    DATA_DIR
    / "delivery_distribution_features.csv"
)

CONTEXT_FILE = (
    DATA_DIR
    / "delivery_context_features.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "delivery_context_ml_features.csv"
)

LINEAGE_FILE = (
    DATA_DIR
    / "delivery_context_ml_feature_lineage.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

TARGET_COLUMN = "future_delivery_risk"


# ============================================================
# FORBIDDEN PREDICTOR COLUMNS
# ============================================================

FORBIDDEN_COLUMNS = {
    SELLER_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,

    "future_at_risk",
    "future_order_count",
    "future_delivered_orders",
    "future_late_delivery_rate",
    "future_average_review_score",
    "future_delivery_risk",
    "future_review_risk",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_file(
    path,
    label,
):
    """Check that an input file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def prepare_dates(
    df,
    label,
):
    """Convert and validate snapshot dates."""

    if DATE_COLUMN not in df.columns:

        raise ValueError(
            f"{label} does not contain "
            f"{DATE_COLUMN}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    invalid = (
        df[DATE_COLUMN]
        .isna()
        .sum()
    )

    print(
        f"{label} invalid snapshot dates: "
        f"{invalid}"
    )

    if invalid > 0:

        raise ValueError(
            f"{label} contains invalid snapshot dates."
        )


def validate_grain(
    df,
    label,
):
    """
    Validate the prediction grain:

        seller_id + snapshot_date
    """

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

    print(
        f"{label} duplicate seller/snapshot rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"{label} contains duplicate "
            "seller/snapshot observations."
        )


def validate_base_target(
    df,
):
    """
    Validate the authoritative target in the BASE dataset.
    """

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Base dataset does not contain "
            f"{TARGET_COLUMN}"
        )

    missing = (
        df[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    invalid = (
        set(
            df[
                TARGET_COLUMN
            ]
            .dropna()
            .unique()
        )
        -
        {0, 1}
    )

    print(
        f"Base missing target values: "
        f"{missing}"
    )

    print(
        f"Base invalid target values: "
        f"{len(invalid)}"
    )

    if missing > 0:

        raise ValueError(
            "Base target contains missing values."
        )

    if invalid:

        raise ValueError(
            f"Base target contains invalid values: "
            f"{invalid}"
        )

    print(
        "Base target validation -> PASS"
    )


def calculate_keys(
    df,
):
    """Return seller/snapshot keys."""

    return set(
        zip(
            df[SELLER_COLUMN],
            df[DATE_COLUMN],
        )
    )


def identify_valid_features(
    df,
    existing_columns,
):
    """
    Identify new predictor columns while explicitly excluding:

    - identifiers
    - dates
    - targets
    - future outcomes
    - columns already present
    """

    features = []

    for column in df.columns:

        if column in existing_columns:
            continue

        if column in FORBIDDEN_COLUMNS:
            continue

        if column.startswith(
            "future_"
        ):
            continue

        features.append(
            column
        )

    return features


def validate_no_forbidden_predictors(
    features,
):
    """Ensure no forbidden columns enter the predictor set."""

    forbidden = [
        feature
        for feature in features
        if (
            feature in FORBIDDEN_COLUMNS
            or feature.startswith(
                "future_"
            )
        )
    ]

    if forbidden:

        raise ValueError(
            "Forbidden predictors detected:\n"
            +
            "\n".join(
                forbidden
            )
        )

    print(
        "Future/target leakage check -> PASS"
    )


def validate_numeric_features(
    df,
    features,
):
    """Ensure all predictors are numeric."""

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
            "Non-numeric predictors detected:\n"
            +
            "\n".join(
                non_numeric
            )
        )

    print(
        "All predictors numeric -> PASS"
    )


def build_lineage(
    base_features,
    distribution_features,
    context_features,
):
    """
    Create feature-lineage documentation.
    """

    rows = []

    for feature in base_features:

        rows.append(
            {
                "feature": feature,
                "source_layer":
                    "base delivery",
                "description":
                    (
                        "Original target-specific "
                        "delivery feature."
                    ),
            }
        )

    for feature in distribution_features:

        rows.append(
            {
                "feature": feature,
                "source_layer":
                    "distribution-aware delivery",
                "description":
                    (
                        "Historical delivery distribution, "
                        "variability, tail, or change feature."
                    ),
            }
        )

    for feature in context_features:

        rows.append(
            {
                "feature": feature,
                "source_layer":
                    "delivery context",
                "description":
                    (
                        "Historical shipping context, "
                        "destination complexity, workload, "
                        "or contextual change feature."
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILDING CONTEXT-AWARE DELIVERY ML DATASET"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK INPUT FILES
    # ========================================================

    print_section(
        "CHECKING INPUT DATASETS"
    )

    validate_file(
        BASE_FILE,
        "Base delivery feature dataset",
    )

    validate_file(
        DISTRIBUTION_FILE,
        "Distribution feature dataset",
    )

    validate_file(
        CONTEXT_FILE,
        "Context feature dataset",
    )

    print(
        "Base delivery features -> FOUND"
    )

    print(
        "Distribution features -> FOUND"
    )

    print(
        "Context features -> FOUND"
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print_section(
        "LOADING FEATURE LAYERS"
    )

    base_df = pd.read_csv(
        BASE_FILE
    )

    distribution_df = pd.read_csv(
        DISTRIBUTION_FILE
    )

    context_df = pd.read_csv(
        CONTEXT_FILE
    )

    print(
        f"Base rows: "
        f"{len(base_df):,}"
    )

    print(
        f"Base columns: "
        f"{len(base_df.columns):,}"
    )

    print(
        f"Distribution rows: "
        f"{len(distribution_df):,}"
    )

    print(
        f"Distribution columns: "
        f"{len(distribution_df.columns):,}"
    )

    print(
        f"Context rows: "
        f"{len(context_df):,}"
    )

    print(
        f"Context columns: "
        f"{len(context_df.columns):,}"
    )

    # ========================================================
    # 3. DATE PREPARATION
    # ========================================================

    print_section(
        "PREPARING SNAPSHOT DATES"
    )

    prepare_dates(
        base_df,
        "Base dataset",
    )

    prepare_dates(
        distribution_df,
        "Distribution dataset",
    )

    prepare_dates(
        context_df,
        "Context dataset",
    )

    print(
        "Snapshot dates -> PASS"
    )

    # ========================================================
    # 4. GRAIN VALIDATION
    # ========================================================

    print_section(
        "VALIDATING PREDICTION GRAIN"
    )

    validate_grain(
        base_df,
        "Base dataset",
    )

    validate_grain(
        distribution_df,
        "Distribution dataset",
    )

    validate_grain(
        context_df,
        "Context dataset",
    )

    print(
        "Prediction grain validation -> PASS"
    )

    # ========================================================
    # 5. AUTHORITATIVE TARGET
    # ========================================================

    print_section(
        "VALIDATING AUTHORITATIVE TARGET"
    )

    print(
        "Target source: Base delivery dataset"
    )

    validate_base_target(
        base_df
    )

    # ========================================================
    # 6. POPULATION ALIGNMENT
    # ========================================================

    print_section(
        "COMPARING FEATURE POPULATIONS"
    )

    base_keys = calculate_keys(
        base_df
    )

    distribution_keys = calculate_keys(
        distribution_df
    )

    context_keys = calculate_keys(
        context_df
    )

    print(
        f"Base seller/snapshot keys: "
        f"{len(base_keys):,}"
    )

    print(
        f"Distribution seller/snapshot keys: "
        f"{len(distribution_keys):,}"
    )

    print(
        f"Context seller/snapshot keys: "
        f"{len(context_keys):,}"
    )

    missing_distribution = (
        base_keys
        -
        distribution_keys
    )

    extra_distribution = (
        distribution_keys
        -
        base_keys
    )

    missing_context = (
        base_keys
        -
        context_keys
    )

    extra_context = (
        context_keys
        -
        base_keys
    )

    print(
        f"Base keys missing distribution layer: "
        f"{len(missing_distribution):,}"
    )

    print(
        f"Distribution keys outside base: "
        f"{len(extra_distribution):,}"
    )

    print(
        f"Base keys missing context layer: "
        f"{len(missing_context):,}"
    )

    print(
        f"Context keys outside base: "
        f"{len(extra_context):,}"
    )

    if missing_distribution:

        raise ValueError(
            "Distribution layer is missing base observations."
        )

    if extra_distribution:

        raise ValueError(
            "Distribution layer contains extra observations."
        )

    if missing_context:

        raise ValueError(
            "Context layer is missing base observations."
        )

    if extra_context:

        raise ValueError(
            "Context layer contains extra observations."
        )

    print(
        "All feature populations align -> PASS"
    )

    # ========================================================
    # 7. BASE FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING BASE DELIVERY FEATURES"
    )

    base_features = [
        column
        for column in base_df.columns
        if column not in {
            SELLER_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        }
        and not column.startswith(
            "future_"
        )
    ]

    print(
        f"Base predictors: "
        f"{len(base_features)}"
    )

    # ========================================================
    # 8. DISTRIBUTION FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING DISTRIBUTION FEATURES"
    )

    existing_columns = set(
        base_df.columns
    )

    distribution_features = (
        identify_valid_features(
            distribution_df,
            existing_columns,
        )
    )

    print(
        f"Distribution predictors: "
        f"{len(distribution_features)}"
    )

    for number, feature in enumerate(
        distribution_features,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 9. CONTEXT FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING CONTEXT FEATURES"
    )

    existing_columns = (
        set(
            base_df.columns
        )
        |
        set(
            distribution_features
        )
    )

    context_features = (
        identify_valid_features(
            context_df,
            existing_columns,
        )
    )

    print(
        f"Context predictors: "
        f"{len(context_features)}"
    )

    for number, feature in enumerate(
        context_features,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 10. EXPECTED FEATURE COUNTS
    # ========================================================

    print_section(
        "FEATURE BLOCK SUMMARY"
    )

    total_expected = (
        len(base_features)
        +
        len(distribution_features)
        +
        len(context_features)
    )

    print(
        f"Base features: "
        f"{len(base_features)}"
    )

    print(
        f"Distribution features: "
        f"{len(distribution_features)}"
    )

    print(
        f"Context features: "
        f"{len(context_features)}"
    )

    print(
        f"Total expected predictors: "
        f"{total_expected}"
    )

    # ========================================================
    # 11. CHECK BLOCK COLLISIONS
    # ========================================================

    print_section(
        "CHECKING FEATURE BLOCK COLLISIONS"
    )

    base_distribution_collision = (
        set(base_features)
        &
        set(distribution_features)
    )

    base_context_collision = (
        set(base_features)
        &
        set(context_features)
    )

    distribution_context_collision = (
        set(distribution_features)
        &
        set(context_features)
    )

    if base_distribution_collision:

        raise ValueError(
            "Base/distribution feature collisions:\n"
            +
            "\n".join(
                sorted(
                    base_distribution_collision
                )
            )
        )

    if base_context_collision:

        raise ValueError(
            "Base/context feature collisions:\n"
            +
            "\n".join(
                sorted(
                    base_context_collision
                )
            )
        )

    if distribution_context_collision:

        raise ValueError(
            "Distribution/context feature collisions:\n"
            +
            "\n".join(
                sorted(
                    distribution_context_collision
                )
            )
        )

    print(
        "Feature block collision check -> PASS"
    )

    # ========================================================
    # 12. MERGE DISTRIBUTION
    # ========================================================

    print_section(
        "MERGING DISTRIBUTION FEATURES"
    )

    distribution_layer = (
        distribution_df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
                *distribution_features,
            ]
        ]
        .copy()
    )

    combined_df = (
        base_df
        .merge(
            distribution_layer,
            on=[
                SELLER_COLUMN,
                DATE_COLUMN,
            ],
            how="left",
            validate="one_to_one",
        )
    )

    print(
        f"Rows after distribution merge: "
        f"{len(combined_df):,}"
    )

    if len(combined_df) != len(
        base_df
    ):

        raise ValueError(
            "Distribution merge changed "
            "the prediction population."
        )

    print(
        "Distribution merge -> PASS"
    )

    # ========================================================
    # 13. MERGE CONTEXT
    # ========================================================

    print_section(
        "MERGING CONTEXT FEATURES"
    )

    context_layer = (
        context_df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
                *context_features,
            ]
        ]
        .copy()
    )

    combined_df = (
        combined_df
        .merge(
            context_layer,
            on=[
                SELLER_COLUMN,
                DATE_COLUMN,
            ],
            how="left",
            validate="one_to_one",
        )
    )

    print(
        f"Rows after context merge: "
        f"{len(combined_df):,}"
    )

    if len(combined_df) != len(
        base_df
    ):

        raise ValueError(
            "Context merge changed "
            "the prediction population."
        )

    print(
        "Context merge -> PASS"
    )

    # ========================================================
    # 14. FINAL PREDICTOR SET
    # ========================================================

    print_section(
        "BUILDING FINAL PREDICTOR SET"
    )

    final_features = (
        base_features
        +
        distribution_features
        +
        context_features
    )

    print(
        f"Final predictors: "
        f"{len(final_features)}"
    )

    # ========================================================
    # 15. LEAKAGE CHECK
    # ========================================================

    print_section(
        "FINAL FUTURE INFORMATION CHECK"
    )

    validate_no_forbidden_predictors(
        final_features
    )

    # ========================================================
    # 16. NUMERIC CHECK
    # ========================================================

    print_section(
        "FINAL FEATURE TYPE VALIDATION"
    )

    validate_numeric_features(
        combined_df,
        final_features,
    )

    # ========================================================
    # 17. TARGET CHECK
    # ========================================================

    print_section(
        "FINAL TARGET VALIDATION"
    )

    validate_base_target(
        combined_df
    )

    print(
        "Authoritative target preserved -> PASS"
    )

    # ========================================================
    # 18. FINAL GRAIN
    # ========================================================

    print_section(
        "FINAL GRAIN VALIDATION"
    )

    validate_grain(
        combined_df,
        "Final dataset",
    )

    print(
        "Final seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 19. ROW CONSERVATION
    # ========================================================

    print_section(
        "ROW CONSERVATION"
    )

    print(
        f"Base rows: "
        f"{len(base_df):,}"
    )

    print(
        f"Final rows: "
        f"{len(combined_df):,}"
    )

    if len(base_df) != len(
        combined_df
    ):

        raise ValueError(
            "Final dataset row count differs "
            "from base population."
        )

    print(
        "Row conservation -> PASS"
    )

    # ========================================================
    # 20. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION"
    )

    target_counts = (
        combined_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    target_rates = (
        combined_df[
            TARGET_COLUMN
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)

    print(
        target_counts
    )

    print(
        "\nTarget percentages:"
    )

    print(
        target_rates
    )

    # ========================================================
    # 21. MISSING VALUES
    # ========================================================

    print_section(
        "FINAL MISSING VALUE ANALYSIS"
    )

    missing = (
        combined_df[
            final_features
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
            "No missing predictors."
        )

    else:

        print(
            missing.to_string()
        )

        print(
            "\nMissing values will be handled later "
            "by the ML training pipeline using "
            "training-data-only imputation."
        )

    # ========================================================
    # 22. FEATURE LINEAGE
    # ========================================================

    print_section(
        "BUILDING FEATURE LINEAGE"
    )

    lineage_df = build_lineage(
        base_features,
        distribution_features,
        context_features,
    )

    print(
        lineage_df
        .groupby(
            "source_layer"
        )
        .size()
        .reset_index(
            name="feature_count"
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 23. SAVE
    # ========================================================

    print_section(
        "SAVING CONTEXT-AWARE ML DATASET"
    )

    final_output_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        *final_features,
        TARGET_COLUMN,
    ]

    final_df = (
        combined_df[
            final_output_columns
        ]
        .sort_values(
            [
                DATE_COLUMN,
                SELLER_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Context-aware ML dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    lineage_df.to_csv(
        LINEAGE_FILE,
        index=False,
    )

    print(
        f"\nFeature lineage saved to:\n"
        f"{LINEAGE_FILE}"
    )

    # ========================================================
    # 24. FINAL SUMMARY
    # ========================================================

    print_section(
        "CONTEXT-AWARE DELIVERY ML DATASET COMPLETE"
    )

    print(
        f"Rows: "
        f"{len(final_df):,}"
    )

    print(
        f"Unique sellers: "
        f"{final_df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{final_df[DATE_COLUMN].nunique():,}"
    )

    print(
        f"Base predictors: "
        f"{len(base_features):,}"
    )

    print(
        f"Distribution predictors: "
        f"{len(distribution_features):,}"
    )

    print(
        f"Context predictors: "
        f"{len(context_features):,}"
    )

    print(
        f"Total predictors: "
        f"{len(final_features):,}"
    )

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print(
        "\nPopulation alignment -> PASS"
    )

    print(
        "Row conservation -> PASS"
    )

    print(
        "Target preservation -> PASS"
    )

    print(
        "Future information check -> PASS"
    )

    print(
        "Numeric feature validation -> PASS"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The context-aware dataset is ready for "
        "the controlled temporal experiment."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()