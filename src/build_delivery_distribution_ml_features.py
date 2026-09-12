"""
Build Delivery Distribution-Aware ML Dataset

Purpose
-------
Combine:

    1. Base target-specific delivery features
    2. Distribution-aware historical delivery features

The resulting dataset is intended for a controlled
walk-forward experiment comparing:

    BASE FEATURES
        versus
    BASE + DISTRIBUTION FEATURES

Prediction grain
-----------------
seller_id + snapshot_date

Prediction target
------------------
future_delivery_risk

Critical rule
-------------
A column being new does NOT automatically make it a valid
predictor.

Future outcomes, targets, identifiers, and other forbidden
columns are explicitly excluded from the predictor set.

No model is trained by this script.

Inputs
-------
data/processed/target_delivery_features.csv
data/processed/delivery_distribution_features.csv

Output
------
data/processed/delivery_distribution_ml_features.csv

Lineage output
--------------
data/processed/delivery_distribution_ml_feature_lineage.csv
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

BASE_FEATURE_FILE = (
    DATA_DIR
    / "target_delivery_features.csv"
)

DISTRIBUTION_FEATURE_FILE = (
    DATA_DIR
    / "delivery_distribution_features.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "delivery_distribution_ml_features.csv"
)

LINEAGE_FILE = (
    DATA_DIR
    / "delivery_distribution_ml_feature_lineage.csv"
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

FORBIDDEN_FEATURES = {
    SELLER_COLUMN,
    DATE_COLUMN,

    # Current prediction target
    TARGET_COLUMN,

    # Future outcomes
    "future_at_risk",
    "future_order_count",
    "future_delivered_orders",
    "future_late_delivery_rate",
    "future_average_review_score",
    "future_delivery_risk",
    "future_review_risk",

    # Explicit future-prefixed variables
    "future_target",
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
    """Confirm that an input file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def validate_dates(
    df,
    label,
):
    """Convert and validate snapshot dates."""

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
        f"{label} invalid dates: "
        f"{invalid}"
    )

    if invalid > 0:

        raise ValueError(
            f"{label} contains invalid snapshot dates."
        )


def validate_keys(
    df,
    label,
):
    """
    Validate the prediction grain:

        seller_id + snapshot_date
    """

    duplicate_count = (
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
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise ValueError(
            f"{label} contains duplicate "
            "seller/snapshot observations."
        )


def validate_target(
    df,
):
    """Validate the delivery-risk target."""

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column not found: "
            f"{TARGET_COLUMN}"
        )

    missing = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    invalid = (
        set(
            df[TARGET_COLUMN]
            .dropna()
            .unique()
        )
        - {0, 1}
    )

    print(
        f"Missing target values: "
        f"{missing}"
    )

    print(
        f"Invalid target values: "
        f"{len(invalid)}"
    )

    if missing > 0:

        raise ValueError(
            "Missing delivery-risk target values detected."
        )

    if invalid:

        raise ValueError(
            f"Invalid delivery-risk target values: "
            f"{invalid}"
        )


def get_predictor_columns(
    df,
):
    """
    Return all valid predictor columns.

    Identifiers, dates, targets, and future outcomes are excluded.
    """

    predictors = []

    for column in df.columns:

        if column in FORBIDDEN_FEATURES:
            continue

        if column.startswith("future_"):
            continue

        predictors.append(
            column
        )

    return predictors


def validate_no_forbidden_predictors(
    predictors,
):
    """
    Ensure no forbidden identifier, target, or future column
    enters the predictor set.
    """

    forbidden_found = []

    for feature in predictors:

        if feature in FORBIDDEN_FEATURES:

            forbidden_found.append(
                feature
            )

        elif feature.startswith(
            "future_"
        ):

            forbidden_found.append(
                feature
            )

    if forbidden_found:

        raise ValueError(
            "Forbidden predictor columns detected:\n"
            + "\n".join(
                forbidden_found
            )
        )

    print(
        "Future/target leakage check -> PASS"
    )


def validate_numeric_features(
    df,
    features,
):
    """Ensure every predictor is numeric."""

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

    print(
        "All predictor features numeric -> PASS"
    )


def key_set(
    df,
):
    """Return seller/snapshot keys."""

    return set(
        zip(
            df[SELLER_COLUMN],
            df[DATE_COLUMN],
        )
    )


def build_lineage(
    base_features,
    distribution_features,
):
    """
    Build feature lineage for the final predictor set.
    """

    rows = []

    for feature in base_features:

        rows.append(
            {
                "feature": feature,
                "source_layer": (
                    "base target-specific delivery"
                ),
                "description": (
                    "Existing seller activity/history "
                    "or delivery behavior feature."
                ),
            }
        )

    for feature in distribution_features:

        rows.append(
            {
                "feature": feature,
                "source_layer": (
                    "distribution-aware delivery"
                ),
                "description": (
                    "Historical delivery distribution, "
                    "variability, tail, or distribution-change feature."
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
        "BUILDING DELIVERY DISTRIBUTION-AWARE ML FEATURES"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK INPUTS
    # ========================================================

    print_section(
        "CHECKING INPUT DATASETS"
    )

    validate_file(
        BASE_FEATURE_FILE,
        "Base delivery feature dataset",
    )

    validate_file(
        DISTRIBUTION_FEATURE_FILE,
        "Distribution feature dataset",
    )

    print(
        "Base delivery features -> FOUND"
    )

    print(
        "Distribution features -> FOUND"
    )

    # ========================================================
    # 2. LOAD
    # ========================================================

    print_section(
        "LOADING FEATURE LAYERS"
    )

    base_df = pd.read_csv(
        BASE_FEATURE_FILE
    )

    distribution_df = pd.read_csv(
        DISTRIBUTION_FEATURE_FILE
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

    # ========================================================
    # 3. DATES
    # ========================================================

    print_section(
        "PREPARING SNAPSHOT DATES"
    )

    validate_dates(
        base_df,
        "Base dataset",
    )

    validate_dates(
        distribution_df,
        "Distribution dataset",
    )

    print(
        "Snapshot date conversion -> PASS"
    )

    # ========================================================
    # 4. GRAIN
    # ========================================================

    print_section(
        "VALIDATING PREDICTION GRAIN"
    )

    validate_keys(
        base_df,
        "Base dataset",
    )

    validate_keys(
        distribution_df,
        "Distribution dataset",
    )

    print(
        "Base seller/snapshot grain -> PASS"
    )

    print(
        "Distribution seller/snapshot grain -> PASS"
    )

    # ========================================================
    # 5. TARGET
    # ========================================================

    print_section(
        "VALIDATING DELIVERY TARGET"
    )

    validate_target(
        base_df
    )

    print(
        "Delivery target -> PASS"
    )

    # ========================================================
    # 6. POPULATION ALIGNMENT
    # ========================================================

    print_section(
        "COMPARING FEATURE POPULATIONS"
    )

    base_keys = key_set(
        base_df
    )

    distribution_keys = key_set(
        distribution_df
    )

    print(
        f"Base seller/snapshot keys: "
        f"{len(base_keys):,}"
    )

    print(
        f"Distribution seller/snapshot keys: "
        f"{len(distribution_keys):,}"
    )

    missing_distribution_keys = (
        base_keys
        - distribution_keys
    )

    extra_distribution_keys = (
        distribution_keys
        - base_keys
    )

    print(
        f"Base keys missing distribution features: "
        f"{len(missing_distribution_keys):,}"
    )

    print(
        f"Distribution keys outside base population: "
        f"{len(extra_distribution_keys):,}"
    )

    if missing_distribution_keys:

        raise ValueError(
            "Some base observations have no matching "
            "distribution feature record."
        )

    if extra_distribution_keys:

        raise ValueError(
            "Distribution observations exist outside "
            "the base prediction population."
        )

    print(
        "Feature population alignment -> PASS"
    )

    # ========================================================
    # 7. IDENTIFY ONLY VALID DISTRIBUTION FEATURES
    # ========================================================

    print_section(
        "IDENTIFYING VALID DISTRIBUTION FEATURES"
    )

    base_columns = set(
        base_df.columns
    )

    candidate_distribution_columns = [
        column
        for column in distribution_df.columns
        if column
        not in {
            SELLER_COLUMN,
            DATE_COLUMN,
        }
    ]

    print(
        f"Distribution columns before filtering: "
        f"{len(candidate_distribution_columns)}"
    )

    forbidden_distribution_columns = [
        column
        for column in candidate_distribution_columns
        if (
            column in FORBIDDEN_FEATURES
            or column.startswith(
                "future_"
            )
        )
    ]

    print(
        f"Forbidden distribution columns excluded: "
        f"{len(forbidden_distribution_columns)}"
    )

    for column in forbidden_distribution_columns:

        print(
            f" - {column}"
        )

    distribution_only_features = [
        column
        for column in candidate_distribution_columns
        if (
            column not in base_columns
            and column
            not in FORBIDDEN_FEATURES
            and not column.startswith(
                "future_"
            )
        )
    ]

    print(
        f"\nValid new distribution predictors: "
        f"{len(distribution_only_features)}"
    )

    for number, feature in enumerate(
        distribution_only_features,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 8. VERIFY DISTRIBUTION LAYER
    # ========================================================

    print_section(
        "VALIDATING DISTRIBUTION FEATURE LAYER"
    )

    validate_no_forbidden_predictors(
        distribution_only_features
    )

    if TARGET_COLUMN in (
        distribution_only_features
    ):

        raise ValueError(
            "Delivery target entered the distribution "
            "predictor set."
        )

    print(
        "Target exclusion -> PASS"
    )

    # ========================================================
    # 9. CHECK NAME COLLISIONS
    # ========================================================

    print_section(
        "CHECKING FEATURE NAME COLLISIONS"
    )

    collision_columns = (
        set(
            distribution_only_features
        )
        & base_columns
    )

    if collision_columns:

        raise ValueError(
            "Unexpected feature collisions:\n"
            + "\n".join(
                sorted(
                    collision_columns
                )
            )
        )

    print(
        "Feature name collision check -> PASS"
    )

    # ========================================================
    # 10. BUILD DISTRIBUTION LAYER
    # ========================================================

    print_section(
        "BUILDING DISTRIBUTION FEATURE LAYER"
    )

    distribution_layer = (
        distribution_df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
                *distribution_only_features,
            ]
        ]
        .copy()
    )

    print(
        f"Distribution layer columns: "
        f"{len(distribution_layer.columns)}"
    )

    # ========================================================
    # 11. MERGE
    # ========================================================

    print_section(
        "MERGING FEATURE LAYERS"
    )

    final_df = (
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
        f"Rows after merge: "
        f"{len(final_df):,}"
    )

    if len(final_df) != len(
        base_df
    ):

        raise ValueError(
            "Feature merge changed the prediction population."
        )

    print(
        "One-to-one feature merge -> PASS"
    )

    # ========================================================
    # 12. BUILD FINAL PREDICTOR LIST
    # ========================================================

    print_section(
        "BUILDING FINAL PREDICTOR SET"
    )

    final_features = get_predictor_columns(
        final_df
    )

    print(
        f"Final predictors: "
        f"{len(final_features)}"
    )

    for number, feature in enumerate(
        final_features,
        start=1,
    ):

        print(
            f"{number:2d}. {feature}"
        )

    # ========================================================
    # 13. FINAL LEAKAGE CHECK
    # ========================================================

    print_section(
        "FINAL FUTURE INFORMATION CHECK"
    )

    validate_no_forbidden_predictors(
        final_features
    )

    # ========================================================
    # 14. NUMERIC VALIDATION
    # ========================================================

    print_section(
        "FEATURE TYPE VALIDATION"
    )

    validate_numeric_features(
        final_df,
        final_features,
    )

    # ========================================================
    # 15. FINAL TARGET CHECK
    # ========================================================

    print_section(
        "FINAL TARGET VALIDATION"
    )

    validate_target(
        final_df
    )

    print(
        "Delivery target -> PASS"
    )

    # ========================================================
    # 16. FINAL GRAIN CHECK
    # ========================================================

    print_section(
        "FINAL SELLER / SNAPSHOT VALIDATION"
    )

    validate_keys(
        final_df,
        "Final dataset",
    )

    print(
        "Final seller/snapshot uniqueness -> PASS"
    )

    # ========================================================
    # 17. ROW CONSERVATION
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
        f"{len(final_df):,}"
    )

    if len(base_df) != len(
        final_df
    ):

        raise ValueError(
            "Final row count does not match base row count."
        )

    print(
        "Row conservation -> PASS"
    )

    # ========================================================
    # 18. TARGET DISTRIBUTION
    # ========================================================

    print_section(
        "TARGET DISTRIBUTION"
    )

    target_counts = (
        final_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    target_percentages = (
        final_df[
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
        target_percentages
    )

    # ========================================================
    # 19. MISSING VALUES
    # ========================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing = (
        final_df[
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
            "No missing predictor values."
        )

    else:

        print(
            missing.to_string()
        )

        print(
            "\nThese values will be handled later "
            "inside the ML pipeline."
        )

    # ========================================================
    # 20. DISTRIBUTION FEATURE COVERAGE
    # ========================================================

    print_section(
        "DISTRIBUTION FEATURE COVERAGE"
    )

    coverage_rows = []

    for feature in distribution_only_features:

        non_missing = (
            final_df[
                feature
            ]
            .notna()
            .sum()
        )

        coverage_rows.append(
            {
                "feature": feature,
                "non_missing": non_missing,
                "missing": (
                    len(final_df)
                    - non_missing
                ),
                "coverage_percent": (
                    non_missing
                    / len(final_df)
                    * 100
                ),
            }
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    if not coverage_df.empty:

        print(
            coverage_df
            .sort_values(
                "coverage_percent"
            )
            .round(2)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 21. FEATURE LINEAGE
    # ========================================================

    print_section(
        "BUILDING FEATURE LINEAGE"
    )

    base_predictors = [
        feature
        for feature in final_features
        if feature not in distribution_only_features
    ]

    lineage_df = build_lineage(
        base_predictors,
        distribution_only_features,
    )

    lineage_df[
        "target"
    ] = TARGET_COLUMN

    print(
        lineage_df
        .groupby(
            [
                "source_layer",
                "target",
            ]
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
    # 22. FINAL PREVIEW
    # ========================================================

    print_section(
        "FINAL DATASET PREVIEW"
    )

    preview_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,

        "recent_average_delivery_days",
        "recent_average_delivery_delay_days",
        "recent_late_delivery_rate",

        "recent_distribution_delivery_median",
        "recent_distribution_delivery_std",
        "recent_distribution_delivery_p90",
        "recent_distribution_delivery_max",

        "delivery_median_change",
        "delivery_std_change",
        "delivery_p90_change",
        "delivery_max_change",

        TARGET_COLUMN,
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in final_df.columns
    ]

    print(
        final_df[
            preview_columns
        ]
        .head(10)
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 23. SAVE DATASET
    # ========================================================

    print_section(
        "SAVING DELIVERY DISTRIBUTION-AWARE ML DATASET"
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 24. SAVE LINEAGE
    # ========================================================

    lineage_df.to_csv(
        LINEAGE_FILE,
        index=False,
    )

    print(
        f"\nFeature lineage saved to:\n"
        f"{LINEAGE_FILE}"
    )

    # ========================================================
    # 25. FINAL SUMMARY
    # ========================================================

    print_section(
        "DELIVERY DISTRIBUTION-AWARE ML DATASET COMPLETE"
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
        f"{len(base_predictors):,}"
    )

    print(
        f"New distribution predictors: "
        f"{len(distribution_only_features):,}"
    )

    print(
        f"Final predictors: "
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
        "Target preservation -> PASS"
    )

    print(
        "Future leakage check -> PASS"
    )

    print(
        "Numeric feature validation -> PASS"
    )

    print(
        "No model was trained."
    )

    print(
        "\nThe dataset is ready for the controlled "
        "distribution-aware temporal experiment."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()