"""
Temporal Target Decomposition

Purpose
-------
Break the future_at_risk target into its underlying future
business outcomes:

    1. Future delivery risk
    2. Future review risk

The analysis also measures:

    - target consistency
    - delivery-risk-only observations
    - review-risk-only observations
    - observations with both risks
    - observations with neither risk
    - risk prevalence by snapshot
    - overlap between the two risk mechanisms

This script does NOT train a model.

Input:
    data/processed/temporal_seller_dataset.csv

Outputs:
    data/processed/temporal_target_decomposition.csv
    data/processed/temporal_target_by_snapshot.csv
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
    / "temporal_target_decomposition.csv"
)

SNAPSHOT_OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "temporal_target_decomposition_by_snapshot.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "future_at_risk"

DELIVERY_RATE_COLUMN = (
    "future_late_delivery_rate"
)

REVIEW_SCORE_COLUMN = (
    "future_average_review_score"
)

FUTURE_ORDER_COLUMN = (
    "future_order_count"
)

FUTURE_DELIVERED_COLUMN = (
    "future_delivered_orders"
)

DATE_COLUMN = "snapshot_date"

FUTURE_LATE_RATE_THRESHOLD = 20.0

FUTURE_LOW_REVIEW_THRESHOLD = 3.5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_percentage(
    count,
    total,
):
    """Calculate a percentage safely."""

    if total == 0:
        return 0.0

    return (
        count / total * 100
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL TARGET DECOMPOSITION")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print_section(
        "LOADING TEMPORAL DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Temporal seller dataset not found:\n"
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

    # ========================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ========================================================

    print_section(
        "COLUMN VALIDATION"
    )

    required_columns = [
        "seller_id",
        DATE_COLUMN,
        TARGET_COLUMN,
        DELIVERY_RATE_COLUMN,
        REVIEW_SCORE_COLUMN,
        FUTURE_ORDER_COLUMN,
        FUTURE_DELIVERED_COLUMN,
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

    print(
        "All required columns exist -> PASS"
    )

    # ========================================================
    # 3. PREPARE TYPES
    # ========================================================

    print_section(
        "PREPARING TARGET VARIABLES"
    )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    df[DELIVERY_RATE_COLUMN] = pd.to_numeric(
        df[DELIVERY_RATE_COLUMN],
        errors="coerce",
    )

    df[REVIEW_SCORE_COLUMN] = pd.to_numeric(
        df[REVIEW_SCORE_COLUMN],
        errors="coerce",
    )

    df[FUTURE_ORDER_COLUMN] = pd.to_numeric(
        df[FUTURE_ORDER_COLUMN],
        errors="coerce",
    )

    df[FUTURE_DELIVERED_COLUMN] = pd.to_numeric(
        df[FUTURE_DELIVERED_COLUMN],
        errors="coerce",
    )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Invalid snapshot dates detected."
        )

    print(
        "Date and numeric conversion -> PASS"
    )

    # ========================================================
    # 4. ORIGINAL TARGET VALIDATION
    # ========================================================

    print_section(
        "ORIGINAL TARGET VALIDATION"
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

    missing_target = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    print(
        f"Missing target values: "
        f"{missing_target}"
    )

    print(
        f"Invalid target values: "
        f"{len(invalid_target_values)}"
    )

    if missing_target > 0:

        raise ValueError(
            "Missing future_at_risk values detected."
        )

    print(
        "Original target validation -> PASS"
    )

    # ========================================================
    # 5. CREATE DELIVERY RISK
    # ========================================================

    print_section(
        "CREATING FUTURE DELIVERY RISK"
    )

    print(
        f"Late delivery threshold: "
        f"{FUTURE_LATE_RATE_THRESHOLD:.1f}%"
    )

    df[
        "future_delivery_risk"
    ] = (
        df[
            DELIVERY_RATE_COLUMN
        ]
        >= FUTURE_LATE_RATE_THRESHOLD
    ).astype(int)

    print(
        "Future delivery-risk target created -> PASS"
    )

    # ========================================================
    # 6. CREATE REVIEW RISK
    # ========================================================

    print_section(
        "CREATING FUTURE REVIEW RISK"
    )

    print(
        f"Low review-score threshold: "
        f"< {FUTURE_LOW_REVIEW_THRESHOLD:.1f}"
    )

    df[
        "future_review_risk"
    ] = (
        df[
            REVIEW_SCORE_COLUMN
        ]
        < FUTURE_LOW_REVIEW_THRESHOLD
    ).astype(int)

    print(
        "Future review-risk target created -> PASS"
    )

    # ========================================================
    # 7. RECONSTRUCT COMBINED TARGET
    # ========================================================

    print_section(
        "RECONSTRUCTING COMBINED FUTURE RISK"
    )

    df[
        "reconstructed_future_at_risk"
    ] = (
        (
            df[
                "future_delivery_risk"
            ]
            == 1
        )
        |
        (
            df[
                "future_review_risk"
            ]
            == 1
        )
    ).astype(int)

    print(
        "Combined target reconstructed using OR logic."
    )

    # ========================================================
    # 8. TARGET CONSISTENCY CHECK
    # ========================================================

    print_section(
        "TARGET CONSISTENCY CHECK"
    )

    target_mismatch = (
        df[
            TARGET_COLUMN
        ]
        != df[
            "reconstructed_future_at_risk"
        ]
    )

    mismatch_count = (
        target_mismatch.sum()
    )

    mismatch_percentage = (
        calculate_percentage(
            mismatch_count,
            len(df),
        )
    )

    print(
        f"Target mismatches: "
        f"{mismatch_count:,}"
    )

    print(
        f"Target mismatch percentage: "
        f"{mismatch_percentage:.4f}%"
    )

    if mismatch_count > 0:

        print(
            "\nWARNING:"
        )

        print(
            "The reconstructed target does not exactly "
            "match the original future_at_risk target."
        )

        print(
            "This needs investigation before replacing "
            "the original target."
        )

    else:

        print(
            "Original target and reconstructed target "
            "match exactly -> PASS"
        )

    # ========================================================
    # 9. BUILD OUTCOME CATEGORIES
    # ========================================================

    print_section(
        "BUILDING FUTURE RISK CATEGORIES"
    )

    conditions = [
        (
            (
                df["future_delivery_risk"]
                == 0
            )
            &
            (
                df["future_review_risk"]
                == 0
            )
        ),

        (
            (
                df["future_delivery_risk"]
                == 1
            )
            &
            (
                df["future_review_risk"]
                == 0
            )
        ),

        (
            (
                df["future_delivery_risk"]
                == 0
            )
            &
            (
                df["future_review_risk"]
                == 1
            )
        ),

        (
            (
                df["future_delivery_risk"]
                == 1
            )
            &
            (
                df["future_review_risk"]
                == 1
            )
        ),
    ]

    categories = [
        "NEITHER",
        "DELIVERY_ONLY",
        "REVIEW_ONLY",
        "BOTH",
    ]

    df[
        "future_risk_category"
    ] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )

    for condition, category in zip(
        conditions,
        categories,
    ):

        df.loc[
            condition,
            "future_risk_category"
        ] = category

    # ========================================================
    # 10. OVERALL DECOMPOSITION
    # ========================================================

    print_section(
        "OVERALL FUTURE RISK DECOMPOSITION"
    )

    category_counts = (
        df[
            "future_risk_category"
        ]
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    category_summary = pd.DataFrame(
        {
            "category": (
                category_counts.index
            ),
            "observations": (
                category_counts.values
            ),
        }
    )

    category_summary[
        "percentage"
    ] = (
        category_summary[
            "observations"
        ]
        / len(df)
        * 100
    )

    category_summary[
        "percentage"
    ] = (
        category_summary[
            "percentage"
        ]
        .round(2)
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 11. RISK MECHANISM COUNTS
    # ========================================================

    print_section(
        "RISK MECHANISM SUMMARY"
    )

    delivery_count = (
        df[
            "future_delivery_risk"
        ]
        .sum()
    )

    review_count = (
        df[
            "future_review_risk"
        ]
        .sum()
    )

    both_count = (
        (
            (
                df[
                    "future_delivery_risk"
                ]
                == 1
            )
            &
            (
                df[
                    "future_review_risk"
                ]
                == 1
            )
        )
        .sum()
    )

    total_at_risk = (
        df[
            TARGET_COLUMN
        ]
        .sum()
    )

    print(
        f"Future delivery-risk observations: "
        f"{delivery_count:,}"
    )

    print(
        f"Future review-risk observations: "
        f"{review_count:,}"
    )

    print(
        f"Both delivery and review risk: "
        f"{both_count:,}"
    )

    print(
        f"Combined future-at-risk observations: "
        f"{total_at_risk:,}"
    )

    # ========================================================
    # 12. CONDITIONAL RISK COMPOSITION
    # ========================================================

    print_section(
        "COMPOSITION OF FUTURE-AT-RISK OBSERVATIONS"
    )

    at_risk_df = df[
        df[
            TARGET_COLUMN
        ]
        == 1
    ].copy()

    delivery_only_count = (
        (
            (
                at_risk_df[
                    "future_delivery_risk"
                ]
                == 1
            )
            &
            (
                at_risk_df[
                    "future_review_risk"
                ]
                == 0
            )
        )
        .sum()
    )

    review_only_count = (
        (
            (
                at_risk_df[
                    "future_delivery_risk"
                ]
                == 0
            )
            &
            (
                at_risk_df[
                    "future_review_risk"
                ]
                == 1
            )
        )
        .sum()
    )

    both_at_risk_count = (
        (
            (
                at_risk_df[
                    "future_delivery_risk"
                ]
                == 1
            )
            &
            (
                at_risk_df[
                    "future_review_risk"
                ]
                == 1
            )
        )
        .sum()
    )

    composition = pd.DataFrame(
        {
            "risk_mechanism": [
                "Delivery only",
                "Review only",
                "Both",
            ],
            "observations": [
                delivery_only_count,
                review_only_count,
                both_at_risk_count,
            ],
        }
    )

    composition[
        "percentage_of_at_risk"
    ] = (
        composition[
            "observations"
        ]
        / len(at_risk_df)
        * 100
    ).round(2)

    print(
        composition.to_string(
            index=False
        )
    )

    # ========================================================
    # 13. TARGET DECOMPOSITION BY SNAPSHOT
    # ========================================================

    print_section(
        "TARGET DECOMPOSITION BY SNAPSHOT"
    )

    snapshot_summary = (
        df
        .groupby(DATE_COLUMN)
        .agg(
            observations=(
                "seller_id",
                "count",
            ),
            delivery_risk_count=(
                "future_delivery_risk",
                "sum",
            ),
            review_risk_count=(
                "future_review_risk",
                "sum",
            ),
            both_count=(
                "future_risk_category",
                lambda x: (
                    x == "BOTH"
                ).sum()
            ),
            future_at_risk_count=(
                TARGET_COLUMN,
                "sum",
            ),
        )
        .reset_index()
    )

    snapshot_summary[
        "delivery_risk_rate"
    ] = (
        snapshot_summary[
            "delivery_risk_count"
        ]
        / snapshot_summary[
            "observations"
        ]
        * 100
    )

    snapshot_summary[
        "review_risk_rate"
    ] = (
        snapshot_summary[
            "review_risk_count"
        ]
        / snapshot_summary[
            "observations"
        ]
        * 100
    )

    snapshot_summary[
        "future_at_risk_rate"
    ] = (
        snapshot_summary[
            "future_at_risk_count"
        ]
        / snapshot_summary[
            "observations"
        ]
        * 100
    )

    snapshot_summary = (
        snapshot_summary
        .round(2)
    )

    print(
        snapshot_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 14. FUTURE OUTCOME QUALITY
    # ========================================================

    print_section(
        "FUTURE OUTCOME DATA QUALITY"
    )

    print(
        f"Future orders missing: "
        f"{df[FUTURE_ORDER_COLUMN].isna().sum()}"
    )

    print(
        f"Future delivered orders missing: "
        f"{df[FUTURE_DELIVERED_COLUMN].isna().sum()}"
    )

    print(
        f"Future late-delivery-rate missing: "
        f"{df[DELIVERY_RATE_COLUMN].isna().sum()}"
    )

    print(
        f"Future review-score missing: "
        f"{df[REVIEW_SCORE_COLUMN].isna().sum()}"
    )

    # ========================================================
    # 15. SAVE DECOMPOSITION
    # ========================================================

    print_section(
        "SAVING TARGET DECOMPOSITION"
    )

    output_columns = [
        "seller_id",
        DATE_COLUMN,
        FUTURE_ORDER_COLUMN,
        FUTURE_DELIVERED_COLUMN,
        DELIVERY_RATE_COLUMN,
        REVIEW_SCORE_COLUMN,
        "future_delivery_risk",
        "future_review_risk",
        "reconstructed_future_at_risk",
        TARGET_COLUMN,
        "future_risk_category",
    ]

    decomposition_df = df[
        output_columns
    ].copy()

    decomposition_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    snapshot_summary.to_csv(
        SNAPSHOT_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Target decomposition saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nSnapshot decomposition saved to:\n"
        f"{SNAPSHOT_OUTPUT_FILE}"
    )

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print_section(
        "TARGET DECOMPOSITION COMPLETE"
    )

    print(
        f"Total temporal observations: "
        f"{len(df):,}"
    )

    print(
        f"Future delivery-risk observations: "
        f"{delivery_count:,}"
    )

    print(
        f"Future review-risk observations: "
        f"{review_count:,}"
    )

    print(
        f"Both mechanisms: "
        f"{both_count:,}"
    )

    print(
        f"Future-at-risk observations: "
        f"{total_at_risk:,}"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The purpose of this analysis is to understand "
        "what actually makes up the future-risk target."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()