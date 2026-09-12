"""
Temporal Target Reliability Audit

Purpose
-------
Investigate whether the future-risk targets are supported by
enough future observations to be considered reliable.

This audit uses two existing sources:

1. temporal_seller_dataset.csv
   - contains the original future outcome measurements

2. temporal_target_decomposition.csv
   - contains the already-validated decomposition into:
       future_delivery_risk
       future_review_risk

The audit examines:

1. Future delivery-rate denominators
2. Future review evidence
3. Future order volume
4. Risk prevalence by evidence size
5. Thin-evidence target observations
6. Delivery threshold proximity
7. Review threshold proximity
8. Target reliability by snapshot
9. Target reliability by train/validation/test period

This script does NOT train a model.

Inputs
------
data/processed/temporal_seller_dataset.csv
data/processed/temporal_target_decomposition.csv

Outputs
-------
data/processed/temporal_target_reliability.csv
data/processed/temporal_target_reliability_by_snapshot.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

TEMPORAL_DATASET_FILE = (
    DATA_DIR
    / "temporal_seller_dataset.csv"
)

DECOMPOSITION_FILE = (
    DATA_DIR
    / "temporal_target_decomposition.csv"
)

RELIABILITY_OUTPUT = (
    DATA_DIR
    / "temporal_target_reliability.csv"
)

SNAPSHOT_OUTPUT = (
    DATA_DIR
    / "temporal_target_reliability_by_snapshot.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

FUTURE_AT_RISK = "future_at_risk"

FUTURE_DELIVERY_RISK = (
    "future_delivery_risk"
)

FUTURE_REVIEW_RISK = (
    "future_review_risk"
)

FUTURE_LATE_RATE = (
    "future_late_delivery_rate"
)

FUTURE_REVIEW_SCORE = (
    "future_average_review_score"
)

FUTURE_ORDER_COUNT = (
    "future_order_count"
)

FUTURE_DELIVERED_ORDERS = (
    "future_delivered_orders"
)


# ============================================================
# BUSINESS THRESHOLDS
# ============================================================

DELIVERY_THRESHOLD = 20.0

REVIEW_THRESHOLD = 3.5


# ============================================================
# EVIDENCE BUCKETS
# ============================================================

def future_order_bucket(value):
    """
    Group future order counts into evidence-size buckets.
    """

    if pd.isna(value):
        return "missing"

    value = float(value)

    if value <= 1:
        return "1"

    if value <= 2:
        return "2"

    if value <= 5:
        return "3-5"

    if value <= 10:
        return "6-10"

    if value <= 20:
        return "11-20"

    if value <= 50:
        return "21-50"

    return "51+"


def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_binary_target(
    df,
    column,
    label,
):
    """
    Validate a binary target.
    """

    if column not in df.columns:

        raise ValueError(
            f"{label} column not found: {column}"
        )

    missing = (
        df[column]
        .isna()
        .sum()
    )

    invalid = (
        set(
            df[column]
            .dropna()
            .unique()
        )
        - {0, 1}
    )

    print(
        f"{label} missing values: "
        f"{missing}"
    )

    print(
        f"{label} invalid values: "
        f"{len(invalid)}"
    )

    if missing > 0:

        raise ValueError(
            f"{label} contains missing values."
        )

    if invalid:

        raise ValueError(
            f"{label} contains invalid values: "
            f"{invalid}"
        )


def validate_seller_snapshot_uniqueness(
    df,
    label,
):
    """
    Ensure one seller has at most one observation per snapshot.
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


def calculate_rate(
    df,
    target_column,
):
    """
    Calculate positive-class percentage.
    """

    if len(df) == 0:
        return np.nan

    return (
        df[target_column]
        .mean()
        * 100
    )


def validate_alignment(
    source_df,
    decomposition_df,
):
    """
    Confirm that the source and decomposition datasets represent
    the same seller/snapshot observations.
    """

    source_keys = set(
        zip(
            source_df[SELLER_COLUMN],
            source_df[DATE_COLUMN],
        )
    )

    decomposition_keys = set(
        zip(
            decomposition_df[SELLER_COLUMN],
            decomposition_df[DATE_COLUMN],
        )
    )

    missing_from_decomposition = (
        source_keys
        - decomposition_keys
    )

    missing_from_source = (
        decomposition_keys
        - source_keys
    )

    if missing_from_decomposition:

        raise ValueError(
            "Seller/snapshot observations exist in the "
            "temporal source but not in the decomposition."
            f"\nMissing decomposition records: "
            f"{len(missing_from_decomposition):,}"
        )

    if missing_from_source:

        raise ValueError(
            "Seller/snapshot observations exist in the "
            "decomposition but not in the temporal source."
            f"\nExtra decomposition records: "
            f"{len(missing_from_source):,}"
        )

    print(
        "Source/decomposition seller-snapshot alignment -> PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL TARGET RELIABILITY AUDIT")
    print("=" * 70)

    # ========================================================
    # 1. LOAD ORIGINAL TEMPORAL DATA
    # ========================================================

    print_section(
        "LOADING TEMPORAL SELLER DATASET"
    )

    if not TEMPORAL_DATASET_FILE.exists():

        raise FileNotFoundError(
            "Temporal seller dataset not found:\n"
            f"{TEMPORAL_DATASET_FILE}\n\n"
            "Run build_temporal_dataset.py first."
        )

    temporal_df = pd.read_csv(
        TEMPORAL_DATASET_FILE
    )

    print(
        f"Rows: {len(temporal_df):,}"
    )

    print(
        f"Columns: {len(temporal_df.columns):,}"
    )

    # ========================================================
    # 2. LOAD TARGET DECOMPOSITION
    # ========================================================

    print_section(
        "LOADING VALIDATED TARGET DECOMPOSITION"
    )

    if not DECOMPOSITION_FILE.exists():

        raise FileNotFoundError(
            "Target decomposition dataset not found:\n"
            f"{DECOMPOSITION_FILE}\n\n"
            "Run temporal_target_decomposition.py first."
        )

    decomposition_df = pd.read_csv(
        DECOMPOSITION_FILE
    )

    print(
        f"Rows: {len(decomposition_df):,}"
    )

    print(
        f"Columns: {len(decomposition_df.columns):,}"
    )

    # ========================================================
    # 3. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING TEMPORAL KEYS"
    )

    temporal_df[
        DATE_COLUMN
    ] = pd.to_datetime(
        temporal_df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    decomposition_df[
        DATE_COLUMN
    ] = pd.to_datetime(
        decomposition_df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    if temporal_df[
        DATE_COLUMN
    ].isna().any():

        raise ValueError(
            "Invalid snapshot dates in temporal dataset."
        )

    if decomposition_df[
        DATE_COLUMN
    ].isna().any():

        raise ValueError(
            "Invalid snapshot dates in target decomposition."
        )

    print(
        "Temporal date conversion -> PASS"
    )

    # ========================================================
    # 4. VALIDATE REQUIRED SOURCE COLUMNS
    # ========================================================

    print_section(
        "SOURCE COLUMN VALIDATION"
    )

    temporal_required = [
        SELLER_COLUMN,
        DATE_COLUMN,
        FUTURE_AT_RISK,
        FUTURE_LATE_RATE,
        FUTURE_REVIEW_SCORE,
        FUTURE_ORDER_COUNT,
        FUTURE_DELIVERED_ORDERS,
    ]

    temporal_missing = [
        column
        for column in temporal_required
        if column not in temporal_df.columns
    ]

    if temporal_missing:

        raise ValueError(
            "Temporal dataset is missing required columns:\n"
            + "\n".join(
                temporal_missing
            )
        )

    decomposition_required = [
        SELLER_COLUMN,
        DATE_COLUMN,
        FUTURE_DELIVERY_RISK,
        FUTURE_REVIEW_RISK,
    ]

    decomposition_missing = [
        column
        for column in decomposition_required
        if column not in decomposition_df.columns
    ]

    if decomposition_missing:

        raise ValueError(
            "Target decomposition is missing required columns:\n"
            + "\n".join(
                decomposition_missing
            )
        )

    print(
        "Temporal source columns -> PASS"
    )

    print(
        "Decomposition columns -> PASS"
    )

    # ========================================================
    # 5. VALIDATE SELLER / SNAPSHOT STRUCTURE
    # ========================================================

    print_section(
        "SELLER / SNAPSHOT VALIDATION"
    )

    validate_seller_snapshot_uniqueness(
        temporal_df,
        "Temporal dataset",
    )

    validate_seller_snapshot_uniqueness(
        decomposition_df,
        "Target decomposition",
    )

    # ========================================================
    # 6. ALIGN DATASETS
    # ========================================================

    print_section(
        "ALIGNING TARGET DATA"
    )

    validate_alignment(
        temporal_df,
        decomposition_df,
    )

    # ========================================================
    # 7. MERGE ORIGINAL OUTCOMES + DECOMPOSED TARGETS
    # ========================================================

    print_section(
        "MERGING TARGET RELIABILITY DATA"
    )

    decomposition_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        FUTURE_DELIVERY_RISK,
        FUTURE_REVIEW_RISK,
    ]

    target_df = decomposition_df[
        decomposition_columns
    ].copy()

    df = temporal_df.merge(
        target_df,
        on=[
            SELLER_COLUMN,
            DATE_COLUMN,
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged rows: {len(df):,}"
    )

    if len(df) != len(temporal_df):

        raise ValueError(
            "Merge reduced the number of observations."
        )

    print(
        "Target merge -> PASS"
    )

    # ========================================================
    # 8. TARGET VALIDATION
    # ========================================================

    print_section(
        "TARGET VALIDATION"
    )

    validate_binary_target(
        df,
        FUTURE_AT_RISK,
        "Future at-risk",
    )

    validate_binary_target(
        df,
        FUTURE_DELIVERY_RISK,
        "Future delivery-risk",
    )

    validate_binary_target(
        df,
        FUTURE_REVIEW_RISK,
        "Future review-risk",
    )

    print(
        "\nAll targets -> PASS"
    )

    # ========================================================
    # 9. NUMERIC OUTCOME VALIDATION
    # ========================================================

    print_section(
        "FUTURE OUTCOME VARIABLE VALIDATION"
    )

    outcome_columns = [
        FUTURE_LATE_RATE,
        FUTURE_REVIEW_SCORE,
        FUTURE_ORDER_COUNT,
        FUTURE_DELIVERED_ORDERS,
    ]

    for column in outcome_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        print(
            f"{column}: numeric conversion -> PASS"
        )

    # ========================================================
    # 10. ORIGINAL TARGET CONSISTENCY
    # ========================================================

    print_section(
        "TARGET CONSISTENCY CHECK"
    )

    reconstructed_target = (
        (
            df[
                FUTURE_DELIVERY_RISK
            ]
            == 1
        )
        |
        (
            df[
                FUTURE_REVIEW_RISK
            ]
            == 1
        )
    ).astype(int)

    mismatch_count = (
        reconstructed_target
        != df[
            FUTURE_AT_RISK
        ]
    ).sum()

    print(
        f"Reconstruction mismatches: "
        f"{mismatch_count}"
    )

    if mismatch_count > 0:

        raise ValueError(
            "The decomposed targets no longer reproduce "
            "future_at_risk exactly."
        )

    print(
        "Target reconstruction -> PASS"
    )

    # ========================================================
    # 11. FUTURE EVIDENCE OVERVIEW
    # ========================================================

    print_section(
        "FUTURE EVIDENCE OVERVIEW"
    )

    print(
        "Future orders:"
    )

    print(
        df[
            FUTURE_ORDER_COUNT
        ]
        .describe()
        .round(3)
    )

    print(
        "\nFuture delivered orders:"
    )

    print(
        df[
            FUTURE_DELIVERED_ORDERS
        ]
        .describe()
        .round(3)
    )

    print(
        "\nFuture late-delivery rate:"
    )

    print(
        df[
            FUTURE_LATE_RATE
        ]
        .describe()
        .round(3)
    )

    print(
        "\nFuture average review score:"
    )

    print(
        df[
            FUTURE_REVIEW_SCORE
        ]
        .describe()
        .round(3)
    )

    # ========================================================
    # 12. CREATE EVIDENCE BUCKETS
    # ========================================================

    print_section(
        "CREATING FUTURE EVIDENCE BUCKETS"
    )

    df[
        "future_order_bucket"
    ] = (
        df[
            FUTURE_ORDER_COUNT
        ]
        .apply(
            future_order_bucket
        )
    )

    print(
        "Future order evidence buckets -> PASS"
    )

    # ========================================================
    # 13. DELIVERY RELIABILITY BY EVIDENCE SIZE
    # ========================================================

    print_section(
        "DELIVERY TARGET RELIABILITY"
    )

    delivery_summary = (
        df
        .groupby(
            "future_order_bucket"
        )
        .agg(
            observations=(
                SELLER_COLUMN,
                "count",
            ),
            delivery_risk_count=(
                FUTURE_DELIVERY_RISK,
                "sum",
            ),
            delivery_risk_rate=(
                FUTURE_DELIVERY_RISK,
                "mean",
            ),
            average_future_orders=(
                FUTURE_ORDER_COUNT,
                "mean",
            ),
            average_future_delivered_orders=(
                FUTURE_DELIVERED_ORDERS,
                "mean",
            ),
            average_late_delivery_rate=(
                FUTURE_LATE_RATE,
                "mean",
            ),
        )
        .reset_index()
    )

    delivery_summary[
        "delivery_risk_rate"
    ] = (
        delivery_summary[
            "delivery_risk_rate"
        ]
        * 100
    )

    print(
        delivery_summary
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 14. REVIEW RELIABILITY BY EVIDENCE SIZE
    # ========================================================

    print_section(
        "REVIEW TARGET RELIABILITY"
    )

    review_summary = (
        df
        .groupby(
            "future_order_bucket"
        )
        .agg(
            observations=(
                SELLER_COLUMN,
                "count",
            ),
            review_risk_count=(
                FUTURE_REVIEW_RISK,
                "sum",
            ),
            review_risk_rate=(
                FUTURE_REVIEW_RISK,
                "mean",
            ),
            average_future_orders=(
                FUTURE_ORDER_COUNT,
                "mean",
            ),
            average_future_delivered_orders=(
                FUTURE_DELIVERED_ORDERS,
                "mean",
            ),
            average_future_review_score=(
                FUTURE_REVIEW_SCORE,
                "mean",
            ),
        )
        .reset_index()
    )

    review_summary[
        "review_risk_rate"
    ] = (
        review_summary[
            "review_risk_rate"
        ]
        * 100
    )

    print(
        review_summary
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 15. THIN EVIDENCE ANALYSIS
    # ========================================================

    print_section(
        "THIN-EVIDENCE ANALYSIS"
    )

    thin_evidence = (
        df[
            FUTURE_DELIVERED_ORDERS
        ]
        .fillna(0)
        <= 2
    )

    thin_count = (
        thin_evidence.sum()
    )

    thin_percentage = (
        thin_count
        / len(df)
        * 100
    )

    thin_delivery_risk = (
        (
            thin_evidence
            &
            (
                df[
                    FUTURE_DELIVERY_RISK
                ]
                == 1
            )
        )
        .sum()
    )

    thin_review_risk = (
        (
            thin_evidence
            &
            (
                df[
                    FUTURE_REVIEW_RISK
                ]
                == 1
            )
        )
        .sum()
    )

    print(
        f"Observations with <= 2 future delivered orders: "
        f"{thin_count:,}"
    )

    print(
        f"Share of all observations: "
        f"{thin_percentage:.2f}%"
    )

    print(
        f"Delivery-risk positives within thin evidence: "
        f"{thin_delivery_risk:,}"
    )

    print(
        f"Review-risk positives within thin evidence: "
        f"{thin_review_risk:,}"
    )

    # ========================================================
    # 16. DELIVERY THRESHOLD PROXIMITY
    # ========================================================

    print_section(
        "DELIVERY THRESHOLD PROXIMITY"
    )

    valid_delivery = df[
        FUTURE_LATE_RATE
    ].notna()

    delivery_valid_df = df[
        valid_delivery
    ].copy()

    delivery_regions = [
        (
            "Below 10%",
            delivery_valid_df[
                FUTURE_LATE_RATE
            ] < 10,
        ),
        (
            "10-14.99%",
            (
                delivery_valid_df[
                    FUTURE_LATE_RATE
                ] >= 10
            )
            &
            (
                delivery_valid_df[
                    FUTURE_LATE_RATE
                ] < 15
            ),
        ),
        (
            "15-19.99%",
            (
                delivery_valid_df[
                    FUTURE_LATE_RATE
                ] >= 15
            )
            &
            (
                delivery_valid_df[
                    FUTURE_LATE_RATE
                ] < DELIVERY_THRESHOLD
            ),
        ),
        (
            "20-24.99%",
            (
                delivery_valid_df[
                    FUTURE_LATE_RATE
                ] >= DELIVERY_THRESHOLD
            )
            &
            (
                delivery_valid_df[
                    FUTURE_LATE_RATE
                ] < 25
            ),
        ),
        (
            "25%+",
            delivery_valid_df[
                FUTURE_LATE_RATE
            ] >= 25,
        ),
    ]

    for label, mask in delivery_regions:

        region = delivery_valid_df[
            mask
        ]

        if region.empty:
            continue

        print(
            f"{label}: "
            f"{len(region):,} observations | "
            f"delivery-risk rate = "
            f"{calculate_rate(region, FUTURE_DELIVERY_RISK):.2f}%"
        )

    # ========================================================
    # 17. REVIEW THRESHOLD PROXIMITY
    # ========================================================

    print_section(
        "REVIEW THRESHOLD PROXIMITY"
    )

    valid_review = df[
        FUTURE_REVIEW_SCORE
    ].notna()

    review_valid_df = df[
        valid_review
    ].copy()

    review_regions = [
        (
            "< 3.0",
            review_valid_df[
                FUTURE_REVIEW_SCORE
            ] < 3.0,
        ),
        (
            "3.0-3.49",
            (
                review_valid_df[
                    FUTURE_REVIEW_SCORE
                ] >= 3.0
            )
            &
            (
                review_valid_df[
                    FUTURE_REVIEW_SCORE
                ] < REVIEW_THRESHOLD
            ),
        ),
        (
            "3.5-3.99",
            (
                review_valid_df[
                    FUTURE_REVIEW_SCORE
                ] >= REVIEW_THRESHOLD
            )
            &
            (
                review_valid_df[
                    FUTURE_REVIEW_SCORE
                ] < 4.0
            ),
        ),
        (
            "4.0-4.49",
            (
                review_valid_df[
                    FUTURE_REVIEW_SCORE
                ] >= 4.0
            )
            &
            (
                review_valid_df[
                    FUTURE_REVIEW_SCORE
                ] < 4.5
            ),
        ),
        (
            "4.5+",
            review_valid_df[
                FUTURE_REVIEW_SCORE
            ] >= 4.5,
        ),
    ]

    for label, mask in review_regions:

        region = review_valid_df[
            mask
        ]

        if region.empty:
            continue

        print(
            f"{label}: "
            f"{len(region):,} observations | "
            f"review-risk rate = "
            f"{calculate_rate(region, FUTURE_REVIEW_RISK):.2f}%"
        )

    # ========================================================
    # 18. TARGET RELIABILITY BY SNAPSHOT
    # ========================================================

    print_section(
        "TARGET RELIABILITY BY SNAPSHOT"
    )

    snapshot_summary = (
        df
        .groupby(DATE_COLUMN)
        .agg(
            observations=(
                SELLER_COLUMN,
                "count",
            ),
            future_delivered_orders=(
                FUTURE_DELIVERED_ORDERS,
                "sum",
            ),
            delivery_risk_count=(
                FUTURE_DELIVERY_RISK,
                "sum",
            ),
            review_risk_count=(
                FUTURE_REVIEW_RISK,
                "sum",
            ),
            future_at_risk_count=(
                FUTURE_AT_RISK,
                "sum",
            ),
            average_future_late_rate=(
                FUTURE_LATE_RATE,
                "mean",
            ),
            average_future_review_score=(
                FUTURE_REVIEW_SCORE,
                "mean",
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
        .round(3)
    )

    print(
        snapshot_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 19. TARGET RELIABILITY BY TEMPORAL PERIOD
    # ========================================================

    print_section(
        "TARGET RELIABILITY BY TEMPORAL PERIOD"
    )

    train_mask = (
        df[DATE_COLUMN]
        <= pd.Timestamp(
            "2018-02-01"
        )
    )

    validation_mask = (
        (
            df[DATE_COLUMN]
            >= pd.Timestamp(
                "2018-03-01"
            )
        )
        &
        (
            df[DATE_COLUMN]
            <= pd.Timestamp(
                "2018-05-01"
            )
        )
    )

    test_mask = (
        (
            df[DATE_COLUMN]
            >= pd.Timestamp(
                "2018-06-01"
            )
        )
        &
        (
            df[DATE_COLUMN]
            <= pd.Timestamp(
                "2018-08-01"
            )
        )
    )

    period_rows = []

    for period_name, mask in [
        ("train", train_mask),
        ("validation", validation_mask),
        ("test", test_mask),
    ]:

        period_df = df[
            mask
        ].copy()

        period_rows.append(
            {
                "period": period_name,
                "observations": len(period_df),
                "future_delivery_risk_rate": (
                    calculate_rate(
                        period_df,
                        FUTURE_DELIVERY_RISK,
                    )
                ),
                "future_review_risk_rate": (
                    calculate_rate(
                        period_df,
                        FUTURE_REVIEW_RISK,
                    )
                ),
                "future_at_risk_rate": (
                    calculate_rate(
                        period_df,
                        FUTURE_AT_RISK,
                    )
                ),
                "average_future_delivered_orders": (
                    period_df[
                        FUTURE_DELIVERED_ORDERS
                    ].mean()
                ),
            }
        )

    period_summary = pd.DataFrame(
        period_rows
    )

    print(
        period_summary
        .round(3)
        .to_string(index=False)
    )

    # ========================================================
    # 20. TARGET EVIDENCE FLAGS
    # ========================================================

    print_section(
        "CREATING EVIDENCE FLAGS"
    )

    df[
        "thin_future_evidence"
    ] = (
        df[
            FUTURE_DELIVERED_ORDERS
        ]
        .fillna(0)
        <= 2
    ).astype(int)

    df[
        "low_future_evidence"
    ] = (
        df[
            FUTURE_DELIVERED_ORDERS
        ]
        .fillna(0)
        <= 5
    ).astype(int)

    print(
        "Evidence flags -> PASS"
    )

    # ========================================================
    # 21. BUILD RELIABILITY OUTPUT
    # ========================================================

    print_section(
        "BUILDING RELIABILITY OUTPUT"
    )

    reliability_rows = []

    for _, row in delivery_summary.iterrows():

        reliability_rows.append(
            {
                "target": "future_delivery_risk",
                "evidence_bucket": row[
                    "future_order_bucket"
                ],
                "observations": row[
                    "observations"
                ],
                "positive_count": row[
                    "delivery_risk_count"
                ],
                "positive_rate_percent": row[
                    "delivery_risk_rate"
                ],
                "average_future_evidence": row[
                    "average_future_orders"
                ],
                "average_future_delivered_orders": row[
                    "average_future_delivered_orders"
                ],
                "average_outcome_value": row[
                    "average_late_delivery_rate"
                ],
            }
        )

    for _, row in review_summary.iterrows():

        reliability_rows.append(
            {
                "target": "future_review_risk",
                "evidence_bucket": row[
                    "future_order_bucket"
                ],
                "observations": row[
                    "observations"
                ],
                "positive_count": row[
                    "review_risk_count"
                ],
                "positive_rate_percent": row[
                    "review_risk_rate"
                ],
                "average_future_evidence": row[
                    "average_future_orders"
                ],
                "average_future_delivered_orders": row[
                    "average_future_delivered_orders"
                ],
                "average_outcome_value": row[
                    "average_future_review_score"
                ],
            }
        )

    reliability_df = pd.DataFrame(
        reliability_rows
    )

    reliability_df.to_csv(
        RELIABILITY_OUTPUT,
        index=False,
    )

    snapshot_summary.to_csv(
        SNAPSHOT_OUTPUT,
        index=False,
    )

    # ========================================================
    # 22. FINAL SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL TARGET RELIABILITY AUDIT COMPLETE"
    )

    print(
        f"Observations audited: "
        f"{len(df):,}"
    )

    print(
        f"Unique sellers: "
        f"{df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{df[DATE_COLUMN].nunique():,}"
    )

    print(
        f"\nDelivery-risk observations: "
        f"{df[FUTURE_DELIVERY_RISK].sum():,}"
    )

    print(
        f"Review-risk observations: "
        f"{df[FUTURE_REVIEW_RISK].sum():,}"
    )

    print(
        f"Combined future-at-risk observations: "
        f"{df[FUTURE_AT_RISK].sum():,}"
    )

    print(
        f"\nReliability table saved to:\n"
        f"{RELIABILITY_OUTPUT}"
    )

    print(
        f"\nSnapshot reliability saved to:\n"
        f"{SNAPSHOT_OUTPUT}"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "The purpose of this audit is to determine whether "
        "the future targets are supported by sufficient evidence."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()