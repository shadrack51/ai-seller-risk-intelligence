"""
Build Final Seller Risk Dataset

Purpose
-------
Create the application-ready seller intelligence dataset.

This integrates:

    1. Existing seller-level risk scores
    2. Delivery-risk temporal model outputs
    3. Review-risk temporal model outputs

Important coverage rule
-----------------------
Temporal model coverage is not the same as risk.

A missing temporal score means:

    NOT_AVAILABLE

It does NOT mean:

    low risk
    zero risk

The application-ready dataset therefore explicitly records
model coverage.

Temporal scoring date
---------------------
Delivery and review models must use the same latest snapshot.

That date is stored once as:

    score_snapshot_date

No model is trained here.

Inputs
------
data/processed/seller_risk_scores.csv
data/processed/delivery_risk_model.joblib
data/processed/review_risk_model.joblib
data/processed/target_delivery_features.csv
data/processed/target_review_features.csv

Output
------
data/processed/final_seller_risk_dataset.csv
"""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


SELLER_RISK_FILE = (
    DATA_DIR
    / "seller_risk_scores.csv"
)

DELIVERY_MODEL_FILE = (
    DATA_DIR
    / "delivery_risk_model.joblib"
)

REVIEW_MODEL_FILE = (
    DATA_DIR
    / "review_risk_model.joblib"
)

DELIVERY_FEATURE_FILE = (
    DATA_DIR
    / "target_delivery_features.csv"
)

REVIEW_FEATURE_FILE = (
    DATA_DIR
    / "target_review_features.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "final_seller_risk_dataset.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"

SCORE_DATE_COLUMN = "score_snapshot_date"


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
    """Ensure a required file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def validate_seller_identity(
    df,
    label,
):
    """Validate unique seller IDs."""

    if SELLER_COLUMN not in df.columns:

        raise ValueError(
            f"{label} does not contain "
            f"{SELLER_COLUMN}"
        )

    duplicates = (
        df[
            SELLER_COLUMN
        ]
        .duplicated()
        .sum()
    )

    print(
        f"{label} duplicate seller IDs: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"{label} contains duplicate seller IDs."
        )


def validate_model_package(
    package,
    label,
):
    """Validate the structure of a saved model package."""

    required_fields = [
        "model",
        "features",
        "threshold",
    ]

    missing = [
        field
        for field
        in required_fields
        if field not in package
    ]

    if missing:

        raise ValueError(
            f"{label} model package is missing:\n"
            +
            "\n".join(
                missing
            )
        )


def score_latest_snapshot(
    model_package,
    feature_df,
    label,
):
    """
    Generate scores for the latest available snapshot.

    The result contains one row per seller.

    The date is renamed to score_snapshot_date so that
    delivery and review outputs can be safely combined.
    """

    validate_model_package(
        model_package,
        label,
    )

    model = (
        model_package[
            "model"
        ]
    )

    features = (
        model_package[
            "features"
        ]
    )

    threshold = float(
        model_package[
            "threshold"
        ]
    )

    latest_date = (
        feature_df[
            DATE_COLUMN
        ]
        .max()
    )

    latest_df = (
        feature_df[
            feature_df[
                DATE_COLUMN
            ]
            ==
            latest_date
        ]
        .copy()
    )

    missing_features = [
        feature
        for feature
        in features
        if feature not in latest_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"{label} feature dataset missing:\n"
            +
            "\n".join(
                missing_features
            )
        )

    X = (
        latest_df[
            features
        ]
        .copy()
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    decisions = (
        probabilities
        >= threshold
    ).astype(int)

    result = latest_df[
        [
            SELLER_COLUMN
        ]
    ].copy()

    result[
        SCORE_DATE_COLUMN
    ] = latest_date

    result[
        f"{label.lower()}_risk_probability"
    ] = probabilities

    result[
        f"{label.lower()}_risk_threshold"
    ] = threshold

    result[
        f"{label.lower()}_risk_decision"
    ] = decisions

    result[
        f"{label.lower()}_model_coverage"
    ] = "AVAILABLE"

    return result


def assign_business_priority(
    row,
):
    """
    Assign business priority using model outputs.

    NOT_SCORED means neither temporal model is available.

    It must never be interpreted as LOW risk.
    """

    delivery_available = (
        row[
            "delivery_model_coverage"
        ]
        == "AVAILABLE"
    )

    review_available = (
        row[
            "review_model_coverage"
        ]
        == "AVAILABLE"
    )

    delivery_flag = row[
        "delivery_risk_decision"
    ]

    review_flag = row[
        "review_risk_decision"
    ]

    delivery_probability = row[
        "delivery_risk_probability"
    ]

    review_probability = row[
        "review_risk_probability"
    ]

    # ========================================================
    # NO TEMPORAL MODEL AVAILABLE
    # ========================================================

    if (
        not delivery_available
        and
        not review_available
    ):

        return "NOT_SCORED"

    # ========================================================
    # BOTH MODELS AVAILABLE
    # ========================================================

    if (
        delivery_available
        and
        review_available
    ):

        if (
            delivery_flag == 1
            and
            review_flag == 1
        ):

            return "CRITICAL"

        if (
            delivery_flag == 1
            or
            review_flag == 1
        ):

            return "HIGH"

        if (
            delivery_probability >= 0.10
            or
            review_probability >= 0.10
        ):

            return "MONITOR"

        return "LOW"

    # ========================================================
    # DELIVERY ONLY
    # ========================================================

    if delivery_available:

        if delivery_flag == 1:

            return "HIGH"

        if (
            delivery_probability
            >= 0.10
        ):

            return "MONITOR"

        return "LOW"

    # ========================================================
    # REVIEW ONLY
    # ========================================================

    if review_available:

        if review_flag == 1:

            return "HIGH"

        if (
            review_probability
            >= 0.10
        ):

            return "MONITOR"

        return "LOW"

    return "NOT_SCORED"


def assign_business_action(
    priority,
):
    """Map priority to an operational action."""

    mapping = {
        "CRITICAL":
            "Immediate review",

        "HIGH":
            "Prioritize intervention",

        "MONITOR":
            "Monitor closely",

        "LOW":
            "Routine monitoring",

        "NOT_SCORED":
            "Insufficient temporal model coverage",
    }

    return mapping[
        priority
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILDING FINAL SELLER RISK DATASET"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK INPUTS
    # ========================================================

    print_section(
        "CHECKING REQUIRED INPUTS"
    )

    required_files = [
        (
            SELLER_RISK_FILE,
            "Seller risk score dataset",
        ),
        (
            DELIVERY_MODEL_FILE,
            "Delivery risk model",
        ),
        (
            REVIEW_MODEL_FILE,
            "Review risk model",
        ),
        (
            DELIVERY_FEATURE_FILE,
            "Delivery feature dataset",
        ),
        (
            REVIEW_FEATURE_FILE,
            "Review feature dataset",
        ),
    ]

    for path, label in required_files:

        validate_file(
            path,
            label,
        )

        print(
            f"{label} -> FOUND"
        )

    # ========================================================
    # 2. LOAD MASTER SELLER DATA
    # ========================================================

    print_section(
        "LOADING SELLER RISK DATA"
    )

    seller_risk_df = pd.read_csv(
        SELLER_RISK_FILE
    )

    print(
        f"Seller risk rows: "
        f"{len(seller_risk_df):,}"
    )

    print(
        f"Seller risk columns: "
        f"{len(seller_risk_df.columns):,}"
    )

    validate_seller_identity(
        seller_risk_df,
        "Seller risk dataset",
    )

    # ========================================================
    # 3. LOAD TEMPORAL FEATURE DATA
    # ========================================================

    print_section(
        "LOADING TEMPORAL FEATURE DATA"
    )

    delivery_df = pd.read_csv(
        DELIVERY_FEATURE_FILE
    )

    review_df = pd.read_csv(
        REVIEW_FEATURE_FILE
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
    # 4. PREPARE DATES
    # ========================================================

    print_section(
        "PREPARING TEMPORAL SNAPSHOT DATES"
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

    invalid_delivery_dates = (
        delivery_df[
            DATE_COLUMN
        ]
        .isna()
        .sum()
    )

    invalid_review_dates = (
        review_df[
            DATE_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"Delivery invalid dates: "
        f"{invalid_delivery_dates}"
    )

    print(
        f"Review invalid dates: "
        f"{invalid_review_dates}"
    )

    if invalid_delivery_dates > 0:

        raise ValueError(
            "Invalid delivery snapshot dates."
        )

    if invalid_review_dates > 0:

        raise ValueError(
            "Invalid review snapshot dates."
        )

    print(
        "Temporal dates -> PASS"
    )

    # ========================================================
    # 5. DETERMINE COMMON LATEST SNAPSHOT
    # ========================================================

    print_section(
        "DETERMINING COMMON SCORING SNAPSHOT"
    )

    latest_delivery_date = (
        delivery_df[
            DATE_COLUMN
        ]
        .max()
    )

    latest_review_date = (
        review_df[
            DATE_COLUMN
        ]
        .max()
    )

    print(
        f"Latest delivery snapshot: "
        f"{latest_delivery_date.date()}"
    )

    print(
        f"Latest review snapshot: "
        f"{latest_review_date.date()}"
    )

    if (
        latest_delivery_date
        !=
        latest_review_date
    ):

        raise ValueError(
            "Delivery and review latest snapshots differ."
        )

    score_snapshot_date = (
        latest_delivery_date
    )

    print(
        f"Common scoring snapshot: "
        f"{score_snapshot_date.date()}"
    )

    print(
        "Common scoring snapshot -> PASS"
    )

    # ========================================================
    # 6. LOAD MODELS
    # ========================================================

    print_section(
        "LOADING TARGET-SPECIFIC MODELS"
    )

    delivery_package = joblib.load(
        DELIVERY_MODEL_FILE
    )

    review_package = joblib.load(
        REVIEW_MODEL_FILE
    )

    validate_model_package(
        delivery_package,
        "Delivery",
    )

    validate_model_package(
        review_package,
        "Review",
    )

    print(
        "Delivery risk model loaded -> PASS"
    )

    print(
        "Review risk model loaded -> PASS"
    )

    print(
        f"Delivery threshold: "
        f"{delivery_package['threshold']:.2f}"
    )

    print(
        f"Review threshold: "
        f"{review_package['threshold']:.2f}"
    )

    # ========================================================
    # 7. GENERATE DELIVERY SCORES
    # ========================================================

    print_section(
        "GENERATING LATEST DELIVERY RISK SCORES"
    )

    delivery_scores = (
        score_latest_snapshot(
            delivery_package,
            delivery_df,
            "Delivery",
        )
    )

    print(
        f"Delivery sellers with "
        f"temporal scores: "
        f"{len(delivery_scores):,}"
    )

    # ========================================================
    # 8. GENERATE REVIEW SCORES
    # ========================================================

    print_section(
        "GENERATING LATEST REVIEW RISK SCORES"
    )

    review_scores = (
        score_latest_snapshot(
            review_package,
            review_df,
            "Review",
        )
    )

    print(
        f"Review sellers with "
        f"temporal scores: "
        f"{len(review_scores):,}"
    )

    # ========================================================
    # 9. SCORE DATE VALIDATION
    # ========================================================

    print_section(
        "VALIDATING SCORE SNAPSHOTS"
    )

    delivery_score_dates = (
        delivery_scores[
            SCORE_DATE_COLUMN
        ]
        .unique()
    )

    review_score_dates = (
        review_scores[
            SCORE_DATE_COLUMN
        ]
        .unique()
    )

    if len(
        delivery_score_dates
    ) != 1:

        raise ValueError(
            "Delivery scores contain multiple scoring dates."
        )

    if len(
        review_score_dates
    ) != 1:

        raise ValueError(
            "Review scores contain multiple scoring dates."
        )

    if (
        delivery_score_dates[0]
        !=
        review_score_dates[0]
    ):

        raise ValueError(
            "Delivery and review score dates differ."
        )

    print(
        f"Delivery score date: "
        f"{delivery_score_dates[0].date()}"
    )

    print(
        f"Review score date: "
        f"{review_score_dates[0].date()}"
    )

    print(
        "Score-date alignment -> PASS"
    )

    # ========================================================
    # 10. MASTER POPULATION
    # ========================================================

    print_section(
        "VALIDATING SELLER POPULATION"
    )

    all_sellers = set(
        seller_risk_df[
            SELLER_COLUMN
        ]
    )

    delivery_sellers = set(
        delivery_scores[
            SELLER_COLUMN
        ]
    )

    review_sellers = set(
        review_scores[
            SELLER_COLUMN
        ]
    )

    print(
        f"Total sellers: "
        f"{len(all_sellers):,}"
    )

    print(
        f"Delivery model coverage: "
        f"{len(delivery_sellers):,}"
    )

    print(
        f"Review model coverage: "
        f"{len(review_sellers):,}"
    )

    print(
        f"Both temporal models available: "
        f"{len(delivery_sellers & review_sellers):,}"
    )

    if (
        delivery_sellers
        -
        all_sellers
    ):

        raise ValueError(
            "Delivery model contains sellers "
            "outside master population."
        )

    if (
        review_sellers
        -
        all_sellers
    ):

        raise ValueError(
            "Review model contains sellers "
            "outside master population."
        )

    print(
        "Temporal score populations "
        "contained within master population -> PASS"
    )

    # ========================================================
    # 11. MERGE MODEL OUTPUTS
    # ========================================================

    print_section(
        "COMBINING TEMPORAL MODEL OUTPUTS"
    )

    delivery_model_columns = [
        SELLER_COLUMN,
        SCORE_DATE_COLUMN,
        "delivery_risk_probability",
        "delivery_risk_threshold",
        "delivery_risk_decision",
        "delivery_model_coverage",
    ]

    review_model_columns = [
        SELLER_COLUMN,
        SCORE_DATE_COLUMN,
        "review_risk_probability",
        "review_risk_threshold",
        "review_risk_decision",
        "review_model_coverage",
    ]

    delivery_model_output = (
        delivery_scores[
            delivery_model_columns
        ]
        .copy()
    )

    review_model_output = (
        review_scores[
            review_model_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Explicitly rename review score date before merging.
    #
    # Both datasets have already been verified to use the
    # same date, so retaining one canonical score date is
    # preferable to letting pandas create _x / _y columns.
    # --------------------------------------------------------

    review_model_output = (
        review_model_output
        .drop(
            columns=[
                SCORE_DATE_COLUMN
            ]
        )
    )

    model_scores = (
        delivery_model_output
        .merge(
            review_model_output,
            on=SELLER_COLUMN,
            how="outer",
            validate="one_to_one",
        )
    )

    print(
        f"Combined temporal-score rows: "
        f"{len(model_scores):,}"
    )

    # ========================================================
    # 12. ADD MISSING COVERAGE FLAGS
    # ========================================================

    print_section(
        "SETTING MODEL COVERAGE STATUS"
    )

    model_scores[
        "delivery_model_coverage"
    ] = (
        model_scores[
            "delivery_model_coverage"
        ]
        .fillna(
            "NOT_AVAILABLE"
        )
    )

    model_scores[
        "review_model_coverage"
    ] = (
        model_scores[
            "review_model_coverage"
        ]
        .fillna(
            "NOT_AVAILABLE"
        )
    )

    model_scores[
        "delivery_score_available"
    ] = (
        model_scores[
            "delivery_risk_probability"
        ]
        .notna()
    )

    model_scores[
        "review_score_available"
    ] = (
        model_scores[
            "review_risk_probability"
        ]
        .notna()
    )

    print(
        f"Delivery scores available: "
        f"{model_scores['delivery_score_available'].sum():,}"
    )

    print(
        f"Review scores available: "
        f"{model_scores['review_score_available'].sum():,}"
    )

    print(
        "Coverage status assigned -> PASS"
    )

    # ========================================================
    # 13. MERGE WITH MASTER SELLER TABLE
    # ========================================================

    print_section(
        "BUILDING FINAL SELLER INTELLIGENCE DATASET"
    )

    final_df = (
        seller_risk_df
        .merge(
            model_scores,
            on=SELLER_COLUMN,
            how="left",
            validate="one_to_one",
        )
    )

    print(
        f"Final rows: "
        f"{len(final_df):,}"
    )

    if len(final_df) != len(
        seller_risk_df
    ):

        raise ValueError(
            "Final seller population changed."
        )

    print(
        "Master seller population preserved -> PASS"
    )

    # ========================================================
    # 14. DEFAULT COVERAGE FOR UNSCORED SELLERS
    # ========================================================

    print_section(
        "FINALIZING MODEL COVERAGE"
    )

    final_df[
        "delivery_model_coverage"
    ] = (
        final_df[
            "delivery_model_coverage"
        ]
        .fillna(
            "NOT_AVAILABLE"
        )
    )

    final_df[
        "review_model_coverage"
    ] = (
        final_df[
            "review_model_coverage"
        ]
        .fillna(
            "NOT_AVAILABLE"
        )
    )

    final_df[
        "temporal_model_coverage"
    ] = np.select(
        [
            (
                (
                    final_df[
                        "delivery_model_coverage"
                    ]
                    == "AVAILABLE"
                )
                &
                (
                    final_df[
                        "review_model_coverage"
                    ]
                    == "AVAILABLE"
                )
            ),
            (
                (
                    final_df[
                        "delivery_model_coverage"
                    ]
                    == "AVAILABLE"
                )
                |
                (
                    final_df[
                        "review_model_coverage"
                    ]
                    == "AVAILABLE"
                )
            ),
        ],
        [
            "BOTH_AVAILABLE",
            "PARTIAL",
        ],
        default="NOT_AVAILABLE",
    )

    final_df[
        SCORE_DATE_COLUMN
    ] = (
        final_df[
            SCORE_DATE_COLUMN
        ]
        .fillna(
            score_snapshot_date
        )
    )

    print(
        final_df[
            "temporal_model_coverage"
        ]
        .value_counts()
    )

    print(
        "Temporal coverage status -> PASS"
    )

    # ========================================================
    # 15. BUSINESS PRIORITY
    # ========================================================

    print_section(
        "BUILDING BUSINESS PRIORITY"
    )

    final_df[
        "business_priority"
    ] = (
        final_df
        .apply(
            assign_business_priority,
            axis=1,
        )
    )

    print(
        final_df[
            "business_priority"
        ]
        .value_counts()
    )

    # ========================================================
    # 16. BUSINESS ACTION
    # ========================================================

    print_section(
        "ASSIGNING BUSINESS ACTIONS"
    )

    final_df[
        "business_action"
    ] = (
        final_df[
            "business_priority"
        ]
        .map(
            assign_business_action
        )
    )

    print(
        "Business action mapping -> PASS"
    )

    # ========================================================
    # 17. DUAL-RISK ANALYSIS
    # ========================================================

    print_section(
        "DUAL-RISK ANALYSIS"
    )

    dual_risk = (
        (
            final_df[
                "delivery_risk_decision"
            ]
            == 1
        )
        &
        (
            final_df[
                "review_risk_decision"
            ]
            == 1
        )
    )

    dual_risk_count = (
        dual_risk.sum()
    )

    print(
        f"Dual-risk sellers: "
        f"{dual_risk_count:,}"
    )

    available_both = (
        final_df[
            "temporal_model_coverage"
        ]
        ==
        "BOTH_AVAILABLE"
    )

    if available_both.any():

        dual_risk_rate = (
            dual_risk[
                available_both
            ]
            .mean()
            * 100
        )

        print(
            f"Dual-risk percentage among "
            f"sellers with both scores: "
            f"{dual_risk_rate:.2f}%"
        )

    # ========================================================
    # 18. MODEL OUTPUT VALIDATION
    # ========================================================

    print_section(
        "VALIDATING TEMPORAL MODEL OUTPUTS"
    )

    for label in [
        "delivery",
        "review",
    ]:

        probability_column = (
            f"{label}_risk_probability"
        )

        coverage_column = (
            f"{label}_model_coverage"
        )

        decision_column = (
            f"{label}_risk_decision"
        )

        available = (
            final_df[
                coverage_column
            ]
            ==
            "AVAILABLE"
        )

        missing_probability = (
            (
                available
                &
                final_df[
                    probability_column
                ]
                .isna()
            )
            .sum()
        )

        missing_decision = (
            (
                available
                &
                final_df[
                    decision_column
                ]
                .isna()
            )
            .sum()
        )

        print(
            f"{label.title()} available "
            f"but missing probability: "
            f"{missing_probability}"
        )

        print(
            f"{label.title()} available "
            f"but missing decision: "
            f"{missing_decision}"
        )

        if (
            missing_probability > 0
            or
            missing_decision > 0
        ):

            raise ValueError(
                f"{label.title()} model coverage "
                "is inconsistent."
            )

        available_probabilities = (
            final_df.loc[
                available,
                probability_column,
            ]
        )

        invalid_probability = (
            (
                (
                    available_probabilities
                    < 0
                )
                |
                (
                    available_probabilities
                    > 1
                )
            )
            .sum()
        )

        print(
            f"{label.title()} invalid "
            f"available probabilities: "
            f"{invalid_probability}"
        )

        if invalid_probability > 0:

            raise ValueError(
                f"Invalid {label} probabilities."
            )

    print(
        "Temporal model output validation -> PASS"
    )

    # ========================================================
    # 19. NOT-SCORED VALIDATION
    # ========================================================

    print_section(
        "NOT-SCORED SELLER VALIDATION"
    )

    not_scored = (
        final_df[
            "business_priority"
        ]
        ==
        "NOT_SCORED"
    )

    not_scored_count = (
        not_scored.sum()
    )

    print(
        f"Not-scored sellers: "
        f"{not_scored_count:,}"
    )

    invalid_not_scored = (
        (
            final_df[
                "business_priority"
            ]
            ==
            "NOT_SCORED"
        )
        &
        (
            final_df[
                "delivery_risk_probability"
            ]
            .notna()
            |
            final_df[
                "review_risk_probability"
            ]
            .notna()
        )
    ).sum()

    print(
        f"Not-scored sellers with "
        f"temporal probability: "
        f"{invalid_not_scored}"
    )

    if invalid_not_scored > 0:

        raise ValueError(
            "NOT_SCORED status conflicts "
            "with temporal score availability."
        )

    print(
        "Not-scored logic -> PASS"
    )

    # ========================================================
    # 20. SELLER UNIQUENESS
    # ========================================================

    print_section(
        "FINAL SELLER UNIQUENESS"
    )

    duplicate_sellers = (
        final_df[
            SELLER_COLUMN
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate seller IDs: "
        f"{duplicate_sellers}"
    )

    if duplicate_sellers > 0:

        raise ValueError(
            "Final dataset contains duplicate sellers."
        )

    print(
        "Seller uniqueness -> PASS"
    )

    # ========================================================
    # 21. SCORE SNAPSHOT VALIDATION
    # ========================================================

    print_section(
        "FINAL SCORE SNAPSHOT VALIDATION"
    )

    if (
        final_df[
            SCORE_DATE_COLUMN
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Final score snapshot date contains missing values."
        )

    unique_score_dates = (
        final_df[
            SCORE_DATE_COLUMN
        ]
        .unique()
    )

    print(
        f"Score snapshot dates: "
        f"{len(unique_score_dates)}"
    )

    print(
        f"Score date: "
        f"{unique_score_dates[0].date()}"
    )

    print(
        "Score snapshot validation -> PASS"
    )

    # ========================================================
    # 22. FINAL PREVIEW
    # ========================================================

    print_section(
        "FINAL SELLER INTELLIGENCE PREVIEW"
    )

    preview_columns = [
        SELLER_COLUMN,
        SCORE_DATE_COLUMN,

        "risk_score",
        "risk_level",

        "delivery_risk_probability",
        "delivery_risk_decision",
        "delivery_model_coverage",

        "review_risk_probability",
        "review_risk_decision",
        "review_model_coverage",

        "temporal_model_coverage",

        "business_priority",
        "business_action",
    ]

    preview_columns = [
        column
        for column
        in preview_columns
        if column in final_df.columns
    ]

    priority_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MONITOR": 2,
        "LOW": 3,
        "NOT_SCORED": 4,
    }

    preview = (
        final_df
        .assign(
            _priority_order=
                final_df[
                    "business_priority"
                ]
                .map(
                    priority_order
                )
        )
        .sort_values(
            "_priority_order"
        )
        .drop(
            columns=[
                "_priority_order"
            ]
        )
        .head(20)
    )

    print(
        preview[
            preview_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 23. SAVE
    # ========================================================

    print_section(
        "SAVING FINAL SELLER RISK DATASET"
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Final seller risk dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # 24. FINAL SUMMARY
    # ========================================================

    print_section(
        "FINAL SELLER RISK DATASET COMPLETE"
    )

    print(
        f"Total sellers: "
        f"{len(final_df):,}"
    )

    print(
        f"Delivery temporal coverage: "
        f"{(
            final_df[
                'delivery_model_coverage'
            ]
            == 'AVAILABLE'
        ).sum():,}"
    )

    print(
        f"Review temporal coverage: "
        f"{(
            final_df[
                'review_model_coverage'
            ]
            == 'AVAILABLE'
        ).sum():,}"
    )

    print(
        f"Both temporal models available: "
        f"{(
            final_df[
                'temporal_model_coverage'
            ]
            ==
            'BOTH_AVAILABLE'
        ).sum():,}"
    )

    print(
        f"Not scored by temporal models: "
        f"{not_scored_count:,}"
    )

    print(
        f"Scoring snapshot: "
        f"{score_snapshot_date.date()}"
    )

    print(
        "\nImportant:"
    )

    print(
        "NOT_AVAILABLE means temporal model coverage "
        "was unavailable."
    )

    print(
        "It is not interpreted as zero risk."
    )

    print(
        "\nNo model was retrained."
    )

    print(
        "No test-period threshold selection occurred."
    )

    print(
        "The final dataset is ready for the "
        "application engineering layer."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()