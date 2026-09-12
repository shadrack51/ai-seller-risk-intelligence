from pathlib import Path
import pandas as pd



# PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEGMENT_FILE = PROCESSED_DATA_DIR / "seller_segments.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "segment_analysis.csv"



# CONFIGURATION


SEGMENT_COLUMN = "seller_segment"

PROFILE_COLUMNS = [
    "total_products",
    "incomplete_products",
    "catalog_completeness",
    "total_orders",
    "units_sold",
    "total_revenue",
    "average_item_price",
    "average_freight",
    "average_delivery_days",
    "average_delivery_delay_days",
    "late_delivery_rate",
    "average_review_score",
    "review_count",
    "low_review_rate",
]



# HELPER FUNCTIONS


def load_segment_data():
    """Load the seller segmentation dataset."""

    if not SEGMENT_FILE.exists():
        raise FileNotFoundError(
            f"Segmentation dataset not found:\n{SEGMENT_FILE}\n\n"
            "Run seller_segmentation.py first."
        )

    df = pd.read_csv(SEGMENT_FILE)

    return df


def validate_data(df):
    """Validate the segmentation dataset."""

    required_columns = [
        "seller_id",
        SEGMENT_COLUMN,
        "total_revenue",
        "total_orders",
        "units_sold",
        "catalog_completeness",
        "incomplete_products",
        "average_review_score",
        "low_review_rate",
        "risk_category",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    duplicate_sellers = df["seller_id"].duplicated().sum()

    if duplicate_sellers > 0:
        raise ValueError(
            f"Dataset contains {duplicate_sellers} duplicate seller IDs."
        )


def calculate_segment_profiles(df):
    """Calculate average characteristics for every seller segment."""

    profiles = (
        df.groupby(SEGMENT_COLUMN)[PROFILE_COLUMNS]
        .mean()
        .round(2)
    )

    return profiles


def calculate_segment_counts(df):
    """Calculate seller counts and percentages."""

    counts = (
        df[SEGMENT_COLUMN]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[SEGMENT_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    result = pd.DataFrame({
        "sellers": counts,
        "seller_percentage": percentages.round(2),
    })

    return result


def calculate_segment_commercial_metrics(df):
    """Calculate commercial performance for each segment."""

    grouped = df.groupby(SEGMENT_COLUMN)

    commercial = grouped.agg(
        sellers=("seller_id", "count"),
        total_revenue=("total_revenue", "sum"),
        total_orders=("total_orders", "sum"),
        units_sold=("units_sold", "sum"),
        average_revenue=("total_revenue", "mean"),
        average_orders=("total_orders", "mean"),
        average_units_sold=("units_sold", "mean"),
    )

    commercial["revenue_percentage"] = (
        commercial["total_revenue"]
        / commercial["total_revenue"].sum()
        * 100
    )

    commercial["revenue_per_seller"] = (
        commercial["total_revenue"]
        / commercial["sellers"]
    )

    commercial = commercial.round(2)

    return commercial


def calculate_risk_metrics(df):
    """Calculate risk distribution for every segment."""

    risk_distribution = pd.crosstab(
        df[SEGMENT_COLUMN],
        df["risk_category"]
    )

    risk_distribution["total_sellers"] = (
        risk_distribution.sum(axis=1)
    )

    if "HIGH" in risk_distribution.columns:
        risk_distribution["high_risk_percentage"] = (
            risk_distribution["HIGH"]
            / risk_distribution["total_sellers"]
            * 100
        )
    else:
        risk_distribution["high_risk_percentage"] = 0

    if "MEDIUM" in risk_distribution.columns:
        risk_distribution["medium_risk_percentage"] = (
            risk_distribution["MEDIUM"]
            / risk_distribution["total_sellers"]
            * 100
        )
    else:
        risk_distribution["medium_risk_percentage"] = 0

    risk_distribution = risk_distribution.round(2)

    return risk_distribution


def calculate_business_scores(profiles, commercial, risk_distribution):
    """
    Create interpretable business indicators for each segment.

    These are analytical scores, not machine-learning predictions.
    """

    result = profiles.copy()

    # --------------------------------------------------------
    # COMMERCIAL IMPORTANCE
    # --------------------------------------------------------

    result["revenue_per_seller"] = commercial["revenue_per_seller"]

    result["revenue_percentage"] = commercial["revenue_percentage"]

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    result["high_risk_percentage"] = (
        risk_distribution["high_risk_percentage"]
    )

    result["medium_risk_percentage"] = (
        risk_distribution["medium_risk_percentage"]
    )

    # --------------------------------------------------------
    # QUALITY INDICATORS
    # --------------------------------------------------------

    result["catalog_quality_score"] = (
        result["catalog_completeness"]
    )

    result["customer_satisfaction_score"] = (
        result["average_review_score"] / 5 * 100
    )

    result["delivery_performance_score"] = (
        100 - result["late_delivery_rate"]
    )

    # --------------------------------------------------------
    # BUSINESS PRIORITY SCORE
    # --------------------------------------------------------
    #
    # This score combines:
    #
    #   40% commercial importance
    #   30% risk exposure
    #   30% catalog/customer quality
    #
    # It is intended for prioritization, not prediction.
    #

    revenue_rank = (
        result["revenue_percentage"]
        / result["revenue_percentage"].max()
    )

    risk_rank = (
        result["high_risk_percentage"]
        + result["medium_risk_percentage"] * 0.5
    )

    if risk_rank.max() > 0:
        risk_rank = risk_rank / risk_rank.max()
    else:
        risk_rank = 0

    quality_problem = (
        (
            100 - result["catalog_completeness"]
        )
        +
        (
            100 - result["customer_satisfaction_score"]
        )
        +
        result["late_delivery_rate"]
    ) / 300

    result["business_priority_score"] = (
        revenue_rank * 40
        + risk_rank * 30
        + quality_problem * 30
    )

    result["business_priority_score"] = (
        result["business_priority_score"]
        .round(2)
    )

    return result


def assign_segment_labels(df):
    """
    Assign human-readable labels based on observed segment behavior.

    Labels are generated from the characteristics of the current
    Olist segmentation result.
    """

    labels = {}

    for segment in sorted(df[SEGMENT_COLUMN].unique()):

        segment_data = df[
            df[SEGMENT_COLUMN] == segment
        ]

        avg_revenue = segment_data["total_revenue"].mean()
        avg_orders = segment_data["total_orders"].mean()
        avg_catalog = segment_data["catalog_completeness"].mean()
        avg_review = segment_data["average_review_score"].mean()
        avg_low_review = segment_data["low_review_rate"].mean()
        avg_products = segment_data["total_products"].mean()

        if (
            avg_revenue > 5000
            and avg_orders > 50
            and avg_catalog > 95
        ):
            label = "High-Volume Established Sellers"

        elif (
            avg_catalog < 70
            or segment_data["incomplete_products"].mean() > 1
        ):
            label = "Catalog Risk Sellers"

        elif avg_review < 2.5 or avg_low_review > 60:
            label = "Poor Customer Experience Sellers"

        elif avg_revenue < 1000 and avg_orders < 10:
            label = "Small / Low-Volume Sellers"

        elif avg_products < 5 and avg_review < 3.5:
            label = "Low-Performance Sellers"

        else:
            label = "Developing Sellers"

        labels[segment] = label

    return labels


def build_final_analysis(
    df,
    profiles,
    counts,
    commercial,
    risk_distribution,
    business_scores,
    labels,
):
    """Combine all analytical results into one final dataset."""

    result = profiles.copy()

    result = result.join(counts)

    result = result.join(
        commercial[
            [
                "total_revenue",
                "total_orders",
                "units_sold",
                "average_revenue",
                "average_orders",
                "average_units_sold",
                "revenue_percentage",
                "revenue_per_seller",
            ]
        ],
        rsuffix="_commercial"
    )

    result = result.join(
        risk_distribution[
            [
                "high_risk_percentage",
                "medium_risk_percentage",
            ]
        ]
    )

    result["business_priority_score"] = (
        business_scores["business_priority_score"]
    )

    result["segment_label"] = (
        pd.Series(labels)
    )

    result = result.reset_index()

    return result



# MAIN ANALYSIS


def main():

    print("=" * 70)
    print("SELLER SEGMENT BUSINESS ANALYSIS")
    print("=" * 70)

  
    # LOAD DATA


    print("\nLoading seller segmentation dataset...")

    df = load_segment_data()

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

  
    # VALIDATE
    

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    validate_data(df)

    print("Duplicate seller IDs: 0")
    print("Required columns: PASS")

   
    # SEGMENT COUNTS
  

    print("\n" + "=" * 70)
    print("SEGMENT SIZE ANALYSIS")
    print("=" * 70)

    counts = calculate_segment_counts(df)

    print(counts)

    
    # SEGMENT PROFILES
    

    print("\n" + "=" * 70)
    print("SEGMENT PROFILES")
    print("=" * 70)

    profiles = calculate_segment_profiles(df)

    print(profiles.to_string())

    
    # COMMERCIAL PERFORMANCE
    

    print("\n" + "=" * 70)
    print("COMMERCIAL PERFORMANCE")
    print("=" * 70)

    commercial = calculate_segment_commercial_metrics(df)

    print(commercial.to_string())

  
    # RISK ANALYSIS
   

    print("\n" + "=" * 70)
    print("SEGMENT RISK ANALYSIS")
    print("=" * 70)

    risk_distribution = calculate_risk_metrics(df)

    print(risk_distribution.to_string())

    
    # HUMAN-READABLE LABELS
    

    print("\n" + "=" * 70)
    print("SEGMENT BUSINESS LABELS")
    print("=" * 70)

    labels = assign_segment_labels(df)

    for segment, label in labels.items():
        print(f"Segment {segment}: {label}")

    
    # BUSINESS PRIORITY


    print("\n" + "=" * 70)
    print("BUSINESS PRIORITY ANALYSIS")
    print("=" * 70)

    business_scores = calculate_business_scores(
        profiles,
        commercial,
        risk_distribution,
    )

    priority = business_scores[
        [
            "business_priority_score",
            "revenue_percentage",
            "revenue_per_seller",
            "high_risk_percentage",
            "medium_risk_percentage",
        ]
    ].sort_values(
        "business_priority_score",
        ascending=False,
    )

    print(priority.to_string())

   
    # FINAL ANALYSIS DATASET
    

    final_analysis = build_final_analysis(
        df=df,
        profiles=profiles,
        counts=counts,
        commercial=commercial,
        risk_distribution=risk_distribution,
        business_scores=business_scores,
        labels=labels,
    )

   
    # SAVE
   

    final_analysis.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # FINAL SUMMARY
   

    print("\n" + "=" * 70)
    print("TOP PRIORITY SEGMENTS")
    print("=" * 70)

    top_priority = final_analysis[
        [
            SEGMENT_COLUMN,
            "segment_label",
            "sellers",
            "seller_percentage",
            "total_revenue",
            "revenue_percentage",
            "high_risk_percentage",
            "business_priority_score",
        ]
    ].sort_values(
        "business_priority_score",
        ascending=False,
    )

    print(top_priority.to_string(index=False))

    print("\n" + "=" * 70)
    print("SEGMENT ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nNumber of sellers analyzed: {len(df):,}")
    print(
        f"Number of segments analyzed: "
        f"{df[SEGMENT_COLUMN].nunique()}"
    )

    print(
        "\nSegment analysis dataset saved to:"
    )
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()