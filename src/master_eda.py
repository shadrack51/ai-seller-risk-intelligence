from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_FILE = PROCESSED_DATA_DIR / "seller_master.csv"


# ============================================================
# LOAD SELLER MASTER DATASET
# ============================================================

print("=" * 70)
print("SELLER MASTER EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

print()
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# NUMERIC FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(include="number").columns

print(df[numeric_columns].describe().round(2))


# ============================================================
# SELLER RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SELLER RISK DISTRIBUTION")
print("=" * 70)

risk_counts = df["risk_category"].value_counts()

print(risk_counts)

print("\nRisk percentages:")

risk_percentages = (
    df["risk_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(risk_percentages)


# ============================================================
# CATALOG QUALITY
# ============================================================

print("\n" + "=" * 70)
print("CATALOG QUALITY")
print("=" * 70)

print("\nCatalog completeness statistics:")

print(
    df["catalog_completeness"]
    .describe()
    .round(2)
)

incomplete_sellers = (
    df["incomplete_products"] > 0
).sum()

complete_sellers = (
    df["incomplete_products"] == 0
).sum()

print(
    f"\nSellers with incomplete products: "
    f"{incomplete_sellers:,}"
)

print(
    f"Sellers with complete catalogs: "
    f"{complete_sellers:,}"
)


# ============================================================
# REVENUE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("REVENUE DISTRIBUTION")
print("=" * 70)

print("\nRevenue statistics:")

print(
    df["total_revenue"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .round(2)
)


# ============================================================
# TOP SELLERS BY REVENUE
# ============================================================

print("\nTop 10 sellers by revenue:")

top_revenue = (
    df[
        [
            "seller_id",
            "total_revenue",
            "total_orders",
            "units_sold",
            "catalog_completeness",
            "risk_category",
        ]
    ]
    .sort_values(
        "total_revenue",
        ascending=False
    )
    .head(10)
)

print(
    top_revenue.to_string(index=False)
)


# ============================================================
# HIGH-RISK SELLERS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-RISK SELLERS")
print("=" * 70)

high_risk = df[
    df["risk_category"] == "HIGH"
]

print(
    f"High-risk sellers: {len(high_risk)}"
)

if len(high_risk) > 0:

    high_risk_display = (
        high_risk[
            [
                "seller_id",
                "catalog_completeness",
                "seller_risk_score",
                "total_revenue",
                "total_orders",
                "late_delivery_rate",
                "average_review_score",
            ]
        ]
        .sort_values(
            "seller_risk_score",
            ascending=False
        )
    )

    print(
        high_risk_display.to_string(index=False)
    )


# ============================================================
# CORRELATION WITH REVENUE
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH REVENUE")
print("=" * 70)

correlations = (
    df[numeric_columns]
    .corr()["total_revenue"]
    .sort_values(ascending=False)
)

print(
    correlations.round(3)
)


# ============================================================
# MASTER EDA COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MASTER EDA COMPLETE")
print("=" * 70)