from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# =========================================================
# LOAD SELLER FEATURES
# =========================================================

def load_seller_features():

    file_path = (
        PROCESSED_DATA_DIR
        / "seller_features.csv"
    )

    df = pd.read_csv(file_path)

    return df


# =========================================================
# BASIC DATASET SUMMARY
# =========================================================

def dataset_summary(df):

    print("\n" + "=" * 70)
    print("SELLER FEATURE DATASET SUMMARY")
    print("=" * 70)

    print(f"\nNumber of sellers: {len(df)}")
    print(f"Number of features: {len(df.columns)}")

    print("\nColumns:")

    for column in df.columns:
        print(f" - {column}")


# =========================================================
# CATALOG QUALITY ANALYSIS
# =========================================================

def catalog_quality_analysis(df):

    print("\n" + "=" * 70)
    print("CATALOG QUALITY ANALYSIS")
    print("=" * 70)

    incomplete_sellers = (
        df["incomplete_products"] > 0
    ).sum()

    complete_sellers = (
        df["incomplete_products"] == 0
    ).sum()

    print(
        f"\nSellers with incomplete products: "
        f"{incomplete_sellers}"
    )

    print(
        f"Sellers with completely complete catalogs: "
        f"{complete_sellers}"
    )

    print("\nCatalog completeness statistics:")

    print(
        df["catalog_completeness"]
        .describe()
        .to_string()
    )


# =========================================================
# REVENUE ANALYSIS
# =========================================================

def revenue_analysis(df):

    print("\n" + "=" * 70)
    print("SELLER REVENUE ANALYSIS")
    print("=" * 70)

    print("\nRevenue statistics:")

    print(
        df["total_revenue"]
        .describe()
        .to_string()
    )

    print("\nTop 10 sellers by revenue:")

    top_sellers = (
        df[
            [
                "seller_id",
                "total_orders",
                "units_sold",
                "total_revenue",
                "catalog_completeness"
            ]
        ]
        .sort_values(
            "total_revenue",
            ascending=False
        )
        .head(10)
    )

    print(
        top_sellers
        .to_string(index=False)
    )


# =========================================================
# PERCENTILE ANALYSIS
# =========================================================

def percentile_analysis(df):

    print("\n" + "=" * 70)
    print("SELLER REVENUE PERCENTILES")
    print("=" * 70)

    percentiles = (
        df["total_revenue"]
        .quantile(
            [
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    print(
        percentiles
        .to_string()
    )


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def correlation_analysis(df):

    print("\n" + "=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    numerical_columns = [
        "total_products",
        "incomplete_products",
        "catalog_completeness",
        "total_orders",
        "units_sold",
        "total_revenue",
        "average_item_price",
        "average_freight"
    ]

    correlation_matrix = (
        df[numerical_columns]
        .corr()
    )

    print(
        correlation_matrix
        .round(3)
        .to_string()
    )


# =========================================================
# OUTLIER ANALYSIS
# =========================================================

def outlier_analysis(df):

    print("\n" + "=" * 70)
    print("REVENUE OUTLIER ANALYSIS")
    print("=" * 70)

    q1 = df["total_revenue"].quantile(0.25)

    q3 = df["total_revenue"].quantile(0.75)

    iqr = q3 - q1

    upper_limit = q3 + (1.5 * iqr)

    outliers = df[
        df["total_revenue"] > upper_limit
    ]

    print(f"\nQ1: {q1:,.2f}")

    print(f"Q3: {q3:,.2f}")

    print(f"IQR: {iqr:,.2f}")

    print(
        f"Upper outlier limit: "
        f"{upper_limit:,.2f}"
    )

    print(
        f"\nNumber of revenue outliers: "
        f"{len(outliers)}"
    )

    print("\nTop revenue outliers:")

    print(
        outliers[
            [
                "seller_id",
                "total_revenue",
                "total_orders",
                "catalog_completeness"
            ]
        ]
        .sort_values(
            "total_revenue",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    df = load_seller_features()

    dataset_summary(df)

    catalog_quality_analysis(df)

    revenue_analysis(df)

    percentile_analysis(df)

    correlation_analysis(df)

    outlier_analysis(df)

    print("\n" + "=" * 70)
    print("SELLER FEATURE EDA COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()