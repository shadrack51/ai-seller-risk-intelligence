from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# DATASETS TO ANALYZE
# =========================================================

DATASETS = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv"
]


# =========================================================
# ANALYZE MISSING VALUES
# =========================================================

def analyze_missing_values(file_path):
    """
    Calculate missing-value statistics for a dataset.
    """

    df = pd.read_csv(file_path)

    missing_count = df.isna().sum()

    missing_percentage = (
        df.isna().mean() * 100
    )

    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_percentage.values
    })

    result = result[
        result["missing_count"] > 0
    ]

    result = result.sort_values(
        "missing_percentage",
        ascending=False
    )

    return result


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("=" * 70)
    print("OLIST MISSING-VALUE ANALYSIS")
    print("=" * 70)

    total_missing = 0

    for dataset_name in DATASETS:

        file_path = RAW_DATA_DIR / dataset_name

        result = analyze_missing_values(file_path)

        print("\n" + "-" * 70)
        print(f"DATASET: {dataset_name}")
        print("-" * 70)

        if result.empty:

            print("No missing values found.")

        else:

            print(result.to_string(index=False))

            total_missing += result["missing_count"].sum()


    print("\n" + "=" * 70)
    print(
        f"TOTAL MISSING CELLS ACROSS DATASETS: "
        f"{int(total_missing):,}"
    )
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()