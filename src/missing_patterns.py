from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# INVESTIGATE PRODUCT MISSINGNESS
# =========================================================

def analyze_product_missingness():

    file_path = RAW_DATA_DIR / "olist_products_dataset.csv"

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("PRODUCT MISSINGNESS ANALYSIS")
    print("=" * 70)

    missing_columns = [
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    missing_mask = df[missing_columns].isna()

    print("\nMissing values by column:")
    print(
        missing_mask.sum()
        .sort_values(ascending=False)
    )

    print("\nRows with at least one missing value:")

    rows_with_missing = missing_mask.any(axis=1)

    print(
        rows_with_missing.sum()
    )

    print("\nRows with all four content fields missing:")

    content_columns = [
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty"
    ]

    all_content_missing = (
        df[content_columns]
        .isna()
        .all(axis=1)
    )

    print(
        all_content_missing.sum()
    )

    print("\nRows with missing physical dimensions:")

    physical_columns = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    physical_missing = (
        df[physical_columns]
        .isna()
        .any(axis=1)
    )

    print(
        physical_missing.sum()
    )


# =========================================================
# INVESTIGATE ORDER MISSINGNESS
# =========================================================

def analyze_order_missingness():

    file_path = RAW_DATA_DIR / "olist_orders_dataset.csv"

    df = pd.read_csv(file_path)

    print("\n" + "=" * 70)
    print("ORDER MISSINGNESS ANALYSIS")
    print("=" * 70)

    date_columns = [
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date"
    ]

    print("\nMissing values by order status:")

    for column in date_columns:

        print(f"\n{column}")

        result = (
            df[df[column].isna()]
            ["order_status"]
            .value_counts()
        )

        print(result)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    analyze_product_missingness()

    analyze_order_missingness()

    print("\n" + "=" * 70)
    print("MISSINGNESS PATTERN ANALYSIS COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()