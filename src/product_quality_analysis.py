from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    products = pd.read_csv(
        RAW_DATA_DIR / "olist_products_dataset.csv"
    )

    order_items = pd.read_csv(
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    )

    return products, order_items


# =========================================================
# IDENTIFY INCOMPLETE PRODUCTS
# =========================================================

def identify_incomplete_products(products):

    content_columns = [
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty"
    ]

    incomplete = products[
        products[content_columns]
        .isna()
        .all(axis=1)
    ].copy()

    return incomplete


# =========================================================
# CONNECT PRODUCTS TO SALES
# =========================================================

def analyze_product_sales(incomplete_products, order_items):

    sales = (
        order_items
        .groupby("product_id")
        .agg(
            order_count=("order_id", "nunique"),
            units_sold=("order_item_id", "count"),
            total_revenue=("price", "sum")
        )
        .reset_index()
    )

    result = incomplete_products.merge(
        sales,
        on="product_id",
        how="left"
    )

    result["order_count"] = (
        result["order_count"]
        .fillna(0)
    )

    result["units_sold"] = (
        result["units_sold"]
        .fillna(0)
    )

    result["total_revenue"] = (
        result["total_revenue"]
        .fillna(0)
    )

    return result


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():

    print("=" * 70)
    print("PRODUCT QUALITY ANALYSIS")
    print("=" * 70)

    products, order_items = load_data()

    incomplete_products = identify_incomplete_products(
        products
    )

    print(
        f"\nProducts with all four content fields missing: "
        f"{len(incomplete_products)}"
    )

    result = analyze_product_sales(
        incomplete_products,
        order_items
    )

    print("\nSales activity of incomplete products:")

    print(
        result[
            [
                "product_id",
                "order_count",
                "units_sold",
                "total_revenue"
            ]
        ]
        .sort_values(
            "total_revenue",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nSummary:")

    print(
        f"Products with sales: "
        f"{(result['order_count'] > 0).sum()}"
    )

    print(
        f"Products without sales: "
        f"{(result['order_count'] == 0).sum()}"
    )

    print(
        f"Total orders involving incomplete products: "
        f"{result['order_count'].sum():,.0f}"
    )

    print(
        f"Total units sold: "
        f"{result['units_sold'].sum():,.0f}"
    )

    print(
        f"Total revenue: "
        f"{result['total_revenue'].sum():,.2f}"
    )

    print("\n" + "=" * 70)
    print("PRODUCT QUALITY ANALYSIS COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()