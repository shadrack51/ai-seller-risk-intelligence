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

    sellers = pd.read_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv"
    )

    return products, order_items, sellers


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

    incomplete_products = products[
        products[content_columns]
        .isna()
        .all(axis=1)
    ].copy()

    return incomplete_products


# =========================================================
# CONNECT INCOMPLETE PRODUCTS TO SELLERS
# =========================================================

def analyze_sellers(
    incomplete_products,
    order_items,
    sellers
):

    incomplete_items = order_items[
        order_items["product_id"].isin(
            incomplete_products["product_id"]
        )
    ].copy()

    seller_analysis = (
        incomplete_items
        .groupby("seller_id")
        .agg(
            incomplete_products=(
                "product_id",
                "nunique"
            ),
            orders=(
                "order_id",
                "nunique"
            ),
            units_sold=(
                "order_item_id",
                "count"
            ),
            revenue=(
                "price",
                "sum"
            )
        )
        .reset_index()
    )

    seller_analysis = seller_analysis.merge(
        sellers,
        on="seller_id",
        how="left"
    )

    seller_analysis = seller_analysis.sort_values(
        "revenue",
        ascending=False
    )

    return seller_analysis


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():

    print("=" * 70)
    print("SELLER DATA QUALITY ANALYSIS")
    print("=" * 70)

    products, order_items, sellers = load_data()

    incomplete_products = identify_incomplete_products(
        products
    )

    seller_analysis = analyze_sellers(
        incomplete_products,
        order_items,
        sellers
    )

    print(
        f"\nSellers involved with incomplete products: "
        f"{len(seller_analysis)}"
    )

    print("\nTop sellers by revenue from incomplete products:")

    print(
        seller_analysis[
            [
                "seller_id",
                "seller_city",
                "seller_state",
                "incomplete_products",
                "orders",
                "units_sold",
                "revenue"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSummary:")

    print(
        f"Total sellers involved: "
        f"{len(seller_analysis)}"
    )

    print(
        f"Total incomplete products: "
        f"{seller_analysis['incomplete_products'].sum()}"
    )

    print(
        f"Total orders: "
        f"{seller_analysis['orders'].sum()}"
    )

    print(
        f"Total units sold: "
        f"{seller_analysis['units_sold'].sum()}"
    )

    print(
        f"Total revenue: "
        f"{seller_analysis['revenue'].sum():,.2f}"
    )

    print("\n" + "=" * 70)
    print("SELLER DATA QUALITY ANALYSIS COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()