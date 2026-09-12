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

    return products[
        products[content_columns]
        .isna()
        .all(axis=1)
    ].copy()


# =========================================================
# BUILD SELLER IMPACT TABLE
# =========================================================

def build_seller_impact(
    incomplete_products,
    order_items
):

    incomplete_items = order_items[
        order_items["product_id"].isin(
            incomplete_products["product_id"]
        )
    ].copy()

    seller_impact = (
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

    return seller_impact


# =========================================================
# CALCULATE CONCENTRATION
# =========================================================

def calculate_concentration(seller_impact):

    result = seller_impact.sort_values(
        "revenue",
        ascending=False
    ).copy()

    result["cumulative_revenue"] = (
        result["revenue"].cumsum()
    )

    total_revenue = result["revenue"].sum()

    result["cumulative_revenue_percentage"] = (
        result["cumulative_revenue"]
        / total_revenue
        * 100
    )

    result["seller_rank"] = range(
        1,
        len(result) + 1
    )

    return result


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():

    print("=" * 70)
    print("SELLER CONCENTRATION ANALYSIS")
    print("=" * 70)

    products, order_items, sellers = load_data()

    incomplete_products = (
        identify_incomplete_products(products)
    )

    seller_impact = build_seller_impact(
        incomplete_products,
        order_items
    )

    result = calculate_concentration(
        seller_impact
    )

    total_revenue = result["revenue"].sum()

    print(
        f"\nTotal revenue from incomplete products: "
        f"{total_revenue:,.2f}"
    )

    print(
        f"Total affected sellers: "
        f"{len(result)}"
    )

    # -----------------------------------------------------
    # TOP 5
    # -----------------------------------------------------

    print("\nTop 5 sellers by affected revenue:")

    print(
        result[
            [
                "seller_rank",
                "seller_id",
                "incomplete_products",
                "orders",
                "units_sold",
                "revenue",
                "cumulative_revenue_percentage"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # TOP 10
    # -----------------------------------------------------

    print("\nTop 10 sellers by affected revenue:")

    print(
        result[
            [
                "seller_rank",
                "seller_id",
                "incomplete_products",
                "orders",
                "units_sold",
                "revenue",
                "cumulative_revenue_percentage"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # CONCENTRATION THRESHOLDS
    # -----------------------------------------------------

    print("\nRevenue concentration:")

    for percentage in [50, 80, 90, 95]:

        sellers_needed = (
            result[
                result["cumulative_revenue_percentage"]
                <= percentage
            ]
        )

        if len(sellers_needed) == 0:

            sellers_needed = 1

        else:

            sellers_needed = (
                sellers_needed["seller_rank"].max()
            )

        print(
            f"Sellers needed to reach approximately "
            f"{percentage}% of affected revenue: "
            f"{sellers_needed}"
        )

    # -----------------------------------------------------
    # HIGH-COMMERCIAL-IMPACT SELLERS
    # -----------------------------------------------------

    high_impact = result[
        result["revenue"] >= 5000
    ]

    print(
        f"\nSellers generating at least 5,000 "
        f"in affected revenue: "
        f"{len(high_impact)}"
    )

    # -----------------------------------------------------
    # HIGH-CATALOG-IMPACT SELLERS
    # -----------------------------------------------------

    high_catalog = result[
        result["incomplete_products"] >= 10
    ]

    print(
        f"Sellers with at least 10 incomplete products: "
        f"{len(high_catalog)}"
    )

    print("\n" + "=" * 70)
    print("SELLER CONCENTRATION ANALYSIS COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()