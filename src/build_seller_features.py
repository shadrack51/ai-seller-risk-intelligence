from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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
# PRODUCT QUALITY FEATURES
# =========================================================

def build_product_quality_features(products):

    content_columns = [
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty"
    ]

    products = products.copy()

    products["is_incomplete"] = (
        products[content_columns]
        .isna()
        .all(axis=1)
    )

    product_features = (
        products
        .groupby("product_id")
        .agg(
            is_incomplete=("is_incomplete", "max")
        )
        .reset_index()
    )

    return product_features


# =========================================================
# SELLER PRODUCT FEATURES
# =========================================================

def build_seller_product_features(
    order_items,
    product_features
):

    seller_products = (
        order_items[
            [
                "seller_id",
                "product_id"
            ]
        ]
        .drop_duplicates()
    )

    seller_products = seller_products.merge(
        product_features,
        on="product_id",
        how="left"
    )

    result = (
        seller_products
        .groupby("seller_id")
        .agg(
            total_products=(
                "product_id",
                "nunique"
            ),
            incomplete_products=(
                "is_incomplete",
                "sum"
            )
        )
        .reset_index()
    )

    result["catalog_completeness"] = (
        1
        -
        (
            result["incomplete_products"]
            /
            result["total_products"]
        )
    ) * 100

    return result


# =========================================================
# SELLER SALES FEATURES
# =========================================================

def build_seller_sales_features(order_items):

    result = (
        order_items
        .groupby("seller_id")
        .agg(
            total_orders=(
                "order_id",
                "nunique"
            ),
            units_sold=(
                "order_item_id",
                "count"
            ),
            total_revenue=(
                "price",
                "sum"
            ),
            average_item_price=(
                "price",
                "mean"
            ),
            average_freight=(
                "freight_value",
                "mean"
            )
        )
        .reset_index()
    )

    return result


# =========================================================
# BUILD SELLER FEATURE TABLE
# =========================================================

def build_seller_features(
    sellers,
    product_features,
    order_items
):

    seller_product_features = (
        build_seller_product_features(
            order_items,
            product_features
        )
    )

    seller_sales_features = (
        build_seller_sales_features(
            order_items
        )
    )

    result = sellers.merge(
        seller_product_features,
        on="seller_id",
        how="left"
    )

    result = result.merge(
        seller_sales_features,
        on="seller_id",
        how="left"
    )

    result["incomplete_products"] = (
        result["incomplete_products"]
        .fillna(0)
    )

    result["catalog_completeness"] = (
        result["catalog_completeness"]
        .fillna(100)
    )

    return result


# =========================================================
# SAVE FEATURE TABLE
# =========================================================

def save_features(df):

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DATA_DIR
        / "seller_features.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSeller feature table saved to:\n"
        f"{output_file}"
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("=" * 70)
    print("BUILDING SELLER FEATURE TABLE")
    print("=" * 70)

    products, order_items, sellers = load_data()

    product_features = (
        build_product_quality_features(
            products
        )
    )

    seller_features = build_seller_features(
        sellers,
        product_features,
        order_items
    )

    print(
        f"\nNumber of sellers: "
        f"{len(seller_features)}"
    )

    print(
        f"Number of features: "
        f"{len(seller_features.columns)}"
    )

    print("\nFeature table preview:")

    print(
        seller_features.head(10)
        .to_string(index=False)
    )

    save_features(
        seller_features
    )

    print("\n" + "=" * 70)
    print("SELLER FEATURE TABLE COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()