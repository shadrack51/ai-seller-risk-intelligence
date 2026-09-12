from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# LOAD DATASETS
# =========================================================

def load_datasets():

    customers = pd.read_csv(
        RAW_DATA_DIR / "olist_customers_dataset.csv"
    )

    orders = pd.read_csv(
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    )

    order_items = pd.read_csv(
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    )

    payments = pd.read_csv(
        RAW_DATA_DIR / "olist_order_payments_dataset.csv"
    )

    reviews = pd.read_csv(
        RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
    )

    products = pd.read_csv(
        RAW_DATA_DIR / "olist_products_dataset.csv"
    )

    sellers = pd.read_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv"
    )

    category_translation = pd.read_csv(
        RAW_DATA_DIR / "product_category_name_translation.csv"
    )

    return (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        sellers,
        category_translation
    )


# =========================================================
# VALIDATE FOREIGN KEY
# =========================================================

def validate_relationship(
    child_df,
    child_column,
    parent_df,
    parent_column
):
    """
    Check whether every value in the child table
    exists in the parent table.
    """

    child_values = set(
        child_df[child_column].dropna()
    )

    parent_values = set(
        parent_df[parent_column].dropna()
    )

    missing_values = child_values - parent_values

    return missing_values


# =========================================================
# MAIN VALIDATION
# =========================================================

def main():

    (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        sellers,
        category_translation
    ) = load_datasets()


    print("=" * 70)
    print("OLIST SCHEMA VALIDATION")
    print("=" * 70)


    # -----------------------------------------------------
    # ORDERS → CUSTOMERS
    # -----------------------------------------------------

    missing_customers = validate_relationship(
        orders,
        "customer_id",
        customers,
        "customer_id"
    )

    print("\nORDERS → CUSTOMERS")

    print(
        f"Missing customer IDs: {len(missing_customers)}"
    )


    # -----------------------------------------------------
    # ORDER ITEMS → ORDERS
    # -----------------------------------------------------

    missing_orders = validate_relationship(
        order_items,
        "order_id",
        orders,
        "order_id"
    )

    print("\nORDER ITEMS → ORDERS")

    print(
        f"Missing order IDs: {len(missing_orders)}"
    )


    # -----------------------------------------------------
    # ORDER ITEMS → PRODUCTS
    # -----------------------------------------------------

    missing_products = validate_relationship(
        order_items,
        "product_id",
        products,
        "product_id"
    )

    print("\nORDER ITEMS → PRODUCTS")

    print(
        f"Missing product IDs: {len(missing_products)}"
    )


    # -----------------------------------------------------
    # ORDER ITEMS → SELLERS
    # -----------------------------------------------------

    missing_sellers = validate_relationship(
        order_items,
        "seller_id",
        sellers,
        "seller_id"
    )

    print("\nORDER ITEMS → SELLERS")

    print(
        f"Missing seller IDs: {len(missing_sellers)}"
    )


    # -----------------------------------------------------
    # PAYMENTS → ORDERS
    # -----------------------------------------------------

    missing_payment_orders = validate_relationship(
        payments,
        "order_id",
        orders,
        "order_id"
    )

    print("\nPAYMENTS → ORDERS")

    print(
        f"Missing order IDs: {len(missing_payment_orders)}"
    )


    # -----------------------------------------------------
    # REVIEWS → ORDERS
    # -----------------------------------------------------

    missing_review_orders = validate_relationship(
        reviews,
        "order_id",
        orders,
        "order_id"
    )

    print("\nREVIEWS → ORDERS")

    print(
        f"Missing order IDs: {len(missing_review_orders)}"
    )


    # -----------------------------------------------------
    # PRODUCTS → CATEGORY TRANSLATION
    # -----------------------------------------------------

    product_categories = set(
        products["product_category_name"]
        .dropna()
    )

    translated_categories = set(
        category_translation["product_category_name"]
        .dropna()
    )

    missing_categories = (
        product_categories - translated_categories
    )

    print("\nPRODUCTS → CATEGORY TRANSLATION")

    print(
        f"Categories without translation: "
        f"{len(missing_categories)}"
    )

    print("\nCategories without translation:")

    for category in sorted(missing_categories):
        print(f"  - {category}")


    print("\n" + "=" * 70)
    print("SCHEMA VALIDATION COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()