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

    orders = pd.read_csv(
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    )

    order_items = pd.read_csv(
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    )

    reviews = pd.read_csv(
        RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
    )

    sellers = pd.read_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv"
    )

    return orders, order_items, reviews, sellers


# =========================================================
# PREPARE ORDER DATES
# =========================================================

def prepare_orders(orders):

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce"
        )

    return orders


# =========================================================
# CALCULATE DELIVERY PERFORMANCE
# =========================================================

def calculate_delivery_features(orders, order_items):

    delivered_orders = orders[
        orders["order_status"] == "delivered"
    ].copy()

    delivered_orders["delivery_days"] = (
        delivered_orders["order_delivered_customer_date"]
        - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    delivered_orders["estimated_delivery_days"] = (
        delivered_orders["order_estimated_delivery_date"]
        - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    delivered_orders["delivery_delay_days"] = (
        delivered_orders["order_delivered_customer_date"]
        - delivered_orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    delivered_orders["late_delivery"] = (
        delivered_orders["delivery_delay_days"] > 0
    ).astype(int)

    seller_orders = order_items[
        ["order_id", "seller_id"]
    ].drop_duplicates()

    seller_delivery = seller_orders.merge(
        delivered_orders[
            [
                "order_id",
                "delivery_days",
                "estimated_delivery_days",
                "delivery_delay_days",
                "late_delivery"
            ]
        ],
        on="order_id",
        how="inner"
    )

    seller_delivery_features = (
        seller_delivery
        .groupby("seller_id")
        .agg(
            average_delivery_days=(
                "delivery_days",
                "mean"
            ),
            average_estimated_delivery_days=(
                "estimated_delivery_days",
                "mean"
            ),
            average_delivery_delay_days=(
                "delivery_delay_days",
                "mean"
            ),
            late_delivery_rate=(
                "late_delivery",
                "mean"
            ),
            delivered_orders=(
                "order_id",
                "count"
            )
        )
        .reset_index()
    )

    seller_delivery_features["late_delivery_rate"] *= 100

    return seller_delivery_features


# =========================================================
# CALCULATE REVIEW PERFORMANCE
# =========================================================

def calculate_review_features(reviews, order_items):

    seller_orders = order_items[
        ["order_id", "seller_id"]
    ].drop_duplicates()

    seller_reviews = seller_orders.merge(
        reviews[
            [
                "order_id",
                "review_score"
            ]
        ],
        on="order_id",
        how="inner"
    )

    seller_reviews["low_review"] = (
        seller_reviews["review_score"] <= 2
    ).astype(int)

    seller_review_features = (
        seller_reviews
        .groupby("seller_id")
        .agg(
            average_review_score=(
                "review_score",
                "mean"
            ),
            review_count=(
                "review_score",
                "count"
            ),
            low_review_count=(
                "low_review",
                "sum"
            )
        )
        .reset_index()
    )

    seller_review_features["low_review_rate"] = (
        seller_review_features["low_review_count"]
        / seller_review_features["review_count"]
        * 100
    )

    return seller_review_features


# =========================================================
# BUILD FINAL SELLER PERFORMANCE TABLE
# =========================================================

def build_performance_table(
    sellers,
    delivery_features,
    review_features
):

    seller_performance = sellers.merge(
        delivery_features,
        on="seller_id",
        how="left"
    )

    seller_performance = seller_performance.merge(
        review_features,
        on="seller_id",
        how="left"
    )

    return seller_performance


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("BUILDING SELLER PERFORMANCE FEATURES")
    print("=" * 70)

    orders, order_items, reviews, sellers = load_data()

    print("\nPreparing order dates...")

    orders = prepare_orders(orders)

    print("Calculating delivery performance...")

    delivery_features = calculate_delivery_features(
        orders,
        order_items
    )

    print("Calculating customer review performance...")

    review_features = calculate_review_features(
        reviews,
        order_items
    )

    print("Building seller performance table...")

    seller_performance = build_performance_table(
        sellers,
        delivery_features,
        review_features
    )

    print("\nNumber of sellers:", len(seller_performance))
    print(
        "Number of features:",
        len(seller_performance.columns)
    )

    print("\nFeature table preview:")

    print(
        seller_performance.head(10).to_string(
            index=False
        )
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "seller_performance.csv"
    )

    seller_performance.to_csv(
        output_path,
        index=False
    )

    print("\nSeller performance table saved to:")
    print(output_path)

    print("\n" + "=" * 70)
    print("SELLER PERFORMANCE FEATURE TABLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()