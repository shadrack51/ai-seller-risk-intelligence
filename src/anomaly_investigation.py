from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# LOAD ORDERS
# =========================================================

def load_orders():

    file_path = RAW_DATA_DIR / "olist_orders_dataset.csv"

    return pd.read_csv(file_path)


# =========================================================
# INVESTIGATE DELIVERED ORDERS
# =========================================================

def investigate_delivered_orders(df):

    print("=" * 70)
    print("DELIVERED ORDERS WITH MISSING DATES")
    print("=" * 70)


    # -----------------------------------------------------
    # Delivered orders with missing approval date
    # -----------------------------------------------------

    missing_approval = df[
        (df["order_status"] == "delivered")
        & (df["order_approved_at"].isna())
    ]

    print("\nDelivered orders with missing approval date:")

    print(
        missing_approval[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nCount: {len(missing_approval)}"
    )


    # -----------------------------------------------------
    # Delivered orders with missing carrier date
    # -----------------------------------------------------

    missing_carrier = df[
        (df["order_status"] == "delivered")
        & (df["order_delivered_carrier_date"].isna())
    ]

    print(
        "\nDelivered orders with missing carrier date:"
    )

    print(
        missing_carrier[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nCount: {len(missing_carrier)}"
    )


    # -----------------------------------------------------
    # Delivered orders with missing customer delivery date
    # -----------------------------------------------------

    missing_delivery = df[
        (df["order_status"] == "delivered")
        & (df["order_delivered_customer_date"].isna())
    ]

    print(
        "\nDelivered orders with missing customer delivery date:"
    )

    print(
        missing_delivery[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nCount: {len(missing_delivery)}"
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    orders = load_orders()

    investigate_delivered_orders(orders)

    print("\n" + "=" * 70)
    print("ANOMALY INVESTIGATION COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()