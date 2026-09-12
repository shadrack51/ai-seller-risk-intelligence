from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_FILE = PROCESSED_DATA_DIR / "seller_master.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SELLER MASTER DATA QUALITY GATE")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)


print()
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 1. DUPLICATE SELLER IDS
# ============================================================

print()
print("=" * 70)
print("1. DUPLICATE SELLER IDS")
print("=" * 70)

duplicate_sellers = df["seller_id"].duplicated().sum()

print(f"Duplicate seller IDs: {duplicate_sellers}")


# ============================================================
# 2. MISSING VALUES
# ============================================================

print()
print("=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing)


# ============================================================
# 3. CATALOG COMPLETENESS
# ============================================================

print()
print("=" * 70)
print("3. CATALOG COMPLETENESS VALIDATION")
print("=" * 70)

invalid_catalog = df[
    (df["catalog_completeness"] < 0)
    | (df["catalog_completeness"] > 100)
]

print(
    f"Invalid catalog completeness values: "
    f"{len(invalid_catalog)}"
)


# ============================================================
# 4. DELIVERY RATES
# ============================================================

print()
print("=" * 70)
print("4. DELIVERY RATE VALIDATION")
print("=" * 70)

invalid_late_rate = df[
    (df["late_delivery_rate"] < 0)
    | (df["late_delivery_rate"] > 100)
]

invalid_low_review_rate = df[
    (df["low_review_rate"] < 0)
    | (df["low_review_rate"] > 100)
]

print(
    f"Invalid late delivery rates: "
    f"{len(invalid_late_rate)}"
)

print(
    f"Invalid low review rates: "
    f"{len(invalid_low_review_rate)}"
)


# ============================================================
# 5. REVIEW SCORE
# ============================================================

print()
print("=" * 70)
print("5. REVIEW SCORE VALIDATION")
print("=" * 70)

invalid_reviews = df[
    (df["average_review_score"] < 1)
    | (df["average_review_score"] > 5)
]

print(
    f"Invalid average review scores: "
    f"{len(invalid_reviews)}"
)


# ============================================================
# 6. FINANCIAL VALUES
# ============================================================

print()
print("=" * 70)
print("6. FINANCIAL VALUE VALIDATION")
print("=" * 70)

negative_revenue = df[df["total_revenue"] < 0]

negative_price = df[df["average_item_price"] < 0]

negative_freight = df[df["average_freight"] < 0]

print(
    f"Negative total revenue: "
    f"{len(negative_revenue)}"
)

print(
    f"Negative average item price: "
    f"{len(negative_price)}"
)

print(
    f"Negative average freight: "
    f"{len(negative_freight)}"
)


# ============================================================
# 7. COUNT VALIDATION
# ============================================================

print()
print("=" * 70)
print("7. COUNT VALIDATION")
print("=" * 70)

count_columns = [
    "total_products",
    "incomplete_products",
    "total_orders",
    "units_sold",
    "delivered_orders",
    "review_count",
    "low_review_count"
]

for column in count_columns:

    invalid_count = df[df[column] < 0]

    print(
        f"{column}: "
        f"{len(invalid_count)} invalid negative values"
    )


# ============================================================
# 8. LOGICAL CONSISTENCY
# ============================================================

print()
print("=" * 70)
print("8. LOGICAL CONSISTENCY CHECKS")
print("=" * 70)


# Incomplete products cannot exceed total products

invalid_products = df[
    df["incomplete_products"] > df["total_products"]
]

print(
    "Incomplete products > total products: "
    f"{len(invalid_products)}"
)


# Low review count cannot exceed review count

invalid_reviews_count = df[
    df["low_review_count"] > df["review_count"]
]

print(
    "Low reviews > total reviews: "
    f"{len(invalid_reviews_count)}"
)


# Delivered orders cannot exceed total orders

invalid_delivered_orders = df[
    df["delivered_orders"] > df["total_orders"]
]

print(
    "Delivered orders > total orders: "
    f"{len(invalid_delivered_orders)}"
)


# Units sold should not be lower than total orders
# because an order may contain multiple units.

invalid_units = df[
    df["units_sold"] < df["total_orders"]
]

print(
    "Units sold < total orders: "
    f"{len(invalid_units)}"
)


# ============================================================
# 9. DELIVERY DAYS
# ============================================================

print()
print("=" * 70)
print("9. DELIVERY TIME VALIDATION")
print("=" * 70)

negative_delivery = df[
    df["average_delivery_days"] < 0
]

print(
    f"Negative average delivery days: "
    f"{len(negative_delivery)}"
)


# ============================================================
# 10. CONSTANT FEATURES
# ============================================================

print()
print("=" * 70)
print("10. CONSTANT FEATURE CHECK")
print("=" * 70)

constant_features = []

for column in df.columns:

    if df[column].nunique(dropna=False) <= 1:
        constant_features.append(column)


if constant_features:

    print("Constant features found:")

    for column in constant_features:
        print(f" - {column}")

else:

    print("No constant features found.")


# ============================================================
# FINAL QUALITY SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL DATA QUALITY SUMMARY")
print("=" * 70)

quality_checks = {
    "Duplicate seller IDs": duplicate_sellers,
    "Invalid catalog completeness": len(invalid_catalog),
    "Invalid late delivery rates": len(invalid_late_rate),
    "Invalid low review rates": len(invalid_low_review_rate),
    "Invalid review scores": len(invalid_reviews),
    "Negative revenue": len(negative_revenue),
    "Negative prices": len(negative_price),
    "Negative freight": len(negative_freight),
    "Invalid product counts": len(invalid_products),
    "Invalid review counts": len(invalid_reviews_count),
    "Invalid delivered orders": len(invalid_delivered_orders),
    "Invalid units sold": len(invalid_units),
    "Negative delivery days": len(negative_delivery),
}


print()

for check, result in quality_checks.items():

    status = "PASS" if result == 0 else "CHECK"

    print(
        f"{status:6} | "
        f"{check}: {result}"
    )


print()
print("=" * 70)
print("MASTER DATA QUALITY GATE COMPLETE")
print("=" * 70)