from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# LOAD DATASETS
# ============================================================

seller_features = pd.read_csv(
    PROCESSED_DATA_DIR / "seller_features.csv"
)

seller_risk = pd.read_csv(
    PROCESSED_DATA_DIR / "seller_risk_scores.csv"
)

seller_performance = pd.read_csv(
    PROCESSED_DATA_DIR / "seller_performance.csv"
)


print("=" * 70)
print("BUILDING SELLER MASTER DATASET")
print("=" * 70)

print()

print(f"Seller feature table:")
print(f"Rows: {len(seller_features):,}")
print(f"Columns: {len(seller_features.columns)}")

print()

print(f"Seller risk table:")
print(f"Rows: {len(seller_risk):,}")
print(f"Columns: {len(seller_risk.columns)}")

print()

print(f"Seller performance table:")
print(f"Rows: {len(seller_performance):,}")
print(f"Columns: {len(seller_performance.columns)}")


# ============================================================
# SELECT RISK FEATURES
# ============================================================

risk_columns = [
    "seller_id",
    "catalog_risk",
    "commercial_exposure",
    "order_exposure",
    "seller_risk_score",
    "risk_category"
]

seller_risk = seller_risk[risk_columns]


# ============================================================
# SELECT PERFORMANCE FEATURES
# ============================================================

performance_columns = [
    "seller_id",
    "average_delivery_days",
    "average_estimated_delivery_days",
    "average_delivery_delay_days",
    "late_delivery_rate",
    "delivered_orders",
    "average_review_score",
    "review_count",
    "low_review_count",
    "low_review_rate"
]

seller_performance = seller_performance[performance_columns]


# ============================================================
# MERGE SELLER FEATURES + RISK
# ============================================================

master = seller_features.merge(
    seller_risk,
    on="seller_id",
    how="left"
)


# ============================================================
# MERGE PERFORMANCE FEATURES
# ============================================================

master = master.merge(
    seller_performance,
    on="seller_id",
    how="left"
)


# ============================================================
# VALIDATE SELLER UNIQUENESS
# ============================================================

duplicate_sellers = master["seller_id"].duplicated().sum()


print()
print("=" * 70)
print("MASTER DATASET VALIDATION")
print("=" * 70)

print(f"Duplicate seller IDs: {duplicate_sellers}")


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_values = master.isna().sum()

missing_values = missing_values[missing_values > 0]


print()

print("Columns with missing values:")

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values)


# ============================================================
# CHECK COLUMN COUNT
# ============================================================

print()
print("=" * 70)
print("SELLER MASTER DATASET")
print("=" * 70)

print(f"Number of sellers: {len(master):,}")
print(f"Number of features: {len(master.columns)}")


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print()
print("Columns:")

for column in master.columns:
    print(f" - {column}")


# ============================================================
# SAVE MASTER DATASET
# ============================================================

output_path = PROCESSED_DATA_DIR / "seller_master.csv"

master.to_csv(
    output_path,
    index=False
)


print()
print("Seller master dataset saved to:")
print(output_path)

print()
print("=" * 70)
print("SELLER MASTER DATASET COMPLETE")
print("=" * 70)