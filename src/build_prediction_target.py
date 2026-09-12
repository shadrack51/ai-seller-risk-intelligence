from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_master.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_prediction_dataset.csv"
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("BUILDING PREDICTION TARGET")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

print("\nLoaded seller master dataset")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ==========================================================
# CREATE TARGET
# ==========================================================

print("\n" + "=" * 70)
print("CREATING BUSINESS RISK TARGET")
print("=" * 70)

risk_condition = (
    (df["late_delivery_rate"] >= 20)
    |
    (df["average_review_score"] < 3.5)
    |
    (df["catalog_completeness"] < 80)
)

df["at_risk"] = risk_condition.astype(int)

# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

print("\nTarget distribution:")

distribution = (
    df["at_risk"]
    .value_counts()
    .sort_index()
)

print(distribution)

print("\nTarget percentages:")

percentages = (
    df["at_risk"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(percentages.round(2))

# ==========================================================
# SEGMENT BREAKDOWN
# ==========================================================

if "seller_segment" in df.columns:

    print("\n" + "=" * 70)
    print("TARGET BY SEGMENT")
    print("=" * 70)

    segment_breakdown = pd.crosstab(
        df["seller_segment"],
        df["at_risk"],
        normalize="index"
    ) * 100

    print(segment_breakdown.round(2))

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPrediction dataset saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("PREDICTION TARGET COMPLETE")
print("=" * 70)