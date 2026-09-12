from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



# PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "seller_master.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "seller_segments.csv"



# CONFIGURATION

RANDOM_STATE = 42

# We will test several possible cluster counts.
K_VALUES = range(2, 9)



# LOAD DATA

print("=" * 70)
print("SELLER SEGMENTATION ANALYSIS")
print("=" * 70)

print()
print("Loading seller master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")



# SELECT FEATURES


print()
print("=" * 70)
print("SELECTING SEGMENTATION FEATURES")
print("=" * 70)

# These features describe seller scale, commercial activity,
# catalog quality, delivery performance and customer satisfaction.

segmentation_features = [
    "total_products",
    "incomplete_products",
    "catalog_completeness",
    "total_orders",
    "units_sold",
    "total_revenue",
    "average_item_price",
    "average_freight",
    "average_delivery_days",
    "average_delivery_delay_days",
    "late_delivery_rate",
    "average_review_score",
    "review_count",
    "low_review_rate",
]


print()
print("Features used for clustering:")

for feature in segmentation_features:
    print(f" - {feature}")


# CHECK FEATURES


missing_features = [
    feature
    for feature in segmentation_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )



# PREPARE MODEL DATA

print()
print("=" * 70)
print("PREPARING CLUSTERING DATA")
print("=" * 70)

X = df[segmentation_features].copy()


print()
print("Missing values before imputation:")

missing_summary = X.isna().sum()

print(
    missing_summary[
        missing_summary > 0
    ]
)



# Median imputation


# Some sellers have no delivered orders/reviews.
# Their performance fields are therefore missing.
#
# We use median imputation so that those sellers remain in
# the segmentation dataset rather than being discarded.

for column in X.columns:
    if X[column].isna().any():
        X[column] = X[column].fillna(
            X[column].median()
        )


print()
print("Missing values after imputation:")

remaining_missing = X.isna().sum().sum()

print(f"Total missing values: {remaining_missing}")


# LOG TRANSFORMATION
print()
print("=" * 70)
print("APPLYING LOG TRANSFORMATION")
print("=" * 70)

# Seller revenue, orders, units and review counts are highly
# right-skewed.
#
# log1p reduces the influence of extremely large sellers while
# preserving the information contained in the data.

log_features = [
    "total_products",
    "incomplete_products",
    "total_orders",
    "units_sold",
    "total_revenue",
    "average_item_price",
    "average_freight",
    "review_count",
]


for column in log_features:
    X[column] = np.log1p(
        X[column].clip(lower=0)
    )


print("Log transformation applied to:")

for feature in log_features:
    print(f" - {feature}")



# STANDARDIZATION

print()
print("=" * 70)
print("STANDARDIZING FEATURES")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(
    f"Scaled feature matrix shape: {X_scaled.shape}"
)


# TEST DIFFERENT NUMBERS OF CLUSTERS

print()
print("=" * 70)
print("TESTING CLUSTER COUNTS")
print("=" * 70)

results = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    inertia = model.inertia_

    results.append(
        {
            "k": k,
            "silhouette_score": silhouette,
            "inertia": inertia,
        }
    )

    print(
        f"K={k} | "
        f"Silhouette={silhouette:.4f} | "
        f"Inertia={inertia:.2f}"
    )


results_df = pd.DataFrame(results)


# SELECT BEST K

best_row = results_df.loc[
    results_df["silhouette_score"].idxmax()
]

best_k = int(best_row["k"])

best_silhouette = float(
    best_row["silhouette_score"]
)

print()
print("=" * 70)
print("BEST CLUSTER CONFIGURATION")
print("=" * 70)

print(f"Best number of clusters: {best_k}")
print(
    f"Best silhouette score: {best_silhouette:.4f}"
)


# TRAIN FINAL MODEL
print()
print("=" * 70)
print("TRAINING FINAL SELLER SEGMENTATION MODEL")
print("=" * 70)

final_model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=20,
)

df["seller_segment"] = final_model.fit_predict(
    X_scaled
)



# SEGMENT DISTRIBUTION
print()
print("=" * 70)
print("SELLER SEGMENT DISTRIBUTION")
print("=" * 70)

segment_counts = (
    df["seller_segment"]
    .value_counts()
    .sort_index()
)

print(segment_counts)

print()
print("Segment percentages:")

segment_percentages = (
    df["seller_segment"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print(
    segment_percentages.round(2)
)



# SEGMENT PROFILE


print()
print("=" * 70)
print("SELLER SEGMENT PROFILES")
print("=" * 70)

profile_columns = [
    "total_products",
    "incomplete_products",
    "catalog_completeness",
    "total_orders",
    "units_sold",
    "total_revenue",
    "average_item_price",
    "average_freight",
    "average_delivery_days",
    "average_delivery_delay_days",
    "late_delivery_rate",
    "average_review_score",
    "review_count",
    "low_review_rate",
]


segment_profile = (
    df.groupby("seller_segment")[profile_columns]
    .mean()
    .round(2)
)

print(segment_profile)



# SEGMENT REVENUE


print()
print("=" * 70)
print("SEGMENT COMMERCIAL PERFORMANCE")
print("=" * 70)

segment_commercial = (
    df.groupby("seller_segment")
    .agg(
        sellers=("seller_id", "count"),
        total_revenue=("total_revenue", "sum"),
        total_orders=("total_orders", "sum"),
        units_sold=("units_sold", "sum"),
        average_revenue=("total_revenue", "mean"),
        average_orders=("total_orders", "mean"),
        average_units_sold=("units_sold", "mean"),
    )
    .round(2)
)

print(segment_commercial)



# SEGMENT RISK

print()
print("=" * 70)
print("SEGMENT RISK DISTRIBUTION")
print("=" * 70)

risk_by_segment = pd.crosstab(
    df["seller_segment"],
    df["risk_category"],
)

print(risk_by_segment)



# HIGH-RISK SELLERS BY SEGMENT


print()
print("=" * 70)
print("HIGH-RISK SELLERS BY SEGMENT")
print("=" * 70)

high_risk_by_segment = (
    df[df["risk_category"] == "HIGH"]
    .groupby("seller_segment")
    .size()
)

print(high_risk_by_segment)



# SAVE RESULTS


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)



# SAVE CLUSTER EVALUATION


evaluation_file = (
    OUTPUT_DIR /
    "seller_segmentation_evaluation.csv"
)

results_df.to_csv(
    evaluation_file,
    index=False
)



# FINAL SUMMARY


print()
print("=" * 70)
print("SELLER SEGMENTATION COMPLETE")
print("=" * 70)

print()
print(f"Number of sellers: {len(df):,}")
print(f"Number of segments: {best_k}")
print(
    f"Silhouette score: {best_silhouette:.4f}"
)

print()
print("Seller segmentation dataset saved to:")

print(OUTPUT_FILE)

print()
print("Cluster evaluation saved to:")

print(evaluation_file)

print()
print("=" * 70)