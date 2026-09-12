from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_ml_dataset.csv"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_test.csv"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TARGET = "at_risk"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("CREATING TRAIN / TEST DATASETS")
print("=" * 70)

print("\nLoading machine learning dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ==========================================================
# DATA VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)


if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print(f"Target column: {TARGET} -> PASS")


duplicate_rows = df.duplicated().sum()

print(
    f"Duplicate rows: {duplicate_rows}"
)

if duplicate_rows > 0:
    raise ValueError(
        "Duplicate rows detected."
    )


missing_target = df[TARGET].isna().sum()

print(
    f"Missing target values: {missing_target}"
)

if missing_target > 0:
    raise ValueError(
        "Target contains missing values."
    )


# ==========================================================
# SEPARATE FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 70)
print("SEPARATING FEATURES AND TARGET")
print("=" * 70)

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print(
    f"\nFeature matrix shape: {X.shape}"
)

print(
    f"Target shape: {y.shape}"
)


# ==========================================================
# DISPLAY ORIGINAL TARGET DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("ORIGINAL TARGET DISTRIBUTION")
print("=" * 70)

original_counts = (
    y.value_counts()
    .sort_index()
)

original_percentages = (
    y.value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nCounts:")

print(
    original_counts
)

print("\nPercentages:")

print(
    original_percentages.round(2)
)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

print("\n" + "=" * 70)
print("PERFORMING TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"\nTest size: {TEST_SIZE:.0%}"
)

print(
    f"Training size: {1 - TEST_SIZE:.0%}"
)

print(
    f"Random state: {RANDOM_STATE}"
)

print(
    "\nStratification: ENABLED"
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ==========================================================
# REBUILD TRAIN AND TEST DATASETS
# ==========================================================

train_df = X_train.copy()

train_df[TARGET] = y_train.values


test_df = X_test.copy()

test_df[TARGET] = y_test.values


# ==========================================================
# DATASET SIZES
# ==========================================================

print("\n" + "=" * 70)
print("DATASET SIZES")
print("=" * 70)

print(
    f"\nTraining rows: {len(train_df):,}"
)

print(
    f"Testing rows: {len(test_df):,}"
)

print(
    f"Training features: {len(X_train.columns):,}"
)

print(
    f"Testing features: {len(X_test.columns):,}"
)


# ==========================================================
# TRAINING TARGET DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING TARGET DISTRIBUTION")
print("=" * 70)

train_counts = (
    y_train.value_counts()
    .sort_index()
)

train_percentages = (
    y_train.value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nCounts:")

print(
    train_counts
)

print("\nPercentages:")

print(
    train_percentages.round(2)
)


# ==========================================================
# TESTING TARGET DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("TESTING TARGET DISTRIBUTION")
print("=" * 70)

test_counts = (
    y_test.value_counts()
    .sort_index()
)

test_percentages = (
    y_test.value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nCounts:")

print(
    test_counts
)

print("\nPercentages:")

print(
    test_percentages.round(2)
)


# ==========================================================
# STRATIFICATION VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("STRATIFICATION VALIDATION")
print("=" * 70)

original_rate = y.mean()

train_rate = y_train.mean()

test_rate = y_test.mean()


print(
    f"\nOriginal at-risk rate: "
    f"{original_rate * 100:.2f}%"
)

print(
    f"Training at-risk rate: "
    f"{train_rate * 100:.2f}%"
)

print(
    f"Testing at-risk rate: "
    f"{test_rate * 100:.2f}%"
)


train_difference = abs(
    original_rate - train_rate
)

test_difference = abs(
    original_rate - test_rate
)


print(
    f"\nTraining difference: "
    f"{train_difference * 100:.2f} percentage points"
)

print(
    f"Testing difference: "
    f"{test_difference * 100:.2f} percentage points"
)


if train_difference <= 0.01 and test_difference <= 0.01:

    print(
        "\nStratification check: PASS"
    )

else:

    print(
        "\nWarning: class distribution differs "
        "slightly between datasets."
    )


# ==========================================================
# FEATURE CONSISTENCY CHECK
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE CONSISTENCY CHECK")
print("=" * 70)

train_features = list(
    X_train.columns
)

test_features = list(
    X_test.columns
)


if train_features == test_features:

    print(
        "\nTraining and testing features match -> PASS"
    )

else:

    raise ValueError(
        "Training and testing feature columns do not match."
    )


# ==========================================================
# SELLER COUNT CHECK
# ==========================================================

print("\n" + "=" * 70)
print("ROW COUNT VALIDATION")
print("=" * 70)

total_rows = (
    len(train_df)
    + len(test_df)
)

print(
    f"\nOriginal rows: {len(df):,}"
)

print(
    f"Train + test rows: {total_rows:,}"
)


if total_rows == len(df):

    print(
        "Row conservation check: PASS"
    )

else:

    raise ValueError(
        "Train + test rows do not equal original dataset."
    )


# ==========================================================
# SAVE TRAINING DATA
# ==========================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)


# ==========================================================
# SAVE TESTING DATA
# ==========================================================

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("TRAIN / TEST DATASETS SAVED")
print("=" * 70)

print(
    f"\nTraining dataset:\n{TRAIN_FILE}"
)

print(
    f"\nTesting dataset:\n{TEST_FILE}"
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT COMPLETE")
print("=" * 70)