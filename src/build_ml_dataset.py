from pathlib import Path

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_prediction_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seller_ml_dataset.csv"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TARGET = "at_risk"


# ==========================================================
# IDENTIFIER FEATURES
# ==========================================================
# These identify sellers but are not useful predictors for
# our first machine-learning model.

IDENTIFIER_COLUMNS = [
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state",
]


# ==========================================================
# TARGET LEAKAGE FEATURES
# ==========================================================
# These features were directly used to create the at_risk
# target.
#
# Target definition:
#
# late_delivery_rate >= 20
# OR average_review_score < 3.5
# OR catalog_completeness < 80
#
# Therefore they must not be used as predictors.

LEAKAGE_COLUMNS = [
    "catalog_completeness",
    "late_delivery_rate",
    "average_review_score",
]


# ==========================================================
# DERIVED BUSINESS LABELS
# ==========================================================
# risk_category was created by our previous risk-scoring
# system. It is a business label rather than an independent
# predictive feature.
#
# We remove it to prevent the ML model from learning our
# previous risk classification instead of learning underlying
# seller behavior.

DERIVED_LABEL_COLUMNS = [
    "risk_category",
]


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("BUILDING MACHINE LEARNING DATASET")
print("=" * 70)

print("\nLoading prediction dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ==========================================================
# DATA VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)


# ----------------------------------------------------------
# Target validation
# ----------------------------------------------------------

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

print(f"Target column: {TARGET} -> PASS")


# ----------------------------------------------------------
# Identifier validation
# ----------------------------------------------------------

missing_identifier_columns = [
    column
    for column in IDENTIFIER_COLUMNS
    if column not in df.columns
]

if missing_identifier_columns:
    raise ValueError(
        "Missing identifier columns: "
        + ", ".join(missing_identifier_columns)
    )

print("Identifier columns -> PASS")


# ----------------------------------------------------------
# Leakage validation
# ----------------------------------------------------------

missing_leakage_columns = [
    column
    for column in LEAKAGE_COLUMNS
    if column not in df.columns
]

if missing_leakage_columns:
    raise ValueError(
        "Missing leakage columns: "
        + ", ".join(missing_leakage_columns)
    )

print("Leakage columns -> PASS")


# ----------------------------------------------------------
# Derived label validation
# ----------------------------------------------------------

missing_derived_columns = [
    column
    for column in DERIVED_LABEL_COLUMNS
    if column not in df.columns
]

if missing_derived_columns:
    raise ValueError(
        "Missing derived label columns: "
        + ", ".join(missing_derived_columns)
    )

print("Derived business label columns -> PASS")


# ==========================================================
# TARGET VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")

target_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(
    target_percentages.round(2)
)


# ----------------------------------------------------------
# Validate target values
# ----------------------------------------------------------

invalid_targets = df[
    ~df[TARGET].isin([0, 1])
]

print(
    f"\nInvalid target values: {len(invalid_targets)}"
)

if len(invalid_targets) > 0:
    raise ValueError(
        "Target contains values other than 0 and 1."
    )


# ==========================================================
# SEPARATE TARGET FROM FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("SEPARATING TARGET FROM FEATURES")
print("=" * 70)

y = df[TARGET].copy()

X = df.drop(
    columns=[TARGET]
)

print(
    f"\nTarget shape: {y.shape}"
)

print(
    f"Feature table shape before cleaning: {X.shape}"
)


# ==========================================================
# REMOVE IDENTIFIERS
# ==========================================================

print("\n" + "=" * 70)
print("REMOVING IDENTIFIER FEATURES")
print("=" * 70)

print("\nRemoving:")

for column in IDENTIFIER_COLUMNS:
    print(f" - {column}")

X = X.drop(
    columns=IDENTIFIER_COLUMNS
)

print(
    f"\nFeature count after removing identifiers: "
    f"{X.shape[1]}"
)


# ==========================================================
# REMOVE TARGET LEAKAGE
# ==========================================================

print("\n" + "=" * 70)
print("REMOVING TARGET LEAKAGE")
print("=" * 70)

print(
    "\nThe following features directly contributed "
    "to the definition of 'at_risk':"
)

for column in LEAKAGE_COLUMNS:
    print(f" - {column}")

X = X.drop(
    columns=LEAKAGE_COLUMNS
)

print(
    f"\nFeature count after leakage removal: "
    f"{X.shape[1]}"
)


# ==========================================================
# REMOVE DERIVED BUSINESS LABELS
# ==========================================================

print("\n" + "=" * 70)
print("REMOVING DERIVED BUSINESS LABELS")
print("=" * 70)

print(
    "\nThe following columns were produced by previous "
    "business-analysis logic:"
)

for column in DERIVED_LABEL_COLUMNS:
    print(f" - {column}")

X = X.drop(
    columns=DERIVED_LABEL_COLUMNS
)

print(
    f"\nFeature count after removing derived labels: "
    f"{X.shape[1]}"
)


# ==========================================================
# FEATURE TYPE VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE TYPE VALIDATION")
print("=" * 70)

non_numeric_columns = (
    X
    .select_dtypes(exclude="number")
    .columns
    .tolist()
)

if non_numeric_columns:

    print("\nNon-numeric columns found:")

    for column in non_numeric_columns:
        print(f" - {column}")

    raise ValueError(
        "The ML dataset contains non-numeric features."
    )

else:

    print(
        "\nAll remaining features are numeric -> PASS"
    )


# ==========================================================
# DUPLICATE CHECK
# ==========================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_rows = X.duplicated().sum()

print(
    f"Duplicate feature rows: {duplicate_rows}"
)


# ==========================================================
# MISSING VALUE ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = X.isna().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) == 0:

    print("\nNo missing values found.")

else:

    print("\nFeatures containing missing values:")

    print(missing_values)


# ==========================================================
# BUILD FINAL ML DATASET
# ==========================================================

print("\n" + "=" * 70)
print("CREATING ML DATASET")
print("=" * 70)

ml_df = X.copy()

ml_df[TARGET] = y.values


# ==========================================================
# FINAL DATASET VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("FINAL ML DATASET VALIDATION")
print("=" * 70)

print(
    f"\nRows: {len(ml_df):,}"
)

print(
    f"Predictor features: {len(X.columns):,}"
)

print(
    f"Total columns including target: "
    f"{len(ml_df.columns):,}"
)


# ----------------------------------------------------------
# Display final predictors
# ----------------------------------------------------------

print("\nFinal predictor features:")

for column in X.columns:
    print(f" - {column}")


# ----------------------------------------------------------
# Leakage check
# ----------------------------------------------------------

remaining_leakage = [
    column
    for column in LEAKAGE_COLUMNS
    if column in ml_df.columns
]

if remaining_leakage:

    raise ValueError(
        "Target leakage detected: "
        + ", ".join(remaining_leakage)
    )

print(
    "\nLeakage check: PASS"
)


# ----------------------------------------------------------
# Identifier check
# ----------------------------------------------------------

remaining_identifiers = [
    column
    for column in IDENTIFIER_COLUMNS
    if column in ml_df.columns
]

if remaining_identifiers:

    raise ValueError(
        "Identifier columns still present: "
        + ", ".join(remaining_identifiers)
    )

print(
    "Identifier check: PASS"
)


# ----------------------------------------------------------
# Derived-label check
# ----------------------------------------------------------

remaining_derived_labels = [
    column
    for column in DERIVED_LABEL_COLUMNS
    if column in ml_df.columns
]

if remaining_derived_labels:

    raise ValueError(
        "Derived business labels still present: "
        + ", ".join(remaining_derived_labels)
    )

print(
    "Derived-label check: PASS"
)


# ----------------------------------------------------------
# Target check
# ----------------------------------------------------------

print(
    f"Target check: PASS -> {TARGET}"
)


# ==========================================================
# SAVE DATASET
# ==========================================================

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("MACHINE LEARNING DATASET SAVED")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("MACHINE LEARNING DATASET COMPLETE")
print("=" * 70)