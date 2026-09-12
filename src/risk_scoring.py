"""
Seller Risk Scoring

Purpose:
    Apply the trained Random Forest model to all sellers and create
    an operational seller risk dataset.

Inputs:
    - Trained Random Forest model
    - Seller prediction dataset

Outputs:
    - Seller risk scores
    - Risk levels
    - Business actions
    - Model risk decisions
    - Seller risk rankings
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = (
    PROCESSED_DATA_DIR
    / "random_forest_model.joblib"
)

PREDICTION_DATASET_PATH = (
    PROCESSED_DATA_DIR
    / "seller_prediction_dataset.csv"
)

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "seller_risk_scores.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "total_products",
    "incomplete_products",
    "total_orders",
    "units_sold",
    "total_revenue",
    "average_item_price",
    "average_freight",
    "catalog_risk",
    "commercial_exposure",
    "order_exposure",
    "seller_risk_score",
    "average_delivery_days",
    "average_estimated_delivery_days",
    "average_delivery_delay_days",
    "delivered_orders",
    "review_count",
    "low_review_count",
    "low_review_rate",
]


TARGET_COLUMN = "at_risk"

IDENTIFIER_COLUMNS = [
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state",
]


# ============================================================
# BUSINESS THRESHOLD
# ============================================================

OPERATING_THRESHOLD = 0.35


# ============================================================
# RISK LEVEL FUNCTION
# ============================================================

def assign_risk_level(risk_score):
    """
    Convert a 0-100 risk score into a business risk level.
    """

    if risk_score >= 75:
        return "CRITICAL"

    elif risk_score >= 50:
        return "HIGH"

    elif risk_score >= 25:
        return "MODERATE"

    else:
        return "LOW"


# ============================================================
# BUSINESS ACTION FUNCTION
# ============================================================

def assign_business_action(risk_level):
    """
    Assign a recommended business action based on risk level.
    """

    action_mapping = {
        "LOW": "Monitor seller",
        "MODERATE": "Review seller performance",
        "HIGH": "Intervene and investigate",
        "CRITICAL": "Immediate intervention",
    }

    return action_mapping[risk_level]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SELLER RISK SCORING")
    print("=" * 70)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading trained Random Forest model...")

    model = joblib.load(MODEL_PATH)

    print("Random Forest model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")

    # ========================================================
    # LOAD PREDICTION DATASET
    # ========================================================

    print("\nLoading seller prediction dataset...")

    df = pd.read_csv(PREDICTION_DATASET_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ========================================================
    # DATA VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    print(f"Target column: {TARGET_COLUMN} -> PASS")

    missing_target = df[TARGET_COLUMN].isna().sum()

    print(
        f"Missing target values: {missing_target}"
    )

    if missing_target > 0:
        raise ValueError(
            "Target contains missing values."
        )

    print("Missing target values -> PASS")

    # --------------------------------------------------------
    # IDENTIFIER VALIDATION
    # --------------------------------------------------------

    print("\nChecking seller identity columns...")

    missing_identifiers = [
        column
        for column in IDENTIFIER_COLUMNS
        if column not in df.columns
    ]

    if missing_identifiers:
        raise ValueError(
            "Missing identifier columns:\n"
            + "\n".join(missing_identifiers)
        )

    print("Seller identity columns -> PASS")

    # --------------------------------------------------------
    # DUPLICATE SELLER CHECK
    # --------------------------------------------------------

    duplicate_sellers = df["seller_id"].duplicated().sum()

    print(
        f"Duplicate seller IDs: {duplicate_sellers}"
    )

    if duplicate_sellers > 0:
        raise ValueError(
            "Duplicate seller IDs detected."
        )

    print("Seller uniqueness -> PASS")

    # ========================================================
    # PREPARING MODEL INPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPARING MODEL INPUT")
    print("=" * 70)

    # Keep seller identity outside the model matrix.
    seller_identity = df[
        [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ]
    ].copy()

    # Select ONLY the features used by the model.
    X = df[MODEL_FEATURES].copy()

    print(
        f"Feature matrix shape: {X.shape}"
    )

    # ========================================================
    # MODEL FEATURE VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL FEATURE VALIDATION")
    print("=" * 70)

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Expected the trained model to be a sklearn Pipeline."
        )

    print("Pipeline steps:")

    for step_name, step_object in model.named_steps.items():
        print(
            f" - {step_name}: "
            f"{step_object.__class__.__name__}"
        )

    print("Required pipeline steps -> PASS")

    # --------------------------------------------------------
    # GET MODEL FEATURE NAMES
    # --------------------------------------------------------

    model_feature_names = None

    for step_name, step_object in model.named_steps.items():

        if hasattr(step_object, "feature_names_in_"):

            model_feature_names = list(
                step_object.feature_names_in_
            )

            print(
                "\nFeature names obtained from "
                f"pipeline step: {step_name}"
            )

            break

    if model_feature_names is None:
        raise ValueError(
            "Could not obtain feature names from the model."
        )

    print(
        f"Model features: {len(model_feature_names)}"
    )

    print(
        f"Dataset features: {len(X.columns)}"
    )

    # --------------------------------------------------------
    # FEATURE SET CHECK
    # --------------------------------------------------------

    if set(model_feature_names) != set(X.columns):

        missing_from_dataset = [
            feature
            for feature in model_feature_names
            if feature not in X.columns
        ]

        extra_in_dataset = [
            feature
            for feature in X.columns
            if feature not in model_feature_names
        ]

        raise ValueError(
            "\nModel/dataset feature mismatch.\n"
            f"Missing from dataset: {missing_from_dataset}\n"
            f"Extra in dataset: {extra_in_dataset}"
        )

    print(
        "Model/dataset feature consistency -> PASS"
    )

    # --------------------------------------------------------
    # ALIGN FEATURE ORDER
    # --------------------------------------------------------

    X = X[model_feature_names]

    print(
        "Feature order aligned with model -> PASS"
    )

    # ========================================================
    # GENERATE RISK PROBABILITIES
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING RISK PROBABILITIES")
    print("=" * 70)

    print("Generating probability predictions...")

    probabilities = model.predict_proba(X)[:, 1]

    print(
        f"Risk probabilities generated: "
        f"{len(probabilities):,}"
    )

    print("\nFirst 10 risk probabilities:")

    print(
        np.round(probabilities[:10], 4)
    )

    # ========================================================
    # CREATE RISK SCORES
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING RISK SCORES")
    print("=" * 70)

    # Convert probability from 0-1 into a 0-100 score.

    risk_scores = probabilities * 100

    print("Risk score range:")

    print(
        f"Minimum: {risk_scores.min():.2f}"
    )

    print(
        f"Maximum: {risk_scores.max():.2f}"
    )

    print(
        f"Mean:    {risk_scores.mean():.2f}"
    )

    print(
        f"Median:  {np.median(risk_scores):.2f}"
    )

    # ========================================================
    # ASSIGN RISK LEVELS
    # ========================================================

    print("\n" + "=" * 70)
    print("ASSIGNING RISK LEVELS")
    print("=" * 70)

    risk_levels = [
        assign_risk_level(score)
        for score in risk_scores
    ]

    risk_level_series = pd.Series(
        risk_levels,
        name="risk_level"
    )

    print("\nRisk level distribution:")

    print(
        risk_level_series.value_counts()
    )

    print("\nRisk level percentages:")

    print(
        (
            risk_level_series
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )

    # ========================================================
    # BUSINESS ACTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("ASSIGNING BUSINESS ACTIONS")
    print("=" * 70)

    business_actions = [
        assign_business_action(level)
        for level in risk_levels
    ]

    print(
        "Business action mapping created -> PASS"
    )

    # ========================================================
    # MODEL RISK DECISION
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING MODEL RISK DECISION")
    print("=" * 70)

    print(
        f"Operating threshold: "
        f"{OPERATING_THRESHOLD:.2f}"
    )

    predicted_at_risk = (
        probabilities >= OPERATING_THRESHOLD
    ).astype(int)

    predicted_at_risk_series = pd.Series(
        predicted_at_risk,
        name="predicted_at_risk"
    )

    print("\nPredicted at-risk distribution:")

    print(
        predicted_at_risk_series.value_counts()
        .sort_index()
    )

    # ========================================================
    # BUILD SELLER RISK DATASET
    # ========================================================

    print("\n" + "=" * 70)
    print("BUILDING SELLER RISK DATASET")
    print("=" * 70)

    risk_df = seller_identity.copy()

    # Add model outputs.

    risk_df["risk_probability"] = probabilities

    risk_df["risk_score"] = risk_scores

    risk_df["risk_level"] = risk_levels

    risk_df["business_action"] = business_actions

    risk_df["predicted_at_risk"] = predicted_at_risk

    # Keep actual target for analysis.
    risk_df["actual_at_risk"] = df[TARGET_COLUMN].values

    # --------------------------------------------------------
    # RANK SELLERS
    # --------------------------------------------------------

    print("\nRanking sellers by risk...")

    risk_df = risk_df.sort_values(
        by="risk_score",
        ascending=False
    ).reset_index(drop=True)

    risk_df["risk_rank"] = (
        risk_df.index + 1
    )

    print(
        "Risk ranking created -> PASS"
    )

    # ========================================================
    # TOP 20 SELLERS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 20 HIGHEST-RISK SELLERS")
    print("=" * 70)

    display_columns = [
        "risk_rank",
        "seller_id",
        "seller_city",
        "seller_state",
        "risk_probability",
        "risk_score",
        "risk_level",
        "predicted_at_risk",
        "business_action",
    ]

    top_sellers = risk_df[
        display_columns
    ].head(20).copy()

    top_sellers["risk_probability"] = (
        top_sellers["risk_probability"]
        .round(4)
    )

    top_sellers["risk_score"] = (
        top_sellers["risk_score"]
        .round(2)
    )

    print(
        top_sellers.to_string(
            index=False
        )
    )

    # ========================================================
    # RISK LEVEL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("RISK LEVEL SUMMARY")
    print("=" * 70)

    risk_summary = (
        risk_df
        .groupby("risk_level")
        .agg(
            sellers=("seller_id", "count"),
            average_risk_score=("risk_score", "mean"),
            average_probability=(
                "risk_probability",
                "mean"
            ),
        )
        .reset_index()
    )

    risk_summary["seller_percentage"] = (
        risk_summary["sellers"]
        / len(risk_df)
        * 100
    )

    risk_summary = risk_summary.sort_values(
        by="average_risk_score",
        ascending=False
    )

    risk_summary[
        [
            "risk_level",
            "sellers",
            "seller_percentage",
            "average_risk_score",
            "average_probability",
        ]
    ] = risk_summary[
        [
            "risk_level",
            "sellers",
            "seller_percentage",
            "average_risk_score",
            "average_probability",
        ]
    ]

    print(
        risk_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # MODEL DECISION SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL DECISION SUMMARY")
    print("=" * 70)

    decision_summary = (
        risk_df["predicted_at_risk"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Not predicted at risk: "
        f"{decision_summary.get(0, 0):,}"
    )

    print(
        f"Predicted at risk: "
        f"{decision_summary.get(1, 0):,}"
    )

    print(
        f"Total sellers: "
        f"{len(risk_df):,}"
    )

    # ========================================================
    # HIGH-PRIORITY SELLERS
    # ========================================================

    print("\n" + "=" * 70)
    print("HIGH-PRIORITY SELLERS")
    print("=" * 70)

    high_priority = risk_df[
        risk_df["predicted_at_risk"] == 1
    ].copy()

    print(
        f"Sellers requiring attention: "
        f"{len(high_priority):,}"
    )

    print(
        f"Percentage requiring attention: "
        f"{len(high_priority) / len(risk_df) * 100:.2f}%"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RISK DATASET VALIDATION")
    print("=" * 70)

    if len(risk_df) != len(df):
        raise ValueError(
            "Risk dataset row count does not match input dataset."
        )

    print(
        "Row count consistency -> PASS"
    )

    if risk_df["seller_id"].duplicated().sum() > 0:
        raise ValueError(
            "Duplicate seller IDs found in risk dataset."
        )

    print(
        "Seller uniqueness -> PASS"
    )

    if risk_df["risk_score"].isna().sum() > 0:
        raise ValueError(
            "Missing risk scores detected."
        )

    print(
        "Risk score completeness -> PASS"
    )

    if risk_df["risk_level"].isna().sum() > 0:
        raise ValueError(
            "Missing risk levels detected."
        )

    print(
        "Risk level completeness -> PASS"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVING SELLER RISK DATASET")
    print("=" * 70)

    risk_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSeller risk dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("SELLER RISK SCORING COMPLETE")
    print("=" * 70)

    print(
        f"Total sellers scored: "
        f"{len(risk_df):,}"
    )

    print(
        f"Average risk score: "
        f"{risk_df['risk_score'].mean():.2f}"
    )

    print(
        f"Median risk score: "
        f"{risk_df['risk_score'].median():.2f}"
    )

    print(
        f"Operating threshold: "
        f"{OPERATING_THRESHOLD:.2f}"
    )

    print(
        f"Predicted at-risk sellers: "
        f"{(risk_df['predicted_at_risk'] == 1).sum():,}"
    )

    print(
        f"Predicted not-at-risk sellers: "
        f"{(risk_df['predicted_at_risk'] == 0).sum():,}"
    )

    print(
        "\nThe Random Forest model was used for scoring "
        "only. It was not retrained."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()