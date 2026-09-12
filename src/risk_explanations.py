"""
Seller Risk Explainability

Purpose:
    Explain why individual sellers receive high or low risk
    predictions from the trained Random Forest model.

This script uses SHAP (SHapley Additive exPlanations) to
calculate local feature contributions.

Inputs:
    - random_forest_model.joblib
    - seller_prediction_dataset.csv
    - seller_risk_scores.csv

Outputs:
    - risk_explanations.csv

Important:
    - The model is NOT retrained.
    - The test set is NOT used for training.
    - SHAP explains the existing trained model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


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

RISK_SCORES_PATH = (
    PROCESSED_DATA_DIR
    / "seller_risk_scores.csv"
)

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "risk_explanations.csv"
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


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "at_risk"

TOP_N_SELLERS = 20

TOP_N_FEATURES = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_feature_name(feature_name):
    """
    Convert a technical column name into a readable label.
    """

    replacements = {
        "total_products": "Total products",
        "incomplete_products": "Incomplete products",
        "total_orders": "Total orders",
        "units_sold": "Units sold",
        "total_revenue": "Total revenue",
        "average_item_price": "Average item price",
        "average_freight": "Average freight",
        "catalog_risk": "Catalog risk",
        "commercial_exposure": "Commercial exposure",
        "order_exposure": "Order exposure",
        "seller_risk_score": "Seller risk score",
        "average_delivery_days": "Average delivery days",
        "average_estimated_delivery_days": (
            "Average estimated delivery days"
        ),
        "average_delivery_delay_days": (
            "Average delivery delay"
        ),
        "delivered_orders": "Delivered orders",
        "review_count": "Review count",
        "low_review_count": "Low-review count",
        "low_review_rate": "Low-review rate",
    }

    return replacements.get(
        feature_name,
        feature_name.replace("_", " ").title()
    )


def create_explanation_text(
    feature_name,
    feature_value,
    shap_value,
):
    """
    Create a simple human-readable explanation for a
    feature's contribution.

    This is an interpretive layer on top of SHAP values.
    It does not claim causality.
    """

    readable_name = format_feature_name(
        feature_name
    )

    direction = (
        "increased"
        if shap_value > 0
        else "reduced"
    )

    if isinstance(feature_value, (int, float, np.integer, np.floating)):

        if "rate" in feature_name:
            value_text = f"{feature_value:.2f}"

        elif "revenue" in feature_name:
            value_text = f"{feature_value:,.2f}"

        elif "days" in feature_name:
            value_text = f"{feature_value:.2f}"

        else:
            value_text = f"{feature_value:.2f}"

    else:

        value_text = str(feature_value)

    return (
        f"{readable_name} "
        f"({value_text}) "
        f"{direction} the model's predicted risk."
    )


def get_shap_values_for_positive_class(
    shap_explanation,
):
    """
    Handle common SHAP output structures across versions.

    We want SHAP contributions for class 1:
    'at_risk'.
    """

    values = shap_explanation.values

    # Recent SHAP versions can return:
    #
    # (rows, features, classes)
    #
    if values.ndim == 3:

        return values[:, :, 1]

    # Older/common TreeExplainer behavior:
    #
    # (rows, features)
    #
    if values.ndim == 2:

        return values

    raise ValueError(
        f"Unexpected SHAP value shape: {values.shape}"
    )


def get_base_value_for_positive_class(
    shap_explanation,
):
    """
    Extract the SHAP base value for the positive class
    where available.
    """

    base_values = shap_explanation.base_values

    if np.ndim(base_values) == 2:

        return np.asarray(
            base_values[:, 1]
        )

    if np.ndim(base_values) == 1:

        return np.asarray(
            base_values
        )

    return np.asarray(
        base_values
    ).reshape(-1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SELLER RISK EXPLAINABILITY")
    print("=" * 70)

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    print("\nLoading trained Random Forest model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    pipeline = joblib.load(
        MODEL_PATH
    )

    print(
        "Random Forest model loaded successfully."
    )

    print(
        f"Model path: {MODEL_PATH}"
    )

    # ========================================================
    # 2. VALIDATE PIPELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL PIPELINE VALIDATION")
    print("=" * 70)

    if not hasattr(
        pipeline,
        "named_steps"
    ):

        raise ValueError(
            "Expected a scikit-learn Pipeline."
        )

    print("Pipeline steps:")

    for step_name, step_object in (
        pipeline.named_steps.items()
    ):

        print(
            f" - {step_name}: "
            f"{type(step_object).__name__}"
        )

    if "imputer" not in pipeline.named_steps:

        raise ValueError(
            "The 'imputer' step was not found."
        )

    if "model" not in pipeline.named_steps:

        raise ValueError(
            "The 'model' step was not found."
        )

    imputer = pipeline.named_steps[
        "imputer"
    ]

    forest = pipeline.named_steps[
        "model"
    ]

    print(
        "\nRequired pipeline components -> PASS"
    )

    # ========================================================
    # 3. LOAD DATASETS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING SELLER DATA")
    print("=" * 70)

    prediction_df = pd.read_csv(
        PREDICTION_DATASET_PATH
    )

    risk_df = pd.read_csv(
        RISK_SCORES_PATH
    )

    print(
        f"Prediction dataset rows: "
        f"{len(prediction_df):,}"
    )

    print(
        f"Risk score dataset rows: "
        f"{len(risk_df):,}"
    )

    # ========================================================
    # 4. DATA VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    required_prediction_columns = (
        MODEL_FEATURES
        + [
            TARGET_COLUMN,
            "seller_id",
        ]
    )

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in prediction_df.columns
    ]

    if missing_prediction_columns:

        raise ValueError(
            "Missing prediction columns:\n"
            + "\n".join(
                missing_prediction_columns
            )
        )

    print(
        "Prediction dataset columns -> PASS"
    )

    required_risk_columns = [
        "seller_id",
        "risk_probability",
        "risk_score",
        "risk_level",
        "predicted_at_risk",
    ]

    missing_risk_columns = [
        column
        for column in required_risk_columns
        if column not in risk_df.columns
    ]

    if missing_risk_columns:

        raise ValueError(
            "Missing risk score columns:\n"
            + "\n".join(
                missing_risk_columns
            )
        )

    print(
        "Risk score dataset columns -> PASS"
    )

    # ========================================================
    # 5. ALIGN SELLER DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("ALIGNING SELLER RECORDS")
    print("=" * 70)

    if prediction_df["seller_id"].duplicated().any():

        raise ValueError(
            "Duplicate seller IDs found in prediction dataset."
        )

    if risk_df["seller_id"].duplicated().any():

        raise ValueError(
            "Duplicate seller IDs found in risk dataset."
        )

    # Align risk scores with prediction dataset using seller_id.

    risk_lookup = risk_df[
        [
            "seller_id",
            "risk_probability",
            "risk_score",
            "risk_level",
            "predicted_at_risk",
        ]
    ].copy()

    explanation_df = prediction_df[
        [
            "seller_id",
        ]
    ].merge(
        risk_lookup,
        on="seller_id",
        how="left",
        validate="one_to_one",
    )

    if explanation_df[
        "risk_probability"
    ].isna().any():

        raise ValueError(
            "Some sellers are missing risk scores."
        )

    print(
        "Seller alignment -> PASS"
    )

    # ========================================================
    # 6. SELECT HIGH-RISK SELLERS
    # ========================================================

    print("\n" + "=" * 70)
    print("SELECTING HIGH-RISK SELLERS")
    print("=" * 70)

    selected_sellers = (
        risk_df
        .sort_values(
            "risk_probability",
            ascending=False
        )
        .head(
            TOP_N_SELLERS
        )
        [
            "seller_id"
        ]
        .tolist()
    )

    print(
        f"Sellers selected for explanation: "
        f"{len(selected_sellers)}"
    )

    # ========================================================
    # 7. PREPARE MODEL FEATURES
    # ========================================================

    X = prediction_df[
        MODEL_FEATURES
    ].copy()

    # Make sure column order matches model.

    model_features = list(
        imputer.feature_names_in_
    )

    if set(model_features) != set(
        X.columns
    ):

        raise ValueError(
            "Model features and dataset features do not match."
        )

    X = X[
        model_features
    ]

    print(
        f"Model feature count: "
        f"{len(model_features)}"
    )

    # ========================================================
    # 8. APPLY TRAINED IMPUTER
    # ========================================================

    print("\n" + "=" * 70)
    print("APPLYING TRAINED IMPUTER")
    print("=" * 70)

    X_imputed = imputer.transform(
        X
    )

    X_imputed_df = pd.DataFrame(
        X_imputed,
        columns=model_features,
        index=X.index,
    )

    print(
        f"Imputed feature matrix: "
        f"{X_imputed_df.shape}"
    )

    print(
        "The imputer is the one already fitted "
        "during model training."
    )

    # ========================================================
    # 9. CREATE TREE EXPLAINER
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING SHAP TREE EXPLAINER")
    print("=" * 70)

    print(
        "Creating SHAP TreeExplainer "
        "for the Random Forest..."
    )

    explainer = shap.TreeExplainer(
        forest
    )

    print(
        "SHAP TreeExplainer created successfully."
    )

    # ========================================================
    # 10. CALCULATE SHAP VALUES
    # ========================================================

    print("\n" + "=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)

    selected_indices = (
        prediction_df[
            prediction_df["seller_id"].isin(
                selected_sellers
            )
        ]
        .index
        .tolist()
    )

    X_selected = X_imputed_df.loc[
        selected_indices
    ]

    print(
        f"Rows being explained: "
        f"{len(X_selected)}"
    )

    print(
        "Calculating feature contributions..."
    )

    shap_explanation = explainer(
        X_selected
    )

    shap_values = (
        get_shap_values_for_positive_class(
            shap_explanation
        )
    )

    base_values = (
        get_base_value_for_positive_class(
            shap_explanation
        )
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # ========================================================
    # 11. CREATE LOCAL EXPLANATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING LOCAL SELLER EXPLANATIONS")
    print("=" * 70)

    explanation_rows = []

    for row_number, seller_index in enumerate(
        selected_indices
    ):

        seller_id = prediction_df.loc[
            seller_index,
            "seller_id"
        ]

        risk_probability = risk_df[
            risk_df["seller_id"] == seller_id
        ]["risk_probability"].iloc[0]

        risk_score = risk_df[
            risk_df["seller_id"] == seller_id
        ]["risk_score"].iloc[0]

        risk_level = risk_df[
            risk_df["seller_id"] == seller_id
        ]["risk_level"].iloc[0]

        seller_shap_values = (
            shap_values[row_number]
        )

        seller_features = (
            X_imputed_df.loc[
                seller_index
            ]
        )

        # Create feature contribution table.

        contribution_df = pd.DataFrame(
            {
                "feature": model_features,
                "feature_value": (
                    seller_features.values
                ),
                "shap_value": (
                    seller_shap_values
                ),
            }
        )

        contribution_df[
            "absolute_shap_value"
        ] = (
            contribution_df[
                "shap_value"
            ].abs()
        )

        contribution_df = (
            contribution_df
            .sort_values(
                "absolute_shap_value",
                ascending=False
            )
            .head(
                TOP_N_FEATURES
            )
        )

        for position, (_, row) in enumerate(
            contribution_df.iterrows(),
            start=1,
        ):

            explanation = (
                create_explanation_text(
                    row["feature"],
                    row["feature_value"],
                    row["shap_value"],
                )
            )

            explanation_rows.append(
                {
                    "seller_id": seller_id,
                    "risk_probability": (
                        risk_probability
                    ),
                    "risk_score": (
                        risk_score
                    ),
                    "risk_level": (
                        risk_level
                    ),
                    "explanation_rank": (
                        position
                    ),
                    "feature": (
                        row["feature"]
                    ),
                    "feature_value": (
                        row["feature_value"]
                    ),
                    "shap_value": (
                        row["shap_value"]
                    ),
                    "absolute_shap_value": (
                        row[
                            "absolute_shap_value"
                        ]
                    ),
                    "direction": (
                        "INCREASED RISK"
                        if row["shap_value"] > 0
                        else "REDUCED RISK"
                    ),
                    "explanation": explanation,
                }
            )

    explanations = pd.DataFrame(
        explanation_rows
    )

    print(
        f"Explanation rows created: "
        f"{len(explanations):,}"
    )

    # ========================================================
    # 12. DISPLAY TOP EXPLANATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAMPLE SELLER EXPLANATIONS")
    print("=" * 70)

    for seller_id in selected_sellers[:5]:

        seller_explanations = explanations[
            explanations["seller_id"] == seller_id
        ]

        first_row = seller_explanations.iloc[0]

        print()
        print(
            f"Seller: {seller_id}"
        )

        print(
            f"Risk probability: "
            f"{first_row['risk_probability']:.4f}"
        )

        print(
            f"Risk score: "
            f"{first_row['risk_score']:.2f}"
        )

        print(
            f"Risk level: "
            f"{first_row['risk_level']}"
        )

        print(
            "Top risk drivers:"
        )

        for _, row in (
            seller_explanations
            .head(TOP_N_FEATURES)
            .iterrows()
        ):

            print(
                f"  {row['direction']}: "
                f"{format_feature_name(row['feature'])} "
                f"(SHAP={row['shap_value']:.4f})"
            )

    # ========================================================
    # 13. GLOBAL EXPLANATION SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("GLOBAL SHAP IMPORTANCE")
    print("=" * 70)

    global_importance = pd.DataFrame(
        {
            "feature": model_features,
            "mean_absolute_shap": (
                np.abs(shap_values).mean(axis=0)
            ),
        }
    )

    global_importance = (
        global_importance
        .sort_values(
            "mean_absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        global_importance
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # 14. SAVE EXPLANATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVING RISK EXPLANATIONS")
    print("=" * 70)

    explanations.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nRisk explanations saved to:\n"
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # 15. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("RISK EXPLAINABILITY COMPLETE")
    print("=" * 70)

    print(
        f"Total sellers in dataset: "
        f"{len(prediction_df):,}"
    )

    print(
        f"Sellers explained in detail: "
        f"{len(selected_sellers):,}"
    )

    print(
        f"Top features per seller: "
        f"{TOP_N_FEATURES}"
    )

    print(
        "\nThe Random Forest was not retrained."
    )

    print(
        "SHAP was used to explain the existing model."
    )

    print(
        "\nImportant:"
    )

    print(
        "SHAP contributions explain the model's prediction; "
        "they do not establish causal relationships."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()