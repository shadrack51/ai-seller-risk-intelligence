from functools import lru_cache

import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.config import (
    DELIVERY_THRESHOLD,
    REVIEW_THRESHOLD,
    PROJECT_ROOT,
)

from src.api.models import (
    delivery_model,
    review_model,
)

from src.api.risk_explanation_service import (
    build_shap_explanations,
    build_logistic_explanations,
    build_delivery_action,
    build_review_action,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/assess",
    tags=["Seller Assessment"],
)


# ============================================================
# TEMPORAL FEATURE DATASET
# ============================================================

TEMPORAL_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_model_features.csv"
)


# ============================================================
# MODEL FEATURE SCHEMAS
# ============================================================

REVIEW_FEATURES = [
    "recent_orders",
    "older_orders",
    "orders_change",
    "recent_units_sold",
    "older_units_sold",
    "units_change",
    "recent_revenue",
    "older_revenue",
    "revenue_change",
    "recent_average_item_price",
    "older_average_item_price",
    "recent_average_freight",
    "older_average_freight",
    "seller_history_orders",
    "seller_history_reviews",
    "recent_order_activity",
    "older_order_activity",
    "low_history_orders",
    "low_history_reviews",
    "sufficient_order_history",
    "sufficient_review_history",
    "recent_average_review_score",
    "older_average_review_score",
    "review_score_change",
    "recent_low_review_rate",
    "older_low_review_rate",
    "low_review_rate_change",
    "recent_review_count",
    "older_review_count",
    "review_count_change",
    "recent_low_review_count",
    "older_low_review_count",
    "recent_review_activity",
    "older_review_activity",
    "review_score_declining",
    "low_review_rate_increasing",
]


DELIVERY_FEATURES = [
    "recent_orders",
    "older_orders",
    "orders_change",
    "recent_units_sold",
    "older_units_sold",
    "units_change",
    "recent_revenue",
    "older_revenue",
    "revenue_change",
    "recent_average_item_price",
    "older_average_item_price",
    "recent_average_freight",
    "older_average_freight",
    "seller_history_orders",
    "seller_history_reviews",
    "recent_order_activity",
    "older_order_activity",
    "low_history_orders",
    "low_history_reviews",
    "sufficient_order_history",
    "sufficient_review_history",
    "recent_delivered_orders",
    "older_delivered_orders",
    "recent_average_delivery_days",
    "older_average_delivery_days",
    "delivery_days_change",
    "recent_average_delivery_delay_days",
    "older_average_delivery_delay_days",
    "delivery_delay_change",
    "recent_late_delivery_rate",
    "older_late_delivery_rate",
    "late_delivery_rate_change",
    "recent_late_orders",
    "older_late_orders",
    "recent_delivery_activity",
    "older_delivery_activity",
    "delivery_deteriorating",
    "delivery_delay_increasing",
    "late_rate_increasing",
]


# ============================================================
# REQUEST SCHEMA
# ============================================================

class SellerAssessmentRequest(BaseModel):
    seller_id: str


# ============================================================
# EXPLANATION SCHEMA
# ============================================================

class RiskDriver(BaseModel):
    feature: str
    label: str
    value: float | None
    display_value: str
    contribution: float
    direction: str


# ============================================================
# RISK RESPONSE
# ============================================================

class RiskAssessment(BaseModel):
    risk_score: float
    threshold: float
    flagged: bool
    model: str
    target: str
    drivers: list[RiskDriver]
    recommended_action: str


# ============================================================
# SELLER RESPONSE
# ============================================================

class SellerAssessmentResponse(BaseModel):
    seller_id: str
    snapshot_date: str
    delivery: RiskAssessment
    review: RiskAssessment


# ============================================================
# LOAD TEMPORAL DATASET
# ============================================================

@lru_cache(maxsize=1)
def load_temporal_features() -> pd.DataFrame:

    if not TEMPORAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Temporal model feature dataset not found:\n"
            f"{TEMPORAL_FEATURES_PATH}"
        )

    df = pd.read_csv(
        TEMPORAL_FEATURES_PATH
    )

    required_columns = {
        "seller_id",
        "snapshot_date",
        *REVIEW_FEATURES,
        *DELIVERY_FEATURES,
    }

    missing_columns = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Temporal feature dataset is missing "
            "required columns:\n"
            + "\n".join(missing_columns)
        )

    df["seller_id"] = (
        df["seller_id"]
        .astype(str)
        .str.strip()
    )

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce",
    )

    if df["snapshot_date"].isna().any():
        raise ValueError(
            "Temporal feature dataset contains "
            "invalid snapshot dates."
        )

    return df


# ============================================================
# FIND LATEST SELLER SNAPSHOT
# ============================================================

def get_latest_seller_snapshot(
    seller_id: str,
) -> pd.Series:

    df = load_temporal_features()

    seller_id = seller_id.strip()

    seller_rows = df[
        df["seller_id"] == seller_id
    ].copy()

    if seller_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Seller was not found in the "
                "temporal risk dataset."
            ),
        )

    seller_rows = seller_rows.sort_values(
        "snapshot_date",
        ascending=False,
    )

    return seller_rows.iloc[0]


# ============================================================
# MODEL PIPELINE HELPERS
# ============================================================

def get_pipeline_estimator(
    pipeline,
    preferred_names: tuple[str, ...],
):
    """
    Retrieve the final model estimator from a
    sklearn Pipeline.
    """

    if hasattr(
        pipeline,
        "named_steps",
    ):

        for name in preferred_names:

            if name in pipeline.named_steps:
                return pipeline.named_steps[name]

    return pipeline


def prepare_model_input(
    model,
    row: pd.Series,
    features: list[str],
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                feature: row[feature]
                for feature in features
            }
        ],
        columns=features,
    )


# ============================================================
# DELIVERY PREDICTION + EXPLANATION
# ============================================================

def predict_delivery(
    row: pd.Series,
) -> RiskAssessment:

    X = prepare_model_input(
        delivery_model,
        row,
        DELIVERY_FEATURES,
    )

    risk_score = float(
        delivery_model.predict_proba(X)[0, 1]
    )

    flagged = (
        risk_score
        >= DELIVERY_THRESHOLD
    )

    estimator = get_pipeline_estimator(
        delivery_model,
        (
            "model",
            "random_forest",
            "random_forest_classifier",
        ),
    )

    # --------------------------------------------------------
    # Prepare values for SHAP.
    #
    # The deployed delivery model uses an imputer before
    # the Random Forest, so SHAP must explain the estimator
    # using the same imputed representation.
    # --------------------------------------------------------

    X_explain = X.copy()

    if hasattr(
        delivery_model,
        "named_steps",
    ) and "imputer" in delivery_model.named_steps:

        imputer = (
            delivery_model
            .named_steps["imputer"]
        )

        X_explain = pd.DataFrame(
            imputer.transform(X),
            columns=DELIVERY_FEATURES,
        )

    try:

        import shap

        explainer = shap.TreeExplainer(
            estimator
        )

        shap_values = (
            explainer.shap_values(
                X_explain
            )
        )

        # Handle binary-class SHAP representations.
        if isinstance(
            shap_values,
            list,
        ):

            shap_values = shap_values[1]

        else:

            shap_values = (
                shap_values.values
                if hasattr(
                    shap_values,
                    "values",
                )
                else shap_values
            )

            if (
                getattr(
                    shap_values,
                    "ndim",
                    0,
                )
                == 3
            ):

                shap_values = (
                    shap_values[:, :, 1]
                )

        delivery_contributions = (
            shap_values[0]
        )

        drivers = build_shap_explanations(
            row,
            DELIVERY_FEATURES,
            delivery_contributions,
            top_n=5,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Delivery explanation could not "
                f"be generated: {exc}"
            ),
        )

    recommended_action = (
        build_delivery_action(
            drivers
        )
    )

    return RiskAssessment(
        risk_score=risk_score,
        threshold=DELIVERY_THRESHOLD,
        flagged=flagged,
        model=estimator.__class__.__name__,
        target="future_delivery_risk",
        drivers=drivers,
        recommended_action=recommended_action,
    )


# ============================================================
# REVIEW PREDICTION + EXPLANATION
# ============================================================

def predict_review(
    row: pd.Series,
) -> RiskAssessment:

    X = prepare_model_input(
        review_model,
        row,
        REVIEW_FEATURES,
    )

    risk_score = float(
        review_model.predict_proba(X)[0, 1]
    )

    flagged = (
        risk_score
        >= REVIEW_THRESHOLD
    )

    try:

        drivers = build_logistic_explanations(
            review_model,
            row,
            REVIEW_FEATURES,
            top_n=5,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Review explanation could not "
                f"be generated: {exc}"
            ),
        )

    estimator = get_pipeline_estimator(
        review_model,
        (
            "logistic_regression",
            "model",
        ),
    )

    recommended_action = (
        build_review_action(
            drivers
        )
    )

    return RiskAssessment(
        risk_score=risk_score,
        threshold=REVIEW_THRESHOLD,
        flagged=flagged,
        model=estimator.__class__.__name__,
        target="future_review_risk",
        drivers=drivers,
        recommended_action=recommended_action,
    )


# ============================================================
# SELLER ASSESSMENT ENDPOINT
# ============================================================

@router.post(
    "/seller",
    response_model=SellerAssessmentResponse,
)
def assess_seller(
    request: SellerAssessmentRequest,
):

    seller_id = request.seller_id.strip()

    if not seller_id:
        raise HTTPException(
            status_code=400,
            detail="Seller ID cannot be empty.",
        )

    row = get_latest_seller_snapshot(
        seller_id
    )

    delivery = predict_delivery(
        row
    )

    review = predict_review(
        row
    )

    return SellerAssessmentResponse(
        seller_id=seller_id,
        snapshot_date=(
            row["snapshot_date"]
            .date()
            .isoformat()
        ),
        delivery=delivery,
        review=review,
    )