from fastapi import APIRouter
import pandas as pd

from src.api.config import DELIVERY_THRESHOLD, REVIEW_THRESHOLD
from src.api.models import delivery_model, review_model
from src.api.schemas import (
    DeliveryRequest,
    DeliveryResponse,
    ReviewRequest,
    ReviewResponse,
    SellerPredictionRequest,
    SellerPredictionResponse,
)


router = APIRouter()


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


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post(
    "/predict/review",
    response_model=ReviewResponse,
)
def predict_review(request: ReviewRequest):
    input_data = pd.DataFrame(
        [[getattr(request, feature) for feature in REVIEW_FEATURES]],
        columns=REVIEW_FEATURES,
    )

    risk_score = float(
        review_model.predict_proba(input_data)[0, 1]
    )

    return {
        "risk_score": risk_score,
        "screening_threshold": REVIEW_THRESHOLD,
        "screen_for_review": risk_score >= REVIEW_THRESHOLD,
        "model": "Review Logistic Regression",
        "target": "future_review_risk",
    }


@router.post(
    "/predict/delivery",
    response_model=DeliveryResponse,
)
def predict_delivery(request: DeliveryRequest):
    input_data = pd.DataFrame(
        [[getattr(request, feature) for feature in DELIVERY_FEATURES]],
        columns=DELIVERY_FEATURES,
    )

    risk_score = float(
        delivery_model.predict_proba(input_data)[0, 1]
    )

    return {
        "risk_score": risk_score,
        "monitoring_threshold": DELIVERY_THRESHOLD,
        "monitor_delivery": risk_score >= DELIVERY_THRESHOLD,
        "model": "Delivery Random Forest",
        "target": "future_delivery_risk",
    }


@router.post(
    "/predict/seller",
    response_model=SellerPredictionResponse,
)
def predict_seller(request: SellerPredictionRequest):
    review_input = pd.DataFrame(
        [[getattr(request, feature) for feature in REVIEW_FEATURES]],
        columns=REVIEW_FEATURES,
    )

    delivery_input = pd.DataFrame(
        [[getattr(request, feature) for feature in DELIVERY_FEATURES]],
        columns=DELIVERY_FEATURES,
    )

    review_risk_score = float(
        review_model.predict_proba(review_input)[0, 1]
    )

    delivery_risk_score = float(
        delivery_model.predict_proba(delivery_input)[0, 1]
    )

    return {
        "delivery": {
            "risk_score": delivery_risk_score,
            "monitoring_threshold": DELIVERY_THRESHOLD,
            "monitor_delivery": delivery_risk_score >= DELIVERY_THRESHOLD,
            "model": "Delivery Random Forest",
            "target": "future_delivery_risk",
        },
        "review": {
            "risk_score": review_risk_score,
            "screening_threshold": REVIEW_THRESHOLD,
            "screen_for_review": review_risk_score >= REVIEW_THRESHOLD,
            "model": "Review Logistic Regression",
            "target": "future_review_risk",
        },
    }