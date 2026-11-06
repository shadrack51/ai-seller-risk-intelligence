from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# FEATURE DISPLAY NAMES
# ============================================================

FEATURE_LABELS = {
    "recent_orders": "Recent orders",
    "older_orders": "Older orders",
    "orders_change": "Change in orders",

    "recent_units_sold": "Recent units sold",
    "older_units_sold": "Older units sold",
    "units_change": "Change in units sold",

    "recent_revenue": "Recent revenue",
    "older_revenue": "Older revenue",
    "revenue_change": "Change in revenue",

    "recent_average_item_price": "Recent average item price",
    "older_average_item_price": "Older average item price",

    "recent_average_freight": "Recent average freight",
    "older_average_freight": "Older average freight",

    "seller_history_orders": "Seller order history",
    "seller_history_reviews": "Seller review history",

    "recent_order_activity": "Recent order activity",
    "older_order_activity": "Older order activity",

    "low_history_orders": "Low order history",
    "low_history_reviews": "Low review history",

    "sufficient_order_history": "Sufficient order history",
    "sufficient_review_history": "Sufficient review history",

    "recent_average_review_score": "Recent average review score",
    "older_average_review_score": "Older average review score",
    "review_score_change": "Change in review score",

    "recent_low_review_rate": "Recent low-review rate",
    "older_low_review_rate": "Older low-review rate",
    "low_review_rate_change": "Change in low-review rate",

    "recent_review_count": "Recent review count",
    "older_review_count": "Older review count",
    "review_count_change": "Change in review count",

    "recent_low_review_count": "Recent low-review count",
    "older_low_review_count": "Older low-review count",

    "recent_review_activity": "Recent review activity",
    "older_review_activity": "Older review activity",

    "review_score_declining": "Review score declining",
    "low_review_rate_increasing": "Low-review rate increasing",

    "recent_delivered_orders": "Recent delivered orders",
    "older_delivered_orders": "Older delivered orders",

    "recent_average_delivery_days": "Recent average delivery time",
    "older_average_delivery_days": "Older average delivery time",

    "delivery_days_change": "Change in delivery time",

    "recent_average_delivery_delay_days": (
        "Recent average delivery delay"
    ),

    "older_average_delivery_delay_days": (
        "Older average delivery delay"
    ),

    "delivery_delay_change": "Change in delivery delay",

    "recent_late_delivery_rate": "Recent late-delivery rate",
    "older_late_delivery_rate": "Older late-delivery rate",

    "late_delivery_rate_change": (
        "Change in late-delivery rate"
    ),

    "recent_late_orders": "Recent late orders",
    "older_late_orders": "Older late orders",

    "recent_delivery_activity": "Recent delivery activity",
    "older_delivery_activity": "Older delivery activity",

    "delivery_deteriorating": "Delivery performance deteriorating",
    "delivery_delay_increasing": "Delivery delay increasing",
    "late_rate_increasing": "Late-delivery rate increasing",
}


# ============================================================
# FEATURE FORMATTING
# ============================================================

def feature_label(feature: str) -> str:
    """
    Convert an internal feature name into a
    human-readable dashboard label.
    """

    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title(),
    )


def format_feature_value(
    feature: str,
    value: float,
) -> str:
    """
    Format a seller feature for display.
    """

    if pd.isna(value):
        return "Unavailable"

    percentage_features = {
        "recent_low_review_rate",
        "older_low_review_rate",
        "low_review_rate_change",
        "recent_late_delivery_rate",
        "older_late_delivery_rate",
        "late_delivery_rate_change",
    }

    binary_features = {
        "recent_order_activity",
        "older_order_activity",
        "low_history_orders",
        "low_history_reviews",
        "sufficient_order_history",
        "sufficient_review_history",
        "recent_review_activity",
        "older_review_activity",
        "review_score_declining",
        "low_review_rate_increasing",
        "recent_delivery_activity",
        "older_delivery_activity",
        "delivery_deteriorating",
        "delivery_delay_increasing",
        "late_rate_increasing",
    }

    if feature in percentage_features:
        return f"{value * 100:.1f}%"

    if feature in binary_features:
        return "Yes" if value >= 0.5 else "No"

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    if abs(value) >= 10:
        return f"{value:,.1f}"

    return f"{value:.2f}"


# ============================================================
# SHAP CONTRIBUTION INTERPRETATION
# ============================================================

def contribution_direction(
    contribution: float,
) -> str:
    """
    Positive contribution means the feature pushed the
    model toward the positive-risk class.

    Negative contribution means it pushed the prediction
    away from the positive-risk class.
    """

    if contribution > 0:
        return "increases_risk"

    if contribution < 0:
        return "reduces_risk"

    return "neutral"


def build_shap_explanations(
    row: pd.Series,
    features: list[str],
    shap_values: np.ndarray,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Build human-readable local SHAP explanations
    for one seller snapshot.
    """

    contributions = np.asarray(
        shap_values,
        dtype=float,
    )

    if contributions.ndim != 1:
        raise ValueError(
            "Expected one-dimensional SHAP values "
            "for a single seller observation."
        )

    if len(contributions) != len(features):
        raise ValueError(
            "SHAP contribution count does not match "
            "the model feature count."
        )

    explanation_rows = []

    for feature, contribution in zip(
        features,
        contributions,
    ):
        value = row.get(feature, np.nan)

        explanation_rows.append(
            {
                "feature": feature,
                "label": feature_label(feature),
                "value": (
                    None
                    if pd.isna(value)
                    else float(value)
                ),
                "display_value": format_feature_value(
                    feature,
                    value,
                ),
                "contribution": float(
                    contribution
                ),
                "direction": contribution_direction(
                    contribution
                ),
            }
        )

    explanation_rows.sort(
        key=lambda item: abs(
            item["contribution"]
        ),
        reverse=True,
    )

    return explanation_rows[:top_n]


# ============================================================
# LOGISTIC REGRESSION CONTRIBUTIONS
# ============================================================

def build_logistic_explanations(
    pipeline,
    row: pd.Series,
    features: list[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Calculate local feature contributions for the
    deployed Logistic Regression review model.

    Contribution is calculated in the model's
    standardized feature space:

        coefficient × standardized feature value
    """

    if not hasattr(
        pipeline,
        "named_steps",
    ):
        raise ValueError(
            "Review model must be a sklearn Pipeline."
        )

    if "imputer" not in pipeline.named_steps:
        raise ValueError(
            "Review pipeline is missing its imputer."
        )

    if "scaler" not in pipeline.named_steps:
        raise ValueError(
            "Review pipeline is missing its scaler."
        )

    model = None

    for name in (
        "logistic_regression",
        "model",
    ):
        if name in pipeline.named_steps:
            model = pipeline.named_steps[name]
            break

    if model is None:
        raise ValueError(
            "Review pipeline is missing its "
            "Logistic Regression model."
        )

    X = pd.DataFrame(
        [[row[feature] for feature in features]],
        columns=features,
    )

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    coefficients = np.asarray(
        model.coef_[0],
        dtype=float,
    )

    contributions = (
        coefficients
        * X_scaled[0]
    )

    explanation_rows = []

    for index, feature in enumerate(features):
        value = row.get(feature, np.nan)

        explanation_rows.append(
            {
                "feature": feature,
                "label": feature_label(feature),
                "value": (
                    None
                    if pd.isna(value)
                    else float(value)
                ),
                "display_value": format_feature_value(
                    feature,
                    value,
                ),
                "contribution": float(
                    contributions[index]
                ),
                "direction": contribution_direction(
                    contributions[index]
                ),
            }
        )

    explanation_rows.sort(
        key=lambda item: abs(
            item["contribution"]
        ),
        reverse=True,
    )

    return explanation_rows[:top_n]


# ============================================================
# BUSINESS ACTIONS
# ============================================================

def build_delivery_action(
    explanations: list[dict[str, Any]],
) -> str:
    """
    Convert the strongest delivery-risk signals into
    a conservative business action.
    """

    risk_drivers = {
        item["feature"]
        for item in explanations
        if item["direction"] == "increases_risk"
    }

    if (
        "late_rate_increasing" in risk_drivers
        or "recent_late_delivery_rate" in risk_drivers
        or "late_delivery_rate_change" in risk_drivers
    ):
        return (
            "Investigate recent late-delivery performance "
            "and place the seller under closer delivery monitoring."
        )

    if (
        "delivery_delay_increasing" in risk_drivers
        or "delivery_delay_change" in risk_drivers
    ):
        return (
            "Review recent delivery delays and monitor "
            "fulfillment performance more closely."
        )

    return (
        "Review recent seller fulfillment activity "
        "and continue monitoring delivery performance."
    )


def build_review_action(
    explanations: list[dict[str, Any]],
) -> str:
    """
    Convert the strongest review-risk signals into
    a conservative business action.
    """

    risk_drivers = {
        item["feature"]
        for item in explanations
        if item["direction"] == "increases_risk"
    }

    if (
        "review_score_declining" in risk_drivers
        or "review_score_change" in risk_drivers
    ):
        return (
            "Screen recent customer feedback and investigate "
            "the change in review performance."
        )

    if (
        "low_review_rate_increasing" in risk_drivers
        or "low_review_rate_change" in risk_drivers
    ):
        return (
            "Review recent low-rating activity and investigate "
            "emerging customer satisfaction concerns."
        )

    return (
        "Review recent customer feedback and continue "
        "monitoring seller review performance."
    )