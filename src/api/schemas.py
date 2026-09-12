from pydantic import BaseModel


class ReviewRequest(BaseModel):
    recent_orders: float
    older_orders: float
    orders_change: float

    recent_units_sold: float
    older_units_sold: float
    units_change: float

    recent_revenue: float
    older_revenue: float
    revenue_change: float

    recent_average_item_price: float
    older_average_item_price: float

    recent_average_freight: float
    older_average_freight: float

    seller_history_orders: float
    seller_history_reviews: float

    recent_order_activity: float
    older_order_activity: float

    low_history_orders: float
    low_history_reviews: float

    sufficient_order_history: float
    sufficient_review_history: float

    recent_average_review_score: float
    older_average_review_score: float
    review_score_change: float

    recent_low_review_rate: float
    older_low_review_rate: float
    low_review_rate_change: float

    recent_review_count: float
    older_review_count: float
    review_count_change: float

    recent_low_review_count: float
    older_low_review_count: float

    recent_review_activity: float
    older_review_activity: float

    review_score_declining: float
    low_review_rate_increasing: float


class DeliveryRequest(BaseModel):
    recent_orders: float
    older_orders: float
    orders_change: float

    recent_units_sold: float
    older_units_sold: float
    units_change: float

    recent_revenue: float
    older_revenue: float
    revenue_change: float

    recent_average_item_price: float
    older_average_item_price: float

    recent_average_freight: float
    older_average_freight: float

    seller_history_orders: float
    seller_history_reviews: float

    recent_order_activity: float
    older_order_activity: float

    low_history_orders: float
    low_history_reviews: float

    sufficient_order_history: float
    sufficient_review_history: float

    recent_delivered_orders: float
    older_delivered_orders: float

    recent_average_delivery_days: float
    older_average_delivery_days: float
    delivery_days_change: float

    recent_average_delivery_delay_days: float
    older_average_delivery_delay_days: float
    delivery_delay_change: float

    recent_late_delivery_rate: float
    older_late_delivery_rate: float
    late_delivery_rate_change: float

    recent_late_orders: float
    older_late_orders: float

    recent_delivery_activity: float
    older_delivery_activity: float

    delivery_deteriorating: float
    delivery_delay_increasing: float
    late_rate_increasing: float


class SellerPredictionRequest(BaseModel):
    recent_orders: float
    older_orders: float
    orders_change: float

    recent_units_sold: float
    older_units_sold: float
    units_change: float

    recent_revenue: float
    older_revenue: float
    revenue_change: float

    recent_average_item_price: float
    older_average_item_price: float

    recent_average_freight: float
    older_average_freight: float

    seller_history_orders: float
    seller_history_reviews: float

    recent_order_activity: float
    older_order_activity: float

    low_history_orders: float
    low_history_reviews: float

    sufficient_order_history: float
    sufficient_review_history: float

    # Delivery features
    recent_delivered_orders: float
    older_delivered_orders: float

    recent_average_delivery_days: float
    older_average_delivery_days: float
    delivery_days_change: float

    recent_average_delivery_delay_days: float
    older_average_delivery_delay_days: float
    delivery_delay_change: float

    recent_late_delivery_rate: float
    older_late_delivery_rate: float
    late_delivery_rate_change: float

    recent_late_orders: float
    older_late_orders: float

    recent_delivery_activity: float
    older_delivery_activity: float

    delivery_deteriorating: float
    delivery_delay_increasing: float
    late_rate_increasing: float

    # Review features
    recent_average_review_score: float
    older_average_review_score: float
    review_score_change: float

    recent_low_review_rate: float
    older_low_review_rate: float
    low_review_rate_change: float

    recent_review_count: float
    older_review_count: float
    review_count_change: float

    recent_low_review_count: float
    older_low_review_count: float

    recent_review_activity: float
    older_review_activity: float

    review_score_declining: float
    low_review_rate_increasing: float


class ReviewResponse(BaseModel):
    risk_score: float
    screening_threshold: float
    screen_for_review: bool
    model: str
    target: str


class DeliveryResponse(BaseModel):
    risk_score: float
    monitoring_threshold: float
    monitor_delivery: bool
    model: str
    target: str


class SellerPredictionResponse(BaseModel):
    delivery: DeliveryResponse
    review: ReviewResponse