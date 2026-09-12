from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def build_valid_review_payload():
    return {
        "recent_orders": 0,
        "older_orders": 0,
        "orders_change": 0,
        "recent_units_sold": 0,
        "older_units_sold": 0,
        "units_change": 0,
        "recent_revenue": 0,
        "older_revenue": 0,
        "revenue_change": 0,
        "recent_average_item_price": 0,
        "older_average_item_price": 0,
        "recent_average_freight": 0,
        "older_average_freight": 0,
        "seller_history_orders": 0,
        "seller_history_reviews": 0,
        "recent_order_activity": 0,
        "older_order_activity": 0,
        "low_history_orders": 0,
        "low_history_reviews": 0,
        "sufficient_order_history": 0,
        "sufficient_review_history": 0,
        "recent_average_review_score": 0,
        "older_average_review_score": 0,
        "review_score_change": 0,
        "recent_low_review_rate": 0,
        "older_low_review_rate": 0,
        "low_review_rate_change": 0,
        "recent_review_count": 0,
        "older_review_count": 0,
        "review_count_change": 0,
        "recent_low_review_count": 0,
        "older_low_review_count": 0,
        "recent_review_activity": 0,
        "older_review_activity": 0,
        "review_score_declining": 0,
        "low_review_rate_increasing": 0,
    }


def test_invalid_review_input():
    payload = build_valid_review_payload()

    payload["recent_orders"] = "hello"

    response = client.post(
        "/predict/review",
        json=payload,
    )

    assert response.status_code == 422

    error_response = response.json()

    assert "detail" in error_response

    validation_error = error_response["detail"][0]

    assert validation_error["loc"] == [
        "body",
        "recent_orders",
    ]

    assert validation_error["type"] == "float_parsing"

    print("Invalid input validation: PASS")
    print(f"Status code: {response.status_code}")
    print(f"Error type: {validation_error['type']}")
    print(f"Error location: {validation_error['loc']}")


def test_missing_review_field():
    payload = build_valid_review_payload()

    del payload["recent_orders"]

    response = client.post(
        "/predict/review",
        json=payload,
    )

    assert response.status_code == 422

    error_response = response.json()

    assert "detail" in error_response

    validation_error = error_response["detail"][0]

    assert validation_error["loc"] == [
        "body",
        "recent_orders",
    ]

    assert validation_error["type"] == "missing"

    print("Missing field validation: PASS")
    print(f"Status code: {response.status_code}")
    print(f"Error type: {validation_error['type']}")
    print(f"Error location: {validation_error['loc']}")


if __name__ == "__main__":
    test_invalid_review_input()
    test_missing_review_field()