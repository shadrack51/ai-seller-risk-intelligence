from fastapi.testclient import TestClient
import pandas as pd

from src.api.main import app
from src.api.models import delivery_model, review_model
from src.api.routes import DELIVERY_FEATURES, REVIEW_FEATURES
from src.api.schemas import SellerPredictionRequest
from src.api.config import DELIVERY_THRESHOLD, REVIEW_THRESHOLD


client = TestClient(app)


def build_test_payload():
    payload = {}

    for feature in REVIEW_FEATURES:
        payload[feature] = 0.0

    for feature in DELIVERY_FEATURES:
        payload[feature] = 0.0

    return payload


def test_feature_contracts():
    assert len(REVIEW_FEATURES) == 36
    assert len(DELIVERY_FEATURES) == 39

    assert len(REVIEW_FEATURES) == len(set(REVIEW_FEATURES))
    assert len(DELIVERY_FEATURES) == len(set(DELIVERY_FEATURES))

    assert SellerPredictionRequest is not None


def test_model_inputs():
    payload = build_test_payload()

    review_input = pd.DataFrame(
        [[payload[feature] for feature in REVIEW_FEATURES]],
        columns=REVIEW_FEATURES,
    )

    delivery_input = pd.DataFrame(
        [[payload[feature] for feature in DELIVERY_FEATURES]],
        columns=DELIVERY_FEATURES,
    )

    assert list(review_input.columns) == REVIEW_FEATURES
    assert list(delivery_input.columns) == DELIVERY_FEATURES

    assert review_input.shape == (1, 36)
    assert delivery_input.shape == (1, 39)


def test_models_load():
    assert review_model is not None
    assert delivery_model is not None

    assert hasattr(review_model, "predict_proba")
    assert hasattr(delivery_model, "predict_proba")


def test_direct_predictions():
    payload = build_test_payload()

    review_input = pd.DataFrame(
        [[payload[feature] for feature in REVIEW_FEATURES]],
        columns=REVIEW_FEATURES,
    )

    delivery_input = pd.DataFrame(
        [[payload[feature] for feature in DELIVERY_FEATURES]],
        columns=DELIVERY_FEATURES,
    )

    review_score = float(
        review_model.predict_proba(review_input)[0, 1]
    )

    delivery_score = float(
        delivery_model.predict_proba(delivery_input)[0, 1]
    )

    assert 0.0 <= review_score <= 1.0
    assert 0.0 <= delivery_score <= 1.0


def test_seller_endpoint():
    payload = build_test_payload()

    response = client.post(
        "/predict/seller",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert "delivery" in result
    assert "review" in result

    assert "risk_score" in result["delivery"]
    assert "monitoring_threshold" in result["delivery"]
    assert "monitor_delivery" in result["delivery"]
    assert "model" in result["delivery"]
    assert "target" in result["delivery"]

    assert "risk_score" in result["review"]
    assert "screening_threshold" in result["review"]
    assert "screen_for_review" in result["review"]
    assert "model" in result["review"]
    assert "target" in result["review"]

    assert result["delivery"]["monitoring_threshold"] == DELIVERY_THRESHOLD
    assert result["review"]["screening_threshold"] == REVIEW_THRESHOLD


def test_prediction_consistency():
    payload = build_test_payload()

    response = client.post(
        "/predict/seller",
        json=payload,
    )

    result = response.json()

    review_input = pd.DataFrame(
        [[payload[feature] for feature in REVIEW_FEATURES]],
        columns=REVIEW_FEATURES,
    )

    delivery_input = pd.DataFrame(
        [[payload[feature] for feature in DELIVERY_FEATURES]],
        columns=DELIVERY_FEATURES,
    )

    direct_review_score = float(
        review_model.predict_proba(review_input)[0, 1]
    )

    direct_delivery_score = float(
        delivery_model.predict_proba(delivery_input)[0, 1]
    )

    api_review_score = result["review"]["risk_score"]
    api_delivery_score = result["delivery"]["risk_score"]

    assert abs(api_review_score - direct_review_score) < 1e-6
    assert abs(api_delivery_score - direct_delivery_score) < 1e-6

    print("Prediction consistency: PASS")
    print(f"Review score: {api_review_score:.4f}")
    print(f"Delivery score: {api_delivery_score:.4f}")


if __name__ == "__main__":
    test_feature_contracts()
    test_model_inputs()
    test_models_load()
    test_direct_predictions()
    test_seller_endpoint()
    test_prediction_consistency()

    print("Seller endpoint integration test: PASS")