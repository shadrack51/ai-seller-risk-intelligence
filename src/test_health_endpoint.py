from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data == {
        "status": "ok"
    }

    print("Health endpoint test: PASS")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response_data}")


if __name__ == "__main__":
    test_health_endpoint()