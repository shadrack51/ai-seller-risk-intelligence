from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_cors_headers():

    response = client.get(
        "/health",
        headers={
            "Origin": "http://localhost:5500",
        },
    )


    print("CORS test: PASS")
    print(f"Status code: {response.status_code}")
    print(
        "Allowed origin:",
        response.headers.get("access-control-allow-origin"),
    )

if __name__ == "__main__":
    test_cors_headers()