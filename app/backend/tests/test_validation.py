VALID_FEATURES = {
    "longitude": -118.24,
    "latitude": 34.05,
    "housing_median_age": 30,
    "total_rooms": 2400,
    "total_bedrooms": 500,
    "population": 1200,
    "households": 400,
    "median_income": 4.5,
    "ocean_proximity": "NEAR BAY",
}


def test_ai_service_unavailable_returns_502(client, backend_module, monkeypatch):
    def fake_post(*args, **kwargs):
        raise backend_module.requests.ConnectionError("connection refused")

    monkeypatch.setattr(backend_module.requests, "post", fake_post)

    response = client.post("/api/predict", json={"features": VALID_FEATURES})

    assert response.status_code == 502
    assert response.json()["detail"]["error"] == "ai_service_unavailable"


def test_ai_service_timeout_returns_504(client, backend_module, monkeypatch):
    def fake_post(*args, **kwargs):
        raise backend_module.requests.Timeout("request timed out")

    monkeypatch.setattr(backend_module.requests, "post", fake_post)

    response = client.post("/api/predict", json={"features": VALID_FEATURES})

    assert response.status_code == 504
    assert response.json()["detail"]["error"] == "ai_service_timeout"
