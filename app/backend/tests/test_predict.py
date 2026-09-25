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


class DummyAIResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "prediction": 432500.0,
            "model_version": "1.0.0",
            "request_id": "backend-test-001",
            "target_column": "median_house_value",
        }


def test_predict_valid_forwards_request_id(client, backend_module, monkeypatch, mock_mongo):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyAIResponse()

    monkeypatch.setattr(backend_module.requests, "post", fake_post)

    response = client.post(
        "/api/predict",
        json={"features": VALID_FEATURES},
        headers={"X-Request-ID": "backend-test-001"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 432500.0
    assert data["model_version"] == "1.0.0"
    assert data["request_id"] == "backend-test-001"
    assert captured["headers"]["X-Request-ID"] == "backend-test-001"
    assert captured["json"]["features"]["rooms_per_household"] == 6.0


def test_predict_invalid_input_returns_400(client):
    features = dict(VALID_FEATURES)
    features.pop("median_income")

    response = client.post("/api/predict", json={"features": features})

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_input"
