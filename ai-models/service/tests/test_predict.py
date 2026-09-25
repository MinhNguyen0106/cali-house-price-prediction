from pathlib import Path
import sys

from fastapi.testclient import TestClient

AI_MODELS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_MODELS_DIR))

from service.main import app  # noqa: E402


client = TestClient(app)

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


def test_predict_valid_input_returns_regression_prediction():
    response = client.post(
        "/predict",
        json={"features": VALID_FEATURES},
        headers={"X-Request-ID": "test-ai-001"},
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["prediction"], float)
    assert data["model_version"]
    assert data["request_id"] == "test-ai-001"
    assert data["target_column"] == "median_house_value"
