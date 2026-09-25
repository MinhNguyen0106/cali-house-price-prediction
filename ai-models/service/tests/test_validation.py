from pathlib import Path
import sys

from fastapi.testclient import TestClient

AI_MODELS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_MODELS_DIR))

from service.main import app, model  # noqa: E402


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


def test_missing_feature_returns_400():
    features = dict(VALID_FEATURES)
    features.pop("median_income")

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_input"


def test_wrong_type_returns_400():
    features = dict(VALID_FEATURES)
    features["median_income"] = "not-a-number"

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_input"


def test_invalid_category_returns_400():
    features = dict(VALID_FEATURES)
    features["ocean_proximity"] = "MOUNTAIN"

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_input"


def test_model_is_loaded_once_at_module_level():
    assert model is not None
