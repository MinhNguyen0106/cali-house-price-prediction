from pathlib import Path
import sys

from fastapi.testclient import TestClient

AI_MODELS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_MODELS_DIR))

from service.main import app  # noqa: E402


client = TestClient(app)


def test_model_info_reads_metadata():
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"]
    assert data["model_version"]
    assert data["target_column"] == "median_house_value"
    assert "metrics" in data
    assert data["feature_count"] > 0
