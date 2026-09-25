from pathlib import Path
import sys

from fastapi.testclient import TestClient

AI_MODELS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_MODELS_DIR))

from service.main import app  # noqa: E402


client = TestClient(app)


def test_health_reports_loaded_model():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ai-service"
    assert data["model_loaded"] is True
