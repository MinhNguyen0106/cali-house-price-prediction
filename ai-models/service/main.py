import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
SCHEMA_PATH = BASE_DIR / "models" / "schema.json"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

logger = logging.getLogger("ai_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


schema = load_json(SCHEMA_PATH)
metadata = load_json(METADATA_PATH)

def build_model_feature_row(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_features, dict):
        raise ValueError("Payload must contain a JSON object under 'features'.")

    feature_schema = schema.get("feature_schema", {})
    normalized: Dict[str, Any] = {}

    for key, spec in feature_schema.items():
        if key not in raw_features:
            continue
        value = raw_features[key]
        field_type = spec.get("type")

        if field_type == "number":
            if value is None or value == "":
                raise ValueError(f"Field '{key}' is required.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - validation branch
                raise ValueError(f"Field '{key}' must be numeric.") from exc

            min_value = spec.get("min")
            max_value = spec.get("max")
            if min_value is not None and numeric_value < float(min_value):
                raise ValueError(f"Field '{key}' is below the allowed minimum ({min_value}).")
            if max_value is not None and numeric_value > float(max_value):
                raise ValueError(f"Field '{key}' exceeds the allowed maximum ({max_value}).")
            normalized[key] = numeric_value
        elif field_type == "string":
            if value is None:
                raise ValueError(f"Field '{key}' is required.")
            text_value = str(value).strip()
            allowed_values = spec.get("allowed_values")
            if allowed_values and text_value not in allowed_values:
                raise ValueError(f"Field '{key}' must be one of {allowed_values}.")
            normalized[key] = text_value

    required_fields = schema.get("required_raw_features", [])
    missing_fields = [field for field in required_fields if field not in normalized]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    if "total_rooms" in normalized and "households" in normalized:
        households = float(normalized["households"])
        normalized["rooms_per_household"] = float(normalized["total_rooms"]) / households if households != 0 else 0.0

    if "total_bedrooms" in normalized and "total_rooms" in normalized:
        total_rooms = float(normalized["total_rooms"])
        normalized["bedrooms_per_room"] = float(normalized["total_bedrooms"]) / total_rooms if total_rooms != 0 else 0.0

    if "population" in normalized and "households" in normalized:
        households = float(normalized["households"])
        normalized["population_per_household"] = float(normalized["population"]) / households if households != 0 else 0.0

    model_feature_names = schema.get("model_feature_names", [])
    for feature_name in model_feature_names:
        if feature_name not in normalized:
            continue

    final_row: Dict[str, Any] = {}
    for column_name in model_feature_names:
        if column_name not in normalized:
            raise ValueError(f"Model input column '{column_name}' is missing after feature preparation.")
        final_row[column_name] = normalized[column_name]

    return final_row


model = joblib.load(MODEL_PATH)

app = FastAPI(title="California Housing AI Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    logger.info("START %s %s request_id=%s", request.method, request.url.path, request_id)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("ERROR %s %s request_id=%s elapsed_ms=%s", request.method, request.url.path, request_id, elapsed_ms)
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info("END %s %s request_id=%s elapsed_ms=%s", request.method, request.url.path, request_id, elapsed_ms)
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "ai-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "target_column": metadata.get("target_column"),
        "metrics": metadata.get("metrics", {}),
        "environment": metadata.get("environment_versions", {}),
        "model_path": str(MODEL_PATH),
        "feature_count": len(schema.get("model_feature_names", [])),
        "schema_path": str(SCHEMA_PATH),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
async def predict(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    try:
        features = payload.get("features") if isinstance(payload, dict) else payload
        prepared = build_model_feature_row(features)
        df = pd.DataFrame([prepared])
        prediction = float(model.predict(df)[0])
        logger.info("prediction_success request_id=%s prediction=%s", request_id, prediction)
        return {
            "prediction": prediction,
            "model_version": metadata.get("model_version", "unknown"),
            "request_id": request_id,
            "target_column": metadata.get("target_column", "median_house_value"),
        }
    except ValueError as exc:
        logger.warning("prediction_invalid request_id=%s error=%s", request_id, str(exc))
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "detail": str(exc), "request_id": request_id}) from exc
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.exception("prediction_failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "detail": str(exc), "request_id": request_id}) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AI_SERVICE_PORT", "8001")))
