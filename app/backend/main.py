import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pymongo
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_DIR = Path(__file__).resolve().parent
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")
AI_SERVICE_TIMEOUT_SECONDS = float(os.getenv("AI_SERVICE_TIMEOUT_SECONDS", "30"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "cali_house_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "predictions")
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

logger = logging.getLogger("backend_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def resolve_schema_path() -> Path:
    configured_path = os.getenv("MODEL_SCHEMA_PATH") or os.getenv("SCHEMA_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    candidates = [
        APP_DIR / "schema.json",
        APP_DIR / "ai-models" / "models" / "schema.json",
    ]
    if len(APP_DIR.parents) > 1:
        candidates.append(APP_DIR.parents[1] / "ai-models" / "models" / "schema.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


SCHEMA_PATH = resolve_schema_path()


def parse_cors_origins() -> List[str]:
    raw_value = os.getenv("CORS_ORIGINS")
    if not raw_value:
        return DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


schema = load_schema()

def validate_features(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must include a JSON object under 'features'.")

    feature_schema = schema.get("feature_schema", {})
    normalized: Dict[str, Any] = {}
    for key, spec in feature_schema.items():
        if key not in payload:
            continue
        value = payload[key]
        field_type = spec.get("type")
        if field_type == "number":
            if value is None or value == "":
                raise ValueError(f"Field '{key}' is required.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
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

    return normalized


app = FastAPI(title="California Housing Backend", version="1.0.0")
cors_origins = parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    logger.info("backend req=%s start method=%s path=%s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("backend req=%s error elapsed_ms=%s", request_id, elapsed_ms)
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info("backend req=%s status=%s total_duration_ms=%s", request_id, response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    mongo_status = "unknown"
    try:
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"unavailable: {exc}"
    return {
        "status": "ok",
        "service": "backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ai_service_url": AI_SERVICE_URL,
        "mongodb_status": mongo_status,
        "schema_path": str(SCHEMA_PATH),
    }


@app.get("/model-info")
def get_model_info() -> Dict[str, Any]:
    try:
        response = requests.get(f"{AI_SERVICE_URL}/model-info", timeout=20)
        response.raise_for_status()
        data = response.json()
        return {"status": "ok", "source": "ai-service", **data}
    except requests.RequestException as exc:
        logger.exception("Failed to fetch model info from AI service")
        raise HTTPException(status_code=502, detail={"error": "ai_service_unavailable", "detail": str(exc)}) from exc


@app.get("/api/history")
def prediction_history() -> Dict[str, Any]:
    try:
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        collection = client[MONGODB_DATABASE][MONGODB_COLLECTION]
        docs = list(collection.find({}, {"_id": 0}).sort("created_at", -1).limit(10))
        return {"status": "ok", "data": docs}
    except Exception as exc:  # pragma: no cover - DB connectivity issue
        logger.exception("History query failed")
        raise HTTPException(status_code=500, detail={"error": "history_unavailable", "detail": str(exc)}) from exc


@app.post("/api/predict")
async def predict(request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", request.headers.get("x-request-id") or uuid.uuid4().hex)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid_json", "detail": "Request payload must be valid JSON.", "request_id": request_id}) from exc

    features = payload.get("features") if isinstance(payload, dict) else payload
    try:
        validated_features = validate_features(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "detail": str(exc), "request_id": request_id}) from exc
    logger.info("backend req=%s validation OK", request_id)

    try:
        logger.info("backend req=%s calling ai-service", request_id)
        ai_response = requests.post(
            f"{AI_SERVICE_URL}/predict",
            json={"features": validated_features},
            headers={"X-Request-ID": request_id},
            timeout=AI_SERVICE_TIMEOUT_SECONDS,
        )
        ai_response.raise_for_status()
    except requests.Timeout as exc:
        logger.exception("AI service timeout for request_id=%s", request_id)
        raise HTTPException(status_code=504, detail={"error": "ai_service_timeout", "detail": str(exc), "request_id": request_id}) from exc
    except requests.RequestException as exc:
        logger.exception("AI service request failed for request_id=%s", request_id)
        raise HTTPException(status_code=502, detail={"error": "ai_service_unavailable", "detail": str(exc), "request_id": request_id}) from exc

    ai_result = ai_response.json()
    record = {
        "request_id": request_id,
        "features": validated_features,
        "prediction": ai_result.get("prediction"),
        "model_version": ai_result.get("model_version"),
        "target_column": ai_result.get("target_column", "median_house_value"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "backend",
    }

    try:
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        collection = client[MONGODB_DATABASE][MONGODB_COLLECTION]
        collection.insert_one(record)
    except Exception as exc:  # pragma: no cover - DB connectivity issue
        logger.exception("Failed to save prediction record for request_id=%s", request_id)

    final_response = {
        "prediction": ai_result.get("prediction"),
        "model_version": ai_result.get("model_version"),
        "request_id": request_id,
        "target_column": ai_result.get("target_column", "median_house_value"),
    }
    return JSONResponse(status_code=200, content=final_response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", "8000")))
