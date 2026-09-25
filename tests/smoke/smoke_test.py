from __future__ import annotations

import os
import sys

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")

SAMPLE_PAYLOAD = {
    "features": {
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
}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    if not BACKEND_URL:
        fail("BACKEND_URL environment variable is required.")

    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=15)
        if health.status_code != 200:
            fail(f"GET /health returned HTTP {health.status_code}")

        response = requests.post(f"{BACKEND_URL}/api/predict", json=SAMPLE_PAYLOAD, timeout=30)
        if response.status_code != 200:
            fail(f"POST /api/predict returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        for key in ["prediction", "model_version", "request_id"]:
            if key not in data:
                fail(f"Missing key in prediction response: {key}")

        print("PASS")
        print(f"prediction={data['prediction']}")
        print(f"model_version={data['model_version']}")
        print(f"request_id={data['request_id']}")
    except requests.RequestException as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
