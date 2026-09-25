from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def evaluate_model(
    model_name: str,
    model_obj: Any,
    X_train,
    y_train,
    X_test,
    y_test,
    best_params: dict[str, Any] | None = None,
    cv_rmse: float | None = None,
) -> dict[str, Any]:
    start_train = time.time()
    model_obj.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_predict = time.time()
    y_pred = model_obj.predict(X_test)
    predict_time = time.time() - start_predict

    metrics = evaluate_regression(y_test, y_pred)
    return {
        "Model": model_name,
        "Best Hyperparameters": best_params or "N/A",
        "CV RMSE": round(cv_rmse, 4) if cv_rmse is not None else "N/A",
        "Test RMSE": round(metrics["rmse"], 4),
        "Test MAE": round(metrics["mae"], 4),
        "Test R2": round(metrics["r2"], 4),
        "Train Time (s)": round(train_time, 4),
        "Predict Time (s)": round(predict_time, 4),
    }


def compare_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("Test RMSE").reset_index(drop=True)


def residual_frame(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": np.asarray(y_true) - np.asarray(y_pred),
        }
    )
