from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

try:
    from .evaluate import compare_results, evaluate_model
    from .preprocess import (
        DATA_PATH,
        TARGET_COLUMN,
        generate_schema,
        load_dataset,
        prepare_training_data,
    )
except ImportError:  # Allows: python ai-models/src/train.py
    from evaluate import compare_results, evaluate_model
    from preprocess import (
        DATA_PATH,
        TARGET_COLUMN,
        generate_schema,
        load_dataset,
        prepare_training_data,
    )

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"

DT_PARAM_GRID = {
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 10, 20],
}
RF_PARAM_DIST = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}
GB_PARAM_DIST = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}


def build_default_models() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_split=20, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
        ),
    }


def tune_models(X_train, y_train) -> dict[str, Any]:
    tuned: dict[str, Any] = {"Linear Regression": (LinearRegression(), None, None)}

    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid=DT_PARAM_GRID,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    tuned["Decision Tree"] = (
        DecisionTreeRegressor(**dt_grid.best_params_, random_state=42),
        dt_grid.best_params_,
        float(np.sqrt(-dt_grid.best_score_)),
    )

    rf_random = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=RF_PARAM_DIST,
        n_iter=6,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(X_train, y_train)
    tuned["Random Forest"] = (
        RandomForestRegressor(**rf_random.best_params_, random_state=42, n_jobs=-1),
        rf_random.best_params_,
        float(np.sqrt(-rf_random.best_score_)),
    )

    gb_random = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=GB_PARAM_DIST,
        n_iter=6,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    gb_random.fit(X_train, y_train)
    tuned["Gradient Boosting"] = (
        GradientBoostingRegressor(**gb_random.best_params_, random_state=42),
        gb_random.best_params_,
        float(np.sqrt(-gb_random.best_score_)),
    )
    return tuned


def train_models(data_path: str | Path = DATA_PATH, run_search: bool = False) -> dict[str, Any]:
    df = load_dataset(data_path)
    prepared = prepare_training_data(df)
    X_train = prepared["X_train_processed"]
    X_test = prepared["X_test_processed"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    if run_search:
        model_specs = tune_models(X_train, y_train)
    else:
        model_specs = {name: (model, None, None) for name, model in build_default_models().items()}

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for name, (model, params, cv_rmse) in model_specs.items():
        result = evaluate_model(name, model, X_train, y_train, X_test, y_test, params, cv_rmse)
        results.append(result)
        fitted_models[name] = model

    comparison = compare_results(results)
    best_name = str(comparison.iloc[0]["Model"])
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", prepared["preprocessor"]),
            ("model", fitted_models[best_name]),
        ]
    )

    return {
        "dataframe": df,
        "prepared": prepared,
        "comparison": comparison,
        "models": fitted_models,
        "best_model_name": best_name,
        "pipeline": final_pipeline,
    }


def build_metadata(best_model_name: str, best_model: Any, comparison: pd.DataFrame) -> dict[str, Any]:
    best_row = comparison[comparison["Model"] == best_model_name].iloc[0]
    return {
        "model_name": best_model.__class__.__name__,
        "model_version": "1.0.0",
        "task": "California Housing Price Prediction",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": TARGET_COLUMN,
        "metrics": {
            "test_rmse": float(best_row["Test RMSE"]),
            "test_mae": float(best_row["Test MAE"]),
            "test_r2": float(best_row["Test R2"]),
        },
        "environment_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_artifacts(training_result: dict[str, Any], output_dir: str | Path = MODELS_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_result["pipeline"], output_dir / "model.joblib", compress=3)

    best_model = training_result["models"][training_result["best_model_name"]]
    metadata = build_metadata(training_result["best_model_name"], best_model, training_result["comparison"])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = generate_schema(training_result["dataframe"])
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train California Housing regression models.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--search", action="store_true", help="Run CV hyperparameter search from the notebook.")
    parser.add_argument("--save-artifacts", action="store_true", help="Overwrite model.joblib/schema/metadata.")
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    result = train_models(args.data_path, run_search=args.search)
    print(result["comparison"].to_string(index=False))
    print(f"Best model: {result['best_model_name']}")

    if args.save_artifacts:
        save_artifacts(result, args.output_dir)
        print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
