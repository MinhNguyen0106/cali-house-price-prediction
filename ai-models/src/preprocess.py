from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "housing.csv.zip"

TARGET_COLUMN = "median_house_value"
RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
REQUIRED_RAW_FEATURES = RAW_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
MODEL_FEATURE_NAMES = RAW_NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
MODEL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_FEATURES


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the project dataset from the original zip artifact."""
    return pd.read_csv(Path(path))


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def impute_total_bedrooms(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    strategy: str = "median",
):
    """Mirror the notebook: impute total_bedrooms before engineered ratios."""
    train_copy = X_train.copy()
    test_copy = X_test.copy() if X_test is not None else None
    imputer = SimpleImputer(strategy=strategy)

    train_copy["total_bedrooms"] = imputer.fit_transform(train_copy[["total_bedrooms"]])
    if test_copy is not None:
        test_copy["total_bedrooms"] = imputer.transform(test_copy[["total_bedrooms"]])
        return train_copy, test_copy, imputer
    return train_copy, imputer


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create the same ratio features used by the original notebook."""
    X_copy = X.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        X_copy["rooms_per_household"] = X_copy["total_rooms"] / X_copy["households"]
        X_copy["bedrooms_per_room"] = X_copy["total_bedrooms"] / X_copy["total_rooms"]
        X_copy["population_per_household"] = X_copy["population"] / X_copy["households"]
    X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
    return X_copy


def prepare_feature_frames(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_imputed, X_test_imputed, imputer = impute_total_bedrooms(X_train, X_test)
    return add_engineered_features(X_train_imputed), add_engineered_features(X_test_imputed), imputer


def build_preprocessor() -> ColumnTransformer:
    """Build the fitted-time preprocessing pipeline from ML_Project.ipynb."""
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, MODEL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_training_data(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_features_target(df)
    X_train_fe, X_test_fe, imputer = prepare_feature_frames(X_train, X_test)
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_fe)
    X_test_processed = preprocessor.transform(X_test_fe)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_fe": X_train_fe,
        "X_test_fe": X_test_fe,
        "y_train": y_train,
        "y_test": y_test,
        "imputer": imputer,
        "preprocessor": preprocessor,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
    }


def build_feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    feature_schema: dict[str, Any] = {}
    for column in REQUIRED_RAW_FEATURES + ENGINEERED_FEATURES:
        if column in CATEGORICAL_FEATURES:
            feature_schema[column] = {
                "type": "string",
                "allowed_values": sorted(df[column].dropna().astype(str).unique().tolist()),
                "description": f"{column} input feature.",
            }
            continue

        values = df[column].dropna()
        feature_schema[column] = {
            "type": "number",
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "description": f"{column} input feature.",
        }
    return feature_schema


def generate_schema(df: pd.DataFrame) -> dict[str, Any]:
    feature_df = add_engineered_features(df.drop(columns=[TARGET_COLUMN]).copy())
    return {
        "dataset_name": "California Housing",
        "target_column": TARGET_COLUMN,
        "required_raw_features": REQUIRED_RAW_FEATURES,
        "model_feature_names": MODEL_FEATURE_NAMES,
        "feature_schema": build_feature_schema(feature_df),
    }
