"""Model-ready data and the scikit-learn preprocessing pipeline.

Same design as the SI notebook:
* numeric feature (MRP): median imputation
* categorical features: most-frequent imputation + One-Hot Encoding with
  handle_unknown='ignore', so unseen brands/categories never crash prediction
"""
from __future__ import annotations

import inspect

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                        RANDOM_STATE, TARGET, TEST_SIZE)


def make_one_hot_encoder() -> OneHotEncoder:
    """Dense OneHotEncoder that works across scikit-learn versions."""
    args = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        args["sparse_output"] = False
    else:  # scikit-learn < 1.2
        args["sparse"] = False
    return OneHotEncoder(**args)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def get_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target. Rows without a valid sale price are dropped.

    Only the five input features are returned in X, so the target (Price) and
    anything derived from it (Price_Segment, Discount) can never leak in.
    """
    data = df[FEATURES + [TARGET]].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET] >= 0]
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(str)
    if len(data) < 10:
        raise ValueError("Fewer than 10 rows have a valid sale price; the model cannot be trained.")
    return data[FEATURES], data[TARGET]


def split_data(X: pd.DataFrame, y: pd.Series):
    """80/20 split with the notebook's random_state, so results are reproducible."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
