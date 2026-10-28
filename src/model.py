"""Train, evaluate, save and use the sale-price models.

Main model: GradientBoostingRegressor (same settings as the SI notebook).
Baselines for comparison: LinearRegression and RandomForestRegressor, trained
on the identical preprocessing and the identical train/test split.
Prediction range: two extra Gradient Boosting models trained with quantile
loss (10th and 90th percentile). Their real coverage is measured on the test
set and reported, rather than assumed.
"""
from __future__ import annotations

import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (CATEGORICAL_FEATURES, CV_FOLDS, FEATURES, GB_PARAMS,
                        INTERVAL_HIGH_Q, INTERVAL_LOW_Q, MODEL_FILE,
                        NUMERIC_FEATURES, RANDOM_STATE, TARGET)
from src.preprocessing import build_preprocessor, get_model_data, split_data

MAIN_MODEL_NAME = "Random Forest Regressor"


def make_pipeline(regressor) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("regressor", regressor)])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Random Forest Regressor": make_pipeline(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        ),
        "Gradient Boosting Regressor": make_pipeline(
            GradientBoostingRegressor(**GB_PARAMS)
        ),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def grouped_feature_importance(pipeline: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (importance per original feature, top individual encoded columns)."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["regressor"].feature_importances_
    detail = pd.DataFrame({"Encoded feature": names, "Importance": importances})

    def original_feature(name: str) -> str:
        stripped = name.split("__", 1)[-1]
        if stripped in NUMERIC_FEATURES:
            return stripped
        for col in CATEGORICAL_FEATURES:
            if stripped.startswith(col + "_"):
                return col
        return stripped

    detail["Feature"] = detail["Encoded feature"].map(original_feature)
    detail["Encoded feature"] = detail["Encoded feature"].str.split("__", n=1).str[-1]
    grouped = (detail.groupby("Feature", as_index=False)["Importance"].sum()
               .sort_values("Importance", ascending=False))
    top = detail.sort_values("Importance", ascending=False).head(15)
    return grouped.reset_index(drop=True), top.reset_index(drop=True)


def train_models(df: pd.DataFrame, fingerprint: str = "", progress=None) -> dict:
    """Train every model, evaluate on the held-out test set and build a bundle.

    `progress` is an optional callback(fraction, message) for the UI.
    """
    def report(fraction, message):
        if progress:
            progress(fraction, message)

    started = time.time()
    X, y = get_model_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    models = candidate_models()
    for i, (name, pipe) in enumerate(models.items()):
        report(0.1 + 0.2 * i, f"Training {name}")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        test_m = regression_metrics(y_test, pipe.predict(X_test))
        train_m = regression_metrics(y_train, pipe.predict(X_train))
        cv_scores = cross_validate(
            clone(pipe), X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error"),
        )
        rows.append({
            "Model": name,
            "MAE": test_m["MAE"], "RMSE": test_m["RMSE"], "R2": test_m["R2"],
            "Train R2": train_m["R2"],
            "CV R2 mean": float(np.mean(cv_scores["test_r2"])),
            "CV R2 std": float(np.std(cv_scores["test_r2"])),
            "CV MAE mean": float(-np.mean(cv_scores["test_neg_mean_absolute_error"])),
        })

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_by_rmse = comparison.iloc[0]["Model"]

    report(0.75, "Training prediction-range models")
    q_low = make_pipeline(GradientBoostingRegressor(loss="quantile", alpha=INTERVAL_LOW_Q, **GB_PARAMS))
    q_high = make_pipeline(GradientBoostingRegressor(loss="quantile", alpha=INTERVAL_HIGH_Q, **GB_PARAMS))
    q_low.fit(X_train, y_train)
    q_high.fit(X_train, y_train)

    main = fitted[MAIN_MODEL_NAME]
    y_pred = main.predict(X_test)
    low, high = q_low.predict(X_test), q_high.predict(X_test)
    low, high = np.minimum(low, high), np.maximum(low, high)
    coverage = float(np.mean((y_test.to_numpy() >= low) & (y_test.to_numpy() <= high)))

    test_results = X_test.copy()
    test_results["Actual"] = y_test.to_numpy()
    test_results["Predicted"] = y_pred
    test_results["Range low"] = low
    test_results["Range high"] = high
    test_results["Residual"] = test_results["Actual"] - test_results["Predicted"]
    test_results["Absolute error"] = test_results["Residual"].abs()

    grouped_imp, top_imp = grouped_feature_importance(main)
    report(0.95, "Saving results")

    main_row = comparison.set_index("Model").loc[MAIN_MODEL_NAME]
    return {
        "pipeline": main,
        "q_low": q_low,
        "q_high": q_high,
        "metrics": {"MAE": main_row["MAE"], "RMSE": main_row["RMSE"], "R2": main_row["R2"],
                    "Train R2": main_row["Train R2"]},
        "comparison": comparison,
        "best_by_rmse": best_by_rmse,
        "test_results": test_results.reset_index(drop=True),
        "interval_coverage": coverage,
        "feature_importance": grouped_imp,
        "feature_importance_top": top_imp,
        "train_frame": X_train.reset_index(drop=True),
        "test_index": [int(i) for i in X_test.index],
        "n_train": len(X_train), "n_test": len(X_test), "n_total": len(X),
        "meta": {
            "fingerprint": fingerprint,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "training_seconds": round(time.time() - started, 1),
            "gb_params": GB_PARAMS,
            "features": FEATURES,
            "target": TARGET,
        },
    }


def save_bundle(bundle: dict, path=MODEL_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_saved_bundle(fingerprint: str, path=MODEL_FILE) -> dict | None:
    """Load the saved bundle only if it was trained on this exact dataset and
    with this scikit-learn version; otherwise return None (retrain needed)."""
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
    if meta.get("fingerprint") != fingerprint or meta.get("sklearn_version") != sklearn.__version__:
        return None
    return bundle


def get_or_train(df: pd.DataFrame, fingerprint: str, progress=None) -> tuple[dict, bool]:
    """Return (bundle, was_trained_now)."""
    bundle = load_saved_bundle(fingerprint)
    if bundle is not None:
        return bundle, False
    bundle = train_models(df, fingerprint, progress)
    try:
        save_bundle(bundle)
    except OSError:
        pass  # read-only folder: the model still works for this session
    return bundle, True


def check_inputs(bundle: dict, inputs: dict) -> list[str]:
    """Explain anything about the inputs that makes the estimate less reliable."""
    train = bundle["train_frame"]
    notes = []
    unseen = [col.replace("_", " ") for col in CATEGORICAL_FEATURES
              if str(inputs[col]) not in set(train[col].astype(str))]
    if unseen:
        notes.append(
            "Not seen during training: " + ", ".join(unseen)
            + ". The encoder ignores unseen values, so the estimate leans on MRP and the other inputs."
        )
    else:
        mask = np.ones(len(train), dtype=bool)
        for col in CATEGORICAL_FEATURES:
            mask &= train[col].astype(str).to_numpy() == str(inputs[col])
        if not mask.any():
            notes.append("Each value appears in the training data, but never in this exact combination.")
    lo, hi = float(train["MRP"].min()), float(train["MRP"].max())
    if not lo <= float(inputs["MRP"]) <= hi:
        notes.append(
            f"MRP is outside the training range (₹{lo:,.0f} to ₹{hi:,.0f}). "
            "Tree models do not extrapolate, so the estimate is less reliable."
        )
    return notes


def predict_price(bundle: dict, inputs: dict) -> dict:
    """Predict the sale price for one product described by the 5 features."""
    row = {"MRP": float(inputs["MRP"])}
    for col in CATEGORICAL_FEATURES:
        value = inputs.get(col)
        value = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
        row[col] = value if value else "Unknown"
    frame = pd.DataFrame([row])[FEATURES]

    point = float(bundle["pipeline"].predict(frame)[0])
    low = float(bundle["q_low"].predict(frame)[0])
    high = float(bundle["q_high"].predict(frame)[0])
    low, high = min(low, high), max(low, high)
    widened = point < low or point > high
    low, high = min(low, point), max(high, point)

    mrp = row["MRP"]
    return {
        "prediction": point,
        "low": max(low, 0.0),
        "high": high,
        "range_widened": widened,
        "saving": mrp - point,
        "discount_pct": (mrp - point) / mrp * 100 if mrp > 0 else float("nan"),
        "above_mrp": point > mrp,
        "notes": check_inputs(bundle, row),
        "inputs": row,
    }
