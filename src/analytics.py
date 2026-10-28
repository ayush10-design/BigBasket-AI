"""Descriptive statistics, filtering and the product Value Score.

Everything here is calculated from the loaded dataset; nothing is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kpis(df: pd.DataFrame) -> dict:
    return {
        "products": len(df),
        "categories": df["Category"].nunique(),
        "subcategories": df["Subcategory"].nunique(),
        "brands": df["Brand"].nunique(),
        "product_types": df["Product_Type"].nunique(),
        "avg_mrp": df["MRP"].mean(),
        "avg_price": df["Price"].mean(),
        "median_price": df["Price"].median(),
        "avg_discount": df["Discount"].mean(),
        "avg_rating": df["Rating"].mean(),
        "rated_products": int(df["Rating"].notna().sum()),
        "discounted_share": float((df["Discount"] > 0).mean() * 100) if len(df) else float("nan"),
    }


def group_summary(df: pd.DataFrame, by: str, min_products: int = 1) -> pd.DataFrame:
    """Per-group product count, average prices, discount and rating."""
    summary = (
        df.groupby(by)
        .agg(Products=("Product", "count"),
             Avg_Price=("Price", "mean"),
             Median_Price=("Price", "median"),
             Avg_MRP=("MRP", "mean"),
             Avg_Discount=("Discount", "mean"),
             Avg_Rating=("Rating", "mean"),
             Rated=("Rating", "count"))
        .reset_index()
    )
    return summary[summary["Products"] >= min_products].sort_values("Products", ascending=False)


def filter_products(df: pd.DataFrame, *, categories=None, subcategories=None, brands=None,
                    product_types=None, price_range=None, rating_range=None,
                    include_unrated=True, discount_range=None, search="") -> pd.DataFrame:
    """Apply every filter used by the exploration pages. Empty selections mean 'all'."""
    out = df
    for col, chosen in [("Category", categories), ("Subcategory", subcategories),
                        ("Brand", brands), ("Product_Type", product_types)]:
        if chosen:
            out = out[out[col].isin(chosen)]
    if price_range:
        out = out[out["Price"].between(*price_range)]
    if discount_range:
        out = out[out["Discount"].between(*discount_range)]
    if rating_range:
        in_range = out["Rating"].between(*rating_range)
        out = out[in_range | (out["Rating"].isna() & include_unrated)]
    if search and search.strip():
        term = search.strip().lower()
        haystack = (out["Product"].astype(str) + " " + out["Brand"].astype(str) + " "
                    + out["Product_Type"].astype(str)).str.lower()
        out = out[haystack.str.contains(term, regex=False)]
    return out


def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey's IQR fences, the same outlier rule used in the SI notebook."""
    values = values.dropna()
    if values.empty:
        return float("nan"), float("nan")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def value_scores(products: pd.DataFrame, max_discount: float, discount_weight: float) -> pd.DataFrame:
    """Transparent Value Score (0-100).

    Discount score = Discount % / highest discount in the dataset × 100
    Rating score   = Rating / 5 × 100
    Value Score    = w × Discount score + (1 − w) × Rating score

    If a product has no rating, its score uses the discount score only and is
    flagged, because a missing rating cannot be scored fairly.
    """
    out = products.copy()
    w = float(np.clip(discount_weight, 0, 1))
    out["Discount score"] = (out["Discount"] / max_discount * 100) if max_discount > 0 else 0.0
    out["Rating score"] = out["Rating"] / 5 * 100
    blended = w * out["Discount score"] + (1 - w) * out["Rating score"]
    out["Value Score"] = blended.where(out["Rating"].notna(), out["Discount score"])
    out["Score basis"] = np.where(out["Rating"].notna(), "Discount + rating", "Discount only (no rating)")
    return out
