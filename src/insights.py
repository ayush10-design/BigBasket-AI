"""Data-driven insights generated with plain Python/pandas rules.

No external AI service is used: each insight is a calculation on the dataset,
and the thresholds (minimum group sizes, IQR fences) are stated in the text.
"""
from __future__ import annotations

import pandas as pd

from src.analytics import group_summary, iqr_bounds
from src.utils import inr, pct

MIN_BRAND_PRODUCTS = 3      # brands with fewer products are too noisy to rank
MIN_RATED_IN_CATEGORY = 5   # categories need this many ratings to be ranked
MIN_CATEGORY_FOR_OUTLIERS = 8


def correlation_strength(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "strong"


def rating_price_relation(df: pd.DataFrame) -> dict | None:
    rated = df.dropna(subset=["Rating", "Price"])
    if len(rated) < 10 or rated["Rating"].nunique() < 2:
        return None
    rho = float(rated["Rating"].corr(rated["Price"], method="spearman"))
    direction = "positive" if rho > 0 else "negative"
    return {"rho": rho, "n": len(rated), "strength": correlation_strength(rho), "direction": direction}


def price_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Products priced above their category's upper IQR fence."""
    rows = []
    for category, group in df.groupby("Category"):
        if len(group) < MIN_CATEGORY_FOR_OUTLIERS:
            continue
        _, upper = iqr_bounds(group["Price"])
        flagged = group[group["Price"] > upper].copy()
        if flagged.empty:
            continue
        median = group["Price"].median()
        flagged["Category median"] = median
        flagged["Times median"] = flagged["Price"] / median if median > 0 else float("nan")
        flagged["Upper fence"] = upper
        rows.append(flagged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).sort_values("Times median", ascending=False)


def high_discount_products(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Products whose discount is above the dataset-wide upper IQR fence."""
    _, fence = iqr_bounds(df["Discount"])
    if pd.isna(fence) or fence <= 0:
        fence = float(df["Discount"].quantile(0.95))
    flagged = df[df["Discount"] > fence].sort_values("Discount", ascending=False)
    return flagged, fence


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """Return insight cards: kind, title, value, detail."""
    insights = []
    cats = group_summary(df, "Category")
    if cats.empty:
        return insights

    top_price = cats.sort_values("Avg_Price", ascending=False).iloc[0]
    low_price = cats.sort_values("Avg_Price").iloc[0]
    insights.append({
        "kind": "price", "title": "Highest average selling price",
        "value": top_price["Category"],
        "detail": f"{inr(top_price['Avg_Price'])} on average across {int(top_price['Products'])} products.",
    })
    if len(cats) > 1:
        insights.append({
            "kind": "price", "title": "Lowest average selling price",
            "value": low_price["Category"],
            "detail": f"{inr(low_price['Avg_Price'])} on average across {int(low_price['Products'])} products.",
        })

    top_disc = cats.sort_values("Avg_Discount", ascending=False).iloc[0]
    insights.append({
        "kind": "discount", "title": "Deepest average discount",
        "value": top_disc["Category"],
        "detail": f"{pct(top_disc['Avg_Discount'])} average discount off MRP.",
    })

    rated = cats[cats["Rated"] >= MIN_RATED_IN_CATEGORY]
    if not rated.empty:
        best = rated.sort_values("Avg_Rating", ascending=False).iloc[0]
        insights.append({
            "kind": "rating", "title": "Best-rated category",
            "value": best["Category"],
            "detail": f"Average rating {best['Avg_Rating']:.2f} / 5 from {int(best['Rated'])} rated products "
                      f"(categories with at least {MIN_RATED_IN_CATEGORY} ratings).",
        })

    brands = group_summary(df, "Brand", min_products=MIN_BRAND_PRODUCTS)
    if not brands.empty:
        b = brands.sort_values("Avg_Price", ascending=False).iloc[0]
        insights.append({
            "kind": "brand", "title": "Most premium brand",
            "value": b["Brand"],
            "detail": f"{inr(b['Avg_Price'])} average price over {int(b['Products'])} products "
                      f"(brands with at least {MIN_BRAND_PRODUCTS} products).",
        })

    biggest = cats.iloc[0]
    insights.append({
        "kind": "volume", "title": "Largest category",
        "value": biggest["Category"],
        "detail": f"{int(biggest['Products'])} products, {biggest['Products'] / len(df) * 100:.1f}% of the catalogue.",
    })

    discounted = (df["Discount"] > 0).mean() * 100
    insights.append({
        "kind": "discount", "title": "Products sold below MRP",
        "value": pct(discounted),
        "detail": f"{int((df['Discount'] > 0).sum())} of {len(df)} products carry a discount; "
                  f"the median discount is {pct(df['Discount'].median())}.",
    })

    relation = rating_price_relation(df)
    if relation:
        insights.append({
            "kind": "relation", "title": "Rating and price",
            "value": f"ρ = {relation['rho']:.2f}",
            "detail": f"A {relation['strength']} {relation['direction']} rank correlation (Spearman) "
                      f"across {relation['n']} rated products.",
        })

    anomalies = price_anomalies(df)
    insights.append({
        "kind": "anomaly", "title": "Unusually expensive products",
        "value": f"{len(anomalies)} flagged",
        "detail": "Priced above their category's upper IQR fence "
                  f"(categories with at least {MIN_CATEGORY_FOR_OUTLIERS} products).",
    })

    deep, fence = high_discount_products(df)
    insights.append({
        "kind": "anomaly", "title": "Unusually high discounts",
        "value": f"{len(deep)} flagged",
        "detail": f"Discount above {pct(fence)}, the dataset's upper IQR fence.",
    })
    return insights
