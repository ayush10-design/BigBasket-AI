"""🤖 AI Price Predictor: estimate a product's selling price from MRP and its catalogue labels."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COLORS, FEATURES
from src.model import MAIN_MODEL_NAME, predict_price
from src.state import get_dataset, get_model
from src.charts import show
from src.ui import key_value_rows, page_header, price_tag
from src.utils import inr, pct

NEW_VALUE = "✎ Enter a new value…"

df, info = get_dataset()
bundle = get_model(df, info)
metrics = bundle["metrics"]

page_header("AI Price Predictor",
            "Describe a product by its MRP, category, subcategory, brand and type. The Gradient Boosting "
            "model estimates the price BigBasket would sell it for.")


def ensure_valid(key: str, options: list, default=None) -> None:
    """Keep a selectbox's stored value valid when its option list changes."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]


def load_example() -> None:
    """Copy a real product's attributes into the form."""
    idx = st.session_state.get("pp_example")
    if idx is None:
        return
    row = df.loc[idx]
    st.session_state.update({
        "pp_mrp": float(row["MRP"]), "pp_cat": row["Category"], "pp_sub": row["Subcategory"],
        "pp_brand": row["Brand"], "pp_type": row["Product_Type"], "pp_all_brands": False,
        "pp_actual": float(row["Price"]), "pp_actual_name": row["Product"],
    })


def clear_actual() -> None:
    st.session_state.pop("pp_actual", None)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.1], gap="large")
with left:
    st.markdown("### Product details")
    test_rows = df.loc[[i for i in bundle["test_index"] if i in df.index]]
    sample = test_rows.sort_values("Product").head(400)
    st.selectbox(
        "Start from a real product (optional)", [None] + list(sample.index),
        format_func=lambda i: "Choose a held-out test product…" if i is None
        else f"{df.at[i, 'Product']} ({df.at[i, 'Brand']})",
        key="pp_example", on_change=load_example,
        help="These products were held out of training, so comparing the estimate with the real price is a fair test.",
    )

    if "pp_mrp" not in st.session_state:
        st.session_state.pp_mrp = float(round(df["MRP"].median()))
    st.number_input("MRP (₹)", min_value=1.0, max_value=1_000_000.0, step=10.0,
                    key="pp_mrp", on_change=clear_actual)

    cat_opts = sorted(df["Category"].unique()) + [NEW_VALUE]
    ensure_valid("pp_cat", cat_opts)
    st.selectbox("Category", cat_opts, key="pp_cat", on_change=clear_actual)
    category = st.text_input("New category", key="pp_cat_new") if st.session_state.pp_cat == NEW_VALUE \
        else st.session_state.pp_cat

    scope = df[df["Category"] == category]
    sub_opts = sorted((scope if not scope.empty else df)["Subcategory"].unique()) + [NEW_VALUE]
    ensure_valid("pp_sub", sub_opts)
    st.selectbox("Subcategory", sub_opts, key="pp_sub", on_change=clear_actual)
    subcategory = st.text_input("New subcategory", key="pp_sub_new") if st.session_state.pp_sub == NEW_VALUE \
        else st.session_state.pp_sub

    sub_scope = scope[scope["Subcategory"] == subcategory] if not scope.empty else scope
    show_all = st.checkbox("Show brands from every category", key="pp_all_brands")
    brand_source = df if show_all or sub_scope.empty else sub_scope
    brand_opts = sorted(brand_source["Brand"].unique()) + [NEW_VALUE]
    ensure_valid("pp_brand", brand_opts)
    st.selectbox("Brand", brand_opts, key="pp_brand", on_change=clear_actual)
    brand = st.text_input("New brand", key="pp_brand_new") if st.session_state.pp_brand == NEW_VALUE \
        else st.session_state.pp_brand

    type_source = sub_scope if not sub_scope.empty else (scope if not scope.empty else df)
    type_opts = sorted(type_source["Product_Type"].unique()) + [NEW_VALUE]
    ensure_valid("pp_type", type_opts)
    st.selectbox("Product type", type_opts, key="pp_type", on_change=clear_actual)
    product_type = st.text_input("New product type", key="pp_type_new") if st.session_state.pp_type == NEW_VALUE \
        else st.session_state.pp_type

inputs = {"MRP": st.session_state.pp_mrp, "Category": category, "Subcategory": subcategory,
          "Brand": brand, "Product_Type": product_type}

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
with right:
    st.markdown("### AI price estimate")
    try:
        result = predict_price(bundle, inputs)
    except Exception:
        st.error("The model could not score these inputs. Check that MRP is a positive number.")
        st.stop()

    price_tag(result, MAIN_MODEL_NAME, bundle["interval_coverage"])
    st.write("")

    rows = [
        ("MRP", inr(result["inputs"]["MRP"])),
        ("Estimated selling price", inr(result["prediction"])),
        ("Expected saving", inr(result["saving"]) if result["saving"] >= 0 else f"−{inr(-result['saving'])}"),
        ("Estimated discount", pct(result["discount_pct"])),
        ("Likely range (estimate)", f"{inr(result['low'])} to {inr(result['high'])}"),
    ]
    actual = st.session_state.get("pp_actual")
    if actual is not None:
        err = result["prediction"] - actual
        rows += [("Actual BigBasket price", inr(actual)),
                 ("Estimate error", f"{'+' if err >= 0 else '−'}{inr(abs(err))} ({abs(err) / actual * 100:.1f}%)")]
    key_value_rows(rows)

    if result["above_mrp"]:
        st.info("The estimate is above MRP. Retail prices in India cannot exceed MRP, so in practice the "
                "product would sell at MRP. The raw model output is shown for transparency.")
    for note in result["notes"]:
        st.warning(note)
    if result["range_widened"]:
        st.caption("The range was widened slightly so that it contains the point estimate.")

    st.caption(f"Model performance on {bundle['n_test']} held-out test products: "
               f"R² {metrics['R2']:.3f}, MAE {inr(metrics['MAE'])}, RMSE {inr(metrics['RMSE'])}.")

# ---------------------------------------------------------------------------
# Sensitivity: how the estimate moves with MRP for the same labels
# ---------------------------------------------------------------------------
st.markdown("### How the estimate changes with MRP")
mrp_now = float(inputs["MRP"])
grid = np.linspace(max(1.0, mrp_now * 0.25), mrp_now * 2.0, 60)
frame = pd.DataFrame([{**result["inputs"], "MRP": m} for m in grid])[FEATURES]
pred = bundle["pipeline"].predict(frame)
lo = bundle["q_low"].predict(frame)
hi = bundle["q_high"].predict(frame)
lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                         fillcolor="rgba(242,179,61,0.16)", line=dict(width=0), hoverinfo="skip",
                         name="Likely range"))
fig.add_trace(go.Scatter(x=grid, y=grid, mode="lines", name="MRP (no discount)",
                         line=dict(color=COLORS["muted"], dash="dot")))
fig.add_trace(go.Scatter(x=grid, y=pred, mode="lines", name="Estimated selling price",
                         line=dict(color=COLORS["turmeric"], width=3),
                         hovertemplate="MRP ₹%{x:,.0f}<br>Estimate ₹%{y:,.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[mrp_now], y=[result["prediction"]], mode="markers", name="Your product",
                         marker=dict(color=COLORS["basil"], size=12, line=dict(color="#0F1626", width=2))))
fig.update_layout(title="Estimated selling price across MRP values (other inputs fixed)",
                  xaxis_title="MRP (₹)", yaxis_title="Selling price (₹)",
                  legend=dict(orientation="h", y=-0.2))
show(fig, height=400)
st.caption("Gradient Boosting builds its prediction from decision trees, so the curve moves in steps "
           "rather than a straight line. The gap between the dotted MRP line and the estimate is the expected discount.")
