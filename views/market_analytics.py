"""📊 Market Analytics: interactive charts of pricing, discounts and ratings."""
import math

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import filter_products, group_summary
from src.charts import bar_by_group, histogram, show
from src.config import CHART_SEQUENCE, COLORS
from src.insights import rating_price_relation
from src.state import get_dataset
from src.ui import empty_state, page_header
from src.utils import short_labels

df, info = get_dataset()

page_header("Market Analytics",
            "How prices, discounts and ratings vary across categories, brands and product types. "
            "Every chart responds to the filters below.")

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns([1.4, 1.4, 1.2])
    categories = c1.multiselect("Categories", sorted(df["Category"].unique()))
    scope = df[df["Category"].isin(categories)] if categories else df
    brands = c2.multiselect("Brands", sorted(scope["Brand"].unique()))
    pmax = float(math.ceil(df["Price"].max()))
    price_range = c3.slider("Selling price (₹)", 0.0, pmax, (0.0, pmax), step=float(max(1, round(pmax / 200))))

data = filter_products(df, categories=categories, brands=brands, price_range=price_range)
if data.empty:
    empty_state("No products in this selection", "Remove a category or brand filter, or widen the price range.")
    st.stop()
if len(data) < 5:
    st.info(f"Only {len(data)} products match, so some charts may not be meaningful.")

st.caption(f"Showing {len(data):,} of {len(df):,} products.")
cats = group_summary(data, "Category")
hover = ["Product", "Brand", "Subcategory"]


pricing, discounts, ratings, leaders = st.tabs(["Pricing", "Discounts", "Ratings", "Top products"])

# ---------------------------------------------------------------------------
with pricing:
    log_axes = st.toggle("Logarithmic axes", value=True,
                         help="Prices range from a few rupees to thousands; log axes keep both ends readable.")
    fig = px.scatter(data, x="MRP", y="Price", color="Category", hover_data=hover,
                     color_discrete_sequence=CHART_SEQUENCE + px.colors.qualitative.Pastel,
                     title="MRP vs selling price", log_x=log_axes, log_y=log_axes, opacity=0.8)
    lim = [max(1, data[["MRP", "Price"]].min().min()), data[["MRP", "Price"]].max().max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Price = MRP",
                             line=dict(color=COLORS["muted"], dash="dot"), hoverinfo="skip"))
    fig.update_layout(xaxis_title="MRP (₹)", yaxis_title="Selling price (₹)")
    show(fig, height=470)
    st.caption("Points on the dotted line sell at full MRP; points below it are discounted. "
               f"Pearson correlation between MRP and price: {data['MRP'].corr(data['Price']):.3f}.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(bar_by_group(cats, "Category", "Avg_Price", "Average selling price by category",
                          "Average selling price (₹)", color=COLORS["turmeric"],
                          hover_extra=["Products", "Median_Price"]), height=420)
    with c2:
        min_n = st.slider("Minimum products per brand", 1, 10, 3,
                          help="Brands with very few products give unreliable averages.")
        brand_sum = group_summary(data, "Brand", min_products=min_n)
        if brand_sum.empty:
            st.info("No brand has that many products in this selection; lower the minimum.")
        else:
            show(bar_by_group(brand_sum, "Brand", "Avg_Price",
                              f"Top 15 brands by average price (≥{min_n} products)",
                              "Average selling price (₹)", color=COLORS["lavender"], top=15), height=360)

    top_types = data["Product_Type"].value_counts().head(12).index
    type_data = data[data["Product_Type"].isin(top_types)]
    order = type_data.groupby("Product_Type")["Price"].median().sort_values(ascending=False).index
    fig = px.box(type_data, x="Product_Type", y="Price", points="outliers", category_orders={"Product_Type": list(order)},
                 title="Selling price by product type (12 most common types)", hover_data=["Product", "Brand"],
                 log_y=log_axes)
    fig.update_traces(marker_color=COLORS["sky"], line_color=COLORS["sky"])
    fig.update_layout(xaxis_title="Product type", yaxis_title="Selling price (₹)")
    show(fig, height=430)

    fig, hidden = histogram(data["Price"], "Selling price distribution", "Selling price (₹)", clip_quantile=0.99)
    show(fig, height=340)
    if hidden:
        st.caption(f"{hidden} products above the 99th percentile are hidden.")

# ---------------------------------------------------------------------------
with discounts:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(bar_by_group(cats, "Category", "Avg_Discount", "Average discount by category",
                          "Average discount (%)", color=COLORS["basil"], text_format=".1f"), height=420)
    with c2:
        fig, _ = histogram(data["Discount"], "Discount distribution", "Discount off MRP (%)", bins=30)
        show(fig, height=420)

    share = (data.assign(Discounted=data["Discount"] > 0).groupby("Category")["Discounted"].mean()
             .mul(100).sort_values().reset_index())
    fig = px.bar(share, x="Discounted", y="Category", orientation="h", text="Discounted",
                 title="Share of products sold below MRP, by category")
    fig.update_traces(marker_color=COLORS["sky"], texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Products with a discount (%)", yaxis_title=None, xaxis_range=[0, 110])
    show(fig, height=380)

# ---------------------------------------------------------------------------
with ratings:
    rated = data.dropna(subset=["Rating"])
    if len(rated) < 3:
        empty_state("Not enough ratings", "Fewer than three products in this selection have a rating.")
    else:
        fig = px.scatter(rated, x="Rating", y="Price", color="Category", hover_data=hover, log_y=True,
                         color_discrete_sequence=CHART_SEQUENCE + px.colors.qualitative.Pastel,
                         title="Rating vs selling price", opacity=0.8)
        fig.update_layout(xaxis_title="Customer rating (1–5)", yaxis_title="Selling price (₹, log scale)")
        show(fig, height=450)
        rel = rating_price_relation(rated)
        if rel:
            st.caption(f"Spearman rank correlation ρ = {rel['rho']:.3f} across {rel['n']} rated products: "
                       f"a {rel['strength']} {rel['direction']} relationship.")
        rated_cats = group_summary(rated, "Category")
        show(bar_by_group(rated_cats, "Category", "Avg_Rating", "Average rating by category",
                          "Average rating (1–5)", color=COLORS["lavender"], text_format=".2f",
                          hover_extra=["Rated"]), height=400)
    st.caption(f"{int(data['Rating'].isna().sum())} products in this selection have no rating.")

# ---------------------------------------------------------------------------
with leaders:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        top = short_labels(data.nlargest(10, "Price").iloc[::-1])
        fig = px.bar(top, x="Price", y="Label", orientation="h", hover_data=["Product", "Brand", "Category", "MRP"],
                     title="10 most expensive products", text="Price")
        fig.update_traces(marker_color=COLORS["turmeric"], texttemplate="₹%{text:,.0f}", textposition="outside",
                          cliponaxis=False)
        fig.update_layout(xaxis_title="Selling price (₹)", yaxis_title=None)
        show(fig, height=440)
    with c2:
        top = short_labels(data[data["Discount"] > 0].nlargest(10, "Discount").iloc[::-1])
        if top.empty:
            st.info("No discounted products in this selection.")
        else:
            fig = px.bar(top, x="Discount", y="Label", orientation="h",
                         hover_data=["Product", "Brand", "Category", "MRP", "Price"], title="10 most discounted products",
                         text="Discount")
            fig.update_traces(marker_color=COLORS["basil"], texttemplate="%{text:.1f}%", textposition="outside",
                              cliponaxis=False)
            fig.update_layout(xaxis_title="Discount off MRP (%)", yaxis_title=None)
            show(fig, height=440)

    show(bar_by_group(cats, "Category", "Median_Price", "Categories ranked by median selling price",
                      "Median selling price (₹)", color=COLORS["rose"], hover_extra=["Products", "Avg_Price"]),
         height=400)
    st.caption("The median is less affected by a few very expensive items than the average.")

