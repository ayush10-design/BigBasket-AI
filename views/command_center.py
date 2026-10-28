"""🏠 Command Center: catalogue overview, KPIs and a live market snapshot."""
import plotly.express as px
import streamlit as st

from src.analytics import group_summary, kpis
from src.charts import bar_by_group, histogram, show
from src.config import COLORS
from src.insights import rating_price_relation
from src.state import get_dataset, get_model
from src.ui import html, kpi_grid, page_header
from src.utils import esc, inr, num, pct

df, info = get_dataset()
bundle = get_model(df, info)
k = kpis(df)

page_header("Command Center",
            f"A live overview of {k['products']:,} BigBasket products across "
            f"{k['categories']} categories, calculated from {info.path.name}.")

kpi_grid([
    ("Total products", num(k["products"]), f"{k['subcategories']} subcategories"),
    ("Categories", num(k["categories"]), None),
    ("Brands", num(k["brands"]), f"{k['product_types']} product types"),
    ("Average MRP", inr(k["avg_mrp"]), None),
    ("Average selling price", inr(k["avg_price"]), f"Median {inr(k['median_price'])}"),
    ("Average discount", pct(k["avg_discount"]), f"{pct(k['discounted_share'], 0)} of products discounted"),
    ("Average rating", f"{k['avg_rating']:.2f}" if k["rated_products"] else "—",
     f"From {k['rated_products']:,} rated products"),
], lead_first=True)

# ---------------------------------------------------------------------------
# Market snapshot: every sentence is computed from the data
# ---------------------------------------------------------------------------
cats = group_summary(df, "Category")
priciest = cats.sort_values("Avg_Price", ascending=False).iloc[0]
cheapest = cats.sort_values("Avg_Price").iloc[0]
deepest = cats.sort_values("Avg_Discount", ascending=False).iloc[0]
largest = cats.iloc[0]
relation = rating_price_relation(df)

lines = [
    f"<b>{esc(priciest['Category'])}</b> has the highest average selling price at {inr(priciest['Avg_Price'])}, "
    f"while <b>{esc(cheapest['Category'])}</b> is the most affordable at {inr(cheapest['Avg_Price'])}.",
    f"<b>{esc(largest['Category'])}</b> is the largest category with {int(largest['Products'])} products "
    f"({largest['Products'] / len(df) * 100:.0f}% of the catalogue).",
    f"<b>{esc(deepest['Category'])}</b> offers the deepest average discount, {pct(deepest['Avg_Discount'])} off MRP.",
    f"The price model explains <b>{bundle['metrics']['R2'] * 100:.1f}%</b> of the variation in selling price on "
    f"unseen test products, with a typical error of {inr(bundle['metrics']['MAE'])}.",
]
if relation:
    lines.append(f"Ratings and prices show a <b>{relation['strength']}</b> relationship "
                 f"(Spearman ρ = {relation['rho']:.2f}), so a higher price does not reliably mean a better rating."
                 if relation["strength"] in ("negligible", "weak") else
                 f"Ratings and prices show a <b>{relation['strength']} {relation['direction']}</b> relationship "
                 f"(Spearman ρ = {relation['rho']:.2f}).")

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown("### Market snapshot")
    html('<div class="bb-panel"><ul class="bb-snapshot">' + "".join(f"<li>{x}</li>" for x in lines) + "</ul></div>")
with right:
    top = cats.head(8).iloc[::-1]
    fig = px.bar(top, x="Products", y="Category", orientation="h", title="Top categories by product count",
                 text="Products")
    fig.update_traces(marker_color=COLORS["sky"], textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Number of products", yaxis_title=None,
                      xaxis_range=[0, top["Products"].max() * 1.15])
    show(fig, height=330)

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
st.markdown("### Prices, discounts and ratings")
c1, c2 = st.columns(2, gap="medium")
with c1:
    fig, hidden = histogram(df["Price"], "Selling price distribution", "Selling price (₹)", clip_quantile=0.99)
    show(fig)
    if hidden:
        st.caption(f"{hidden} products above the 99th percentile are hidden to keep the shape readable.")
with c2:
    fig = bar_by_group(cats, "Category", "Avg_Price", "Average selling price by category",
                       "Average selling price (₹)", color=COLORS["turmeric"])
    show(fig)

c3, c4 = st.columns(2, gap="medium")
with c3:
    fig, _ = histogram(df["Discount"], "Discount distribution", "Discount off MRP (%)",
                       color=COLORS["basil"], bins=30)
    show(fig)
    st.caption(f"{int((df['Discount'] == 0).sum())} products sell at full MRP (0% discount).")
with c4:
    if df["Rating"].notna().any():
        fig, _ = histogram(df["Rating"], "Rating distribution", "Customer rating (1–5)",
                           color=COLORS["lavender"], bins=20)
        show(fig)
        st.caption(f"{int(df['Rating'].isna().sum())} products have no rating and are not shown.")
    else:
        st.info("This dataset has no ratings to chart.")
