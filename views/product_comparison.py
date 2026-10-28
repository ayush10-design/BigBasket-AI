"""⚔️ Product Comparison: put 2–4 products side by side and score their value transparently."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import value_scores
from src.charts import show
from src.config import COLORS
from src.state import get_dataset
from src.ui import empty_state, html, page_header
from src.utils import esc, inr, pct, rating, short_labels

df, info = get_dataset()

page_header("Product Comparison",
            "Pick two to four products to compare price, discount and rating, then see which one offers "
            "better value under a formula you can adjust.")

labels = {idx: f"{r.Product} ({r.Brand})" for idx, r in df.sort_values("Product").iterrows()}

c1, c2 = st.columns([1, 2.4], gap="medium")
with c1:
    sub = st.selectbox("Quick pick from a subcategory", [None] + sorted(df["Subcategory"].unique()),
                       format_func=lambda s: "Choose a subcategory…" if s is None else s)
    if sub and st.button("Load the 3 best-rated products", width="stretch"):
        picks = (df[df["Subcategory"] == sub].sort_values(["Rating", "Discount"], ascending=False,
                                                          na_position="last").head(3).index.tolist())
        st.session_state.cmp_selection = picks
with c2:
    selection = st.multiselect("Products to compare", list(labels), format_func=labels.get,
                               max_selections=4, key="cmp_selection",
                               placeholder="Search by product or brand name")

if len(selection) < 2:
    empty_state("Choose at least two products",
                "Search for products above, or pick a subcategory and load its best-rated items.")
    st.stop()

items = short_labels(df.loc[selection].copy(), max_len=28)

# ---------------------------------------------------------------------------
# Side-by-side table
# ---------------------------------------------------------------------------
st.markdown("### Side by side")
header = "".join(f'<td class="bb-col-head">{esc(lbl)}</td>' for lbl in items["Label"])
fields = [
    ("Brand", lambda r: esc(r["Brand"])),
    ("Category", lambda r: esc(r["Category"])),
    ("Subcategory", lambda r: esc(r["Subcategory"])),
    ("Product type", lambda r: esc(r["Product_Type"])),
    ("MRP", lambda r: inr(r["MRP"])),
    ("Selling price", lambda r: inr(r["Price"])),
    ("Saving", lambda r: inr(r["MRP"] - r["Price"])),
    ("Discount", lambda r: pct(r["Discount"])),
    ("Rating", lambda r: rating(r["Rating"])),
]
body = "".join(
    f"<tr><td>{name}</td>" + "".join(f"<td>{fn(r)}</td>" for _, r in items.iterrows()) + "</tr>"
    for name, fn in fields
)
html(f"""
<div class="bb-panel" style="overflow-x:auto">
<table class="bb-rows bb-compare">
<tr><td></td>{header}</tr>
{body}
</table></div>
""")

# ---------------------------------------------------------------------------
# Visual comparison
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2, gap="medium")
with c1:
    fig = go.Figure([
        go.Bar(name="MRP", x=items["Label"], y=items["MRP"], marker_color=COLORS["muted"],
               hovertemplate="MRP ₹%{y:,.2f}<extra></extra>"),
        go.Bar(name="Selling price", x=items["Label"], y=items["Price"], marker_color=COLORS["turmeric"],
               hovertemplate="Selling price ₹%{y:,.2f}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", title="MRP and selling price", yaxis_title="₹", xaxis_title=None,
                      legend=dict(orientation="h", y=-0.25))
    show(fig, height=380)
with c2:
    fig = go.Figure([
        go.Bar(name="Discount (%)", x=items["Label"], y=items["Discount"], marker_color=COLORS["basil"],
               hovertemplate="Discount %{y:.1f}%<extra></extra>"),
        go.Bar(name="Rating × 20", x=items["Label"], y=items["Rating"] * 20, marker_color=COLORS["lavender"],
               customdata=items["Rating"], hovertemplate="Rating %{customdata:.1f} / 5<extra></extra>"),
    ])
    fig.update_layout(barmode="group", title="Discount and rating (rating rescaled to 0–100)",
                      yaxis_title="Score (0–100)", xaxis_title=None, legend=dict(orientation="h", y=-0.25))
    show(fig, height=380)

# ---------------------------------------------------------------------------
# Value Score
# ---------------------------------------------------------------------------
st.markdown("### Value Score")
max_disc = float(df["Discount"].max())
weight = st.slider("Weight given to discount (the rest goes to rating)", 0, 100, 50, step=5,
                   format="%d%%") / 100

with st.expander("How the Value Score is calculated", expanded=False):
    st.markdown(
        f"""
- **Discount score** = product discount ÷ highest discount in the dataset ({pct(max_disc)}) × 100
- **Rating score** = rating ÷ 5 × 100
- **Value Score** = {weight:.2f} × Discount score + {1 - weight:.2f} × Rating score

A product without a rating is scored on discount only and marked as such. Price itself is not part of the score,
because a ₹50 spice and a ₹2,000 cooker are not cheaper or dearer *versions* of each other.
This is a simple, transparent rule, not a machine-learning prediction.
"""
    )

scored = value_scores(items, max_disc, weight).sort_values("Value Score", ascending=False)
fig = go.Figure(go.Bar(
    x=scored["Value Score"], y=scored["Label"], orientation="h",
    marker_color=[COLORS["basil"]] + [COLORS["sky"]] * (len(scored) - 1),
    text=scored["Value Score"].round(1), textposition="outside", cliponaxis=False,
    customdata=scored[["Score basis"]].to_numpy(),
    hovertemplate="Value Score %{x:.1f}<br>Basis: %{customdata[0]}<extra></extra>",
))
fig.update_layout(title="Value Score (0–100, higher is better)", xaxis_title="Value Score", yaxis_title=None,
                  yaxis=dict(autorange="reversed"), xaxis_range=[0, max(105, scored["Value Score"].max() * 1.15)])
show(fig, height=120 + 60 * len(scored))

best, runner = scored.iloc[0], scored.iloc[1]
gap = best["Value Score"] - runner["Value Score"]
if gap < 1:
    conclusion = (f"**{best['Label']}** and **{runner['Label']}** are effectively tied "
                  f"(Value Scores {best['Value Score']:.1f} and {runner['Value Score']:.1f}).")
else:
    reasons = []
    if best["Discount"] > runner["Discount"]:
        reasons.append(f"a larger discount ({pct(best['Discount'])} vs {pct(runner['Discount'])})")
    if pd.notna(best["Rating"]) and (pd.isna(runner["Rating"]) or best["Rating"] > runner["Rating"]):
        reasons.append(f"a higher rating ({rating(best['Rating'])} vs {rating(runner['Rating'])})")
    why = " thanks to " + " and ".join(reasons) if reasons else ""
    conclusion = (f"**{best['Label']}** offers the best value with a score of {best['Value Score']:.1f}, "
                  f"{gap:.1f} points ahead of {runner['Label']}{why}.")
st.success(conclusion)

notes = []
if items["Category"].nunique() > 1:
    notes.append("These products come from different categories, so they may not be direct substitutes.")
if scored["Rating"].isna().any():
    notes.append("At least one product has no rating; its score is based on discount only.")
cheapest = items.loc[items["Price"].idxmin()]
notes.append(f"Lowest selling price: {cheapest['Label']} at {inr(cheapest['Price'])}.")
for n in notes:
    st.caption(n)
