"""🧠 AI Insights: findings generated automatically from the data and the trained model."""
import plotly.express as px
import streamlit as st

from src.analytics import group_summary
from src.charts import show
from src.config import COLORS
from src.insights import (MIN_CATEGORY_FOR_OUTLIERS, generate_insights,
                          high_discount_products, price_anomalies, rating_price_relation)
from src.state import get_dataset, get_model
from src.ui import empty_state, insight_cards, page_header
from src.utils import pct

df, info = get_dataset()
bundle = get_model(df, info)

page_header("AI Insights",
            "Findings generated automatically by rules applied to the dataset and the trained model. "
            "Each one is recalculated whenever the data changes; nothing here is written by hand.")

insight_cards(generate_insights(df))

money = {"MRP": st.column_config.NumberColumn("MRP", format="₹%.2f"),
         "Price": st.column_config.NumberColumn("Selling price", format="₹%.2f"),
         "Discount": st.column_config.NumberColumn("Discount", format="%.1f%%"),
         "Rating": st.column_config.NumberColumn("Rating", format="%.1f")}

tab1, tab2, tab3, tab4 = st.tabs(["Category scorecard", "Price anomalies", "Deep discounts", "Model surprises"])

with tab1:
    cats = group_summary(df, "Category")
    st.dataframe(
        cats, hide_index=True, width="stretch",
        column_config={
            "Avg_Price": st.column_config.NumberColumn("Avg price", format="₹%.2f"),
            "Median_Price": st.column_config.NumberColumn("Median price", format="₹%.2f"),
            "Avg_MRP": st.column_config.NumberColumn("Avg MRP", format="₹%.2f"),
            "Avg_Discount": st.column_config.NumberColumn("Avg discount", format="%.1f%%"),
            "Avg_Rating": st.column_config.NumberColumn("Avg rating", format="%.2f"),
            "Rated": "Rated products",
        },
    )
    fig = px.scatter(cats, x="Avg_Discount", y="Avg_Price", size="Products", color="Category", text="Category",
                     title="Category positioning: average discount vs average price",
                     hover_data=["Products", "Avg_Rating"], size_max=48)
    fig.update_traces(textposition="top center", textfont=dict(size=11, color=COLORS["muted"]))
    fig.update_layout(xaxis_title="Average discount (%)", yaxis_title="Average selling price (₹)", showlegend=False)
    show(fig, height=480)
    st.caption("Bubble size shows the number of products. Top-right categories are expensive and discount heavily; "
               "bottom-left ones are cheap with small discounts.")

with tab2:
    anomalies = price_anomalies(df)
    st.markdown(f"Products priced above the upper IQR fence (Q3 + 1.5 × IQR) of their own category. "
                f"Only categories with at least {MIN_CATEGORY_FOR_OUTLIERS} products are checked. "
                "A flag means *unusual*, not *wrong*: premium items legitimately sit here.")
    if anomalies.empty:
        empty_state("No price anomalies", "No product is priced above its category's upper fence.")
    else:
        flagged_ids = set(anomalies.index)
        plot = df.assign(Status=["Flagged" if i in flagged_ids else "Typical" for i in df.index])
        fig = px.strip(plot, x="Price", y="Category", color="Status", hover_data=["Product", "Brand"],
                       color_discrete_map={"Typical": "rgba(108,180,238,0.35)", "Flagged": COLORS["rose"]},
                       log_x=True, title="Selling prices by category, with anomalies highlighted")
        fig.update_layout(xaxis_title="Selling price (₹, log scale)", yaxis_title=None,
                          legend=dict(orientation="h", y=-0.15))
        show(fig, height=480)
        st.dataframe(
            anomalies[["Product", "Brand", "Category", "Price", "Category median", "Times median", "Upper fence"]],
            hide_index=True, width="stretch",
            column_config={**money,
                           "Category median": st.column_config.NumberColumn(format="₹%.2f"),
                           "Upper fence": st.column_config.NumberColumn(format="₹%.2f"),
                           "Times median": st.column_config.NumberColumn("× median", format="%.1f×")},
        )

with tab3:
    deep, fence = high_discount_products(df)
    st.markdown(f"Products with a discount above **{pct(fence)}**, the dataset's upper IQR fence for discounts.")
    if deep.empty:
        empty_state("No unusually high discounts", "Every discount is within the normal range for this dataset.")
    else:
        st.dataframe(deep[["Product", "Brand", "Category", "MRP", "Price", "Discount", "Rating"]],
                     hide_index=True, width="stretch", column_config=money)
        by_cat = deep["Category"].value_counts().reset_index()
        by_cat.columns = ["Category", "Products"]
        fig = px.bar(by_cat.iloc[::-1], x="Products", y="Category", orientation="h", text="Products",
                     title="Where the deepest discounts are")
        fig.update_traces(marker_color=COLORS["basil"], textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Products above the discount fence", yaxis_title=None)
        show(fig, height=320)

with tab4:
    st.markdown("Held-out test products whose real price differs most from the model's estimate. These are the "
                "cases the model finds hardest, often because MRP alone does not explain the price.")
    results = bundle["test_results"].copy()
    results.index = bundle["test_index"]
    results = results.join(df[["Product"]], how="left")
    worst = results.sort_values("Absolute error", ascending=False).head(15)
    st.dataframe(
        worst[["Product", "Brand", "Category", "MRP", "Actual", "Predicted", "Absolute error"]],
        hide_index=True, width="stretch",
        column_config={"MRP": money["MRP"],
                       "Actual": st.column_config.NumberColumn("Actual price", format="₹%.2f"),
                       "Predicted": st.column_config.NumberColumn("Predicted", format="₹%.2f"),
                       "Absolute error": st.column_config.NumberColumn("Error", format="₹%.2f")},
    )
    share = results["Absolute error"].le(results["Actual"] * 0.10).mean() * 100
    st.caption(f"For {share:.0f}% of the {len(results)} test products the estimate is within 10% of the real price.")

rel = rating_price_relation(df)
if rel:
    st.markdown("### Does a higher price mean a better rating?")
    st.markdown(
        f"Across {rel['n']} rated products the Spearman rank correlation between rating and price is "
        f"**{rel['rho']:.3f}**, a **{rel['strength']}** {rel['direction']} relationship. "
        + ("In this catalogue, paying more does not reliably buy a better-rated product."
           if rel["strength"] in ("negligible", "weak")
           else "Price and rating tend to move together in this catalogue.")
    )
