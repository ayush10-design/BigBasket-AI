"""🔎 Product Intelligence: search, filter, sort and inspect individual products."""
import math

import streamlit as st

from src.analytics import filter_products
from src.model import MAIN_MODEL_NAME, predict_price
from src.state import get_dataset, get_model
from src.ui import empty_state, kpi_grid, page_header, product_cards
from src.utils import inr, pct, rating

df, info = get_dataset()
bundle = get_model(df, info)

page_header("Product Intelligence",
            "Search the catalogue, narrow it down with filters, and open any product to see how it "
            "compares with similar items.")

SORTS = {
    "Selling price: low to high": ("Price", True),
    "Selling price: high to low": ("Price", False),
    "Discount: highest first": ("Discount", False),
    "Rating: highest first": ("Rating", False),
    "MRP: high to low": ("MRP", False),
    "Name: A to Z": ("Product", True),
}

# ---------------------------------------------------------------------------
# Filters (each list only offers values that still exist given earlier choices)
# ---------------------------------------------------------------------------
search = st.text_input("Search products", placeholder="Try “green tea”, “Nivea” or “pressure cooker”",
                       label_visibility="collapsed")

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    categories = c1.multiselect("Category", sorted(df["Category"].unique()))
    scope = df[df["Category"].isin(categories)] if categories else df
    subcategories = c2.multiselect("Subcategory", sorted(scope["Subcategory"].unique()))
    scope = scope[scope["Subcategory"].isin(subcategories)] if subcategories else scope
    brands = c3.multiselect("Brand", sorted(scope["Brand"].unique()))
    scope = scope[scope["Brand"].isin(brands)] if brands else scope
    types = c4.multiselect("Product type", sorted(scope["Product_Type"].unique()))

    r1, r2, r3 = st.columns(3)
    pmax = float(math.ceil(df["Price"].max()))
    price_range = r1.slider("Selling price (₹)", 0.0, pmax, (0.0, pmax), step=float(max(1, round(pmax / 200))))
    dmax = float(math.ceil(df["Discount"].max())) if df["Discount"].notna().any() else 100.0
    discount_range = r2.slider("Discount (%)", 0.0, max(dmax, 1.0), (0.0, max(dmax, 1.0)), step=1.0)
    rating_range = r3.slider("Rating", 1.0, 5.0, (1.0, 5.0), step=0.1)
    include_unrated = r3.checkbox("Include products without a rating", value=True)

filtered = filter_products(
    df, categories=categories, subcategories=subcategories, brands=brands, product_types=types,
    price_range=price_range, rating_range=rating_range, include_unrated=include_unrated,
    discount_range=discount_range, search=search,
)

if filtered.empty:
    empty_state("No products match these filters",
                "Clear the search box or widen the price, discount or rating ranges to see products again.")
    st.stop()

kpi_grid([
    ("Matching products", f"{len(filtered):,}", f"of {len(df):,} in the catalogue"),
    ("Average selling price", inr(filtered["Price"].mean()), f"Median {inr(filtered['Price'].median())}"),
    ("Average discount", pct(filtered["Discount"].mean()), None),
    ("Average rating", f"{filtered['Rating'].mean():.2f}" if filtered["Rating"].notna().any() else "—",
     f"{int(filtered['Rating'].notna().sum())} rated"),
    ("Brands", f"{filtered['Brand'].nunique():,}", None),
])

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
t1, t2, t3 = st.columns([2, 1.3, 1])
sort_label = t1.selectbox("Sort by", list(SORTS))
view = t2.radio("View", ["Table", "Cards"], horizontal=True)
sort_col, ascending = SORTS[sort_label]
ordered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
t3.download_button("Download CSV", ordered.to_csv(index=False).encode("utf-8"),
                   file_name="bigbasket_filtered_products.csv", mime="text/csv", width="stretch")

if view == "Table":
    st.dataframe(
        ordered[["Product", "Brand", "Category", "Subcategory", "Product_Type", "MRP", "Price", "Discount", "Rating"]],
        hide_index=True, width="stretch", height=430,
        column_config={
            "Product": st.column_config.TextColumn("Product", width="large"),
            "Product_Type": "Product type",
            "MRP": st.column_config.NumberColumn("MRP", format="₹%.2f"),
            "Price": st.column_config.NumberColumn("Selling price", format="₹%.2f"),
            "Discount": st.column_config.ProgressColumn("Discount", format="%.1f%%", min_value=0, max_value=100),
            "Rating": st.column_config.NumberColumn("Rating", format="%.1f ★"),
        },
    )
else:
    per_page = 12
    pages = max(1, math.ceil(len(ordered) / per_page))
    page = st.number_input(f"Page (1–{pages})", 1, pages, 1) if pages > 1 else 1
    product_cards(ordered.iloc[(page - 1) * per_page: page * per_page])

# ---------------------------------------------------------------------------
# Product detail
# ---------------------------------------------------------------------------
st.markdown("### Product detail")
options = ordered.head(500)
labels = {idx: f"{row.Product} ({row.Brand})" for idx, row in options.iterrows()}
chosen = st.selectbox("Choose a product to inspect", list(labels), format_func=labels.get)
if len(ordered) > 500:
    st.caption("The list shows the first 500 products in the current sort order; use search to find others.")

row = df.loc[chosen]
st.markdown(f"#### {row['Product']}")
st.caption(f"{row['Brand']}, {row['Category']} › {row['Subcategory']} › {row['Product_Type']}")

estimate = predict_price(bundle, row.to_dict())
in_test = int(chosen) in set(bundle["test_index"])
kpi_grid([
    ("Selling price", inr(row["Price"]), None),
    ("MRP", inr(row["MRP"]), None),
    ("Discount", pct(row["Discount"]), f"You save {inr(row['MRP'] - row['Price'])}"),
    ("Rating", rating(row["Rating"]), None),
    ("Model estimate", inr(estimate["prediction"]),
     "Held-out test product" if in_test else "Product was used in training"),
], lead_first=True)

sub_peers = df[df["Subcategory"] == row["Subcategory"]]
pricier_share = float((sub_peers["Price"] > row["Price"]).mean() * 100)
cat_disc = df.loc[df["Category"] == row["Category"], "Discount"].mean()
lines = [
    f"Cheaper than {pricier_share:.0f}% of the {len(sub_peers)} products in {row['Subcategory']}."
    if len(sub_peers) > 1 else
    f"It is the only product in {row['Subcategory']} in this dataset.",
    f"Its discount is {pct(row['Discount'])} against a {row['Category']} average of {pct(cat_disc)}.",
    f"{MAIN_MODEL_NAME} estimate: {inr(estimate['prediction'])}, "
    f"{'above' if estimate['prediction'] > row['Price'] else 'below'} the actual price by "
    f"{inr(abs(estimate['prediction'] - row['Price']))}.",
]
for line in lines:
    st.markdown(f"- {line}")
if not in_test:
    st.caption("This product was part of the training data, so the model's estimate here is expected to be "
               "closer than on new products. Use the Model Lab for honest test-set accuracy.")
