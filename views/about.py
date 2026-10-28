"""ℹ️ About Project: problem, objectives, dataset, method, stack, limitations and team."""
import platform

import pandas as pd
import plotly
import sklearn
import streamlit as st

from src import project_info as P
from src.analytics import kpis
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.state import get_dataset, get_model
from src.ui import empty_state, html, kpi_grid, page_header
from src.utils import esc, inr

df, info = get_dataset()
bundle = get_model(df, info)
k = kpis(df)
m = bundle["metrics"]

page_header("BigBasket AI: Smart Product Intelligence & Price Prediction",
            "AI-powered e-commerce analytics and product price intelligence platform. "
            f"{P.COURSE}, {P.PROGRAMME}, {P.INSTITUTION}.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### Problem statement")
    st.markdown(
        "Online grocery catalogues list thousands of products whose selling price differs from the printed MRP "
        "by varying discounts. Sellers and analysts need to know what a fair selling price looks like for a "
        "product, and how prices, discounts and ratings vary across categories and brands. Doing this by hand "
        "does not scale."
    )
    st.markdown("### Objectives")
    st.markdown(
        "- Load, clean and explore the BigBasket product dataset.\n"
        "- Analyse prices, discounts and ratings across categories, brands and product types.\n"
        f"- Train a regression model that predicts the selling price ({TARGET}) from MRP and catalogue labels.\n"
        "- Compare it with baseline models using MAE, RMSE and R² on a held-out test set.\n"
        "- Provide an interactive GUI for exploration, comparison and price prediction."
    )
with c2:
    st.markdown("### Dataset overview")
    kpi_grid([
        ("Products", f"{k['products']:,}", None),
        ("Categories", f"{k['categories']}", None),
        ("Brands", f"{k['brands']:,}", None),
        ("Rated products", f"{k['rated_products']:,}", None),
    ])
    st.markdown(
        f"**Source:** {P.DATASET_SOURCE}. Live scraping of BigBasket returned HTTP 403, so the project uses the "
        f"supplied CSV (`{info.path.name}`) rather than live data.\n\n"
        "**Columns:** Item_ID, Product, Category, Subcategory, Brand, Price (selling price), MRP, Product_Type, "
        "Rating, Price_Segment, and Discount (recalculated as (MRP − Price) ÷ MRP × 100)."
    )

st.markdown("### Machine learning approach")
st.markdown(
    f"- **Task:** regression, predicting {TARGET} (₹).\n"
    f"- **Features:** {', '.join(NUMERIC_FEATURES)} (numeric) and "
    f"{', '.join(c.replace('_', ' ') for c in CATEGORICAL_FEATURES)} (categorical).\n"
    "- **Preprocessing:** scikit-learn Pipeline with median / most-frequent imputation and one-hot encoding "
    "(unknown values ignored).\n"
    "- **Models:** Gradient Boosting Regressor (main model, from the SI notebook), with Linear Regression and "
    "Random Forest as baselines.\n"
    f"- **Result:** test R² {m['R2']:.3f}, MAE {inr(m['MAE'])}, RMSE {inr(m['RMSE'])} on "
    f"{bundle['n_test']} held-out products. See the Model Lab for the full comparison."
)

st.markdown("### Project workflow")
steps = ["Load dataset", "Clean and validate", "Explore and visualise", "Select features",
         "Train and compare models", "Evaluate on test set", "Interactive GUI"]
html('<div class="bb-panel"><ol style="margin:0;padding-left:1.2rem;line-height:1.9;color:#CDD6E4">'
     + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ol></div>")

st.markdown("### Technology stack")
st.dataframe(pd.DataFrame([
    ("Python", platform.python_version(), "Programming language"),
    ("Pandas", pd.__version__, "Data loading, cleaning and aggregation"),
    ("Scikit-learn", sklearn.__version__, "Preprocessing pipeline, models and metrics"),
    ("Plotly", plotly.__version__, "Interactive charts"),
    ("Streamlit", st.__version__, "Web GUI"),
], columns=["Tool", "Version", "Purpose"]), hide_index=True, width="stretch")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### Limitations")
    st.markdown(
        "- The data is a static sample, not live BigBasket prices.\n"
        "- MRP dominates the model; product labels only fine-tune the discount.\n"
        "- Brands and product types with one or two products give the model little to learn from.\n"
        "- About a third of products have no rating, which limits rating-based analysis.\n"
        "- Prediction ranges are empirical estimates, not statistical guarantees."
    )
with c2:
    st.markdown("### Future scope")
    st.markdown(
        "- Train on the full product list for wider brand and type coverage.\n"
        "- Add text features from product names and descriptions (TF-IDF or embeddings).\n"
        "- Tune hyper-parameters with grid or randomised search.\n"
        "- Track prices over time to forecast discounts.\n"
        "- Deploy publicly on Streamlit Community Cloud."
    )

st.markdown("### Team")
if P.TEAM:
    st.dataframe(pd.DataFrame(P.TEAM).rename(columns={"name": "Student name", "roll_no": "Roll no.",
                                                      "enrollment_no": "Enrollment no.", "role": "Role"}),
                 hide_index=True, width="stretch")
else:
    empty_state("Add your team details",
                "Open src/project_info.py and fill in the TEAM list with each member's name, roll number and "
                "enrollment number.")
details = [(label, value) for label, value in [
    ("Department", P.DEPARTMENT), ("Academic year", P.ACADEMIC_YEAR), ("Guide", P.GUIDE),
    ("GitHub", P.GITHUB_URL)] if value]
st.caption("  |  ".join(f"{a}: {b}" for a, b in details))
