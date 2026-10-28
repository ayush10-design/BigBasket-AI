"""BigBasket AI: Smart Product Intelligence & Price Prediction.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException

st.set_page_config(
    page_title="BigBasket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import DATA_DIR, DEFAULT_SAVE_NAME  # noqa: E402
from src.data_loader import (REQUIRED_COLUMNS, DatasetNotFoundError,  # noqa: E402
                             DatasetSchemaError, clean_dataset)
from src.state import get_dataset, get_model  # noqa: E402
from src.ui import inject_css, page_header, sidebar_brand, sidebar_status  # noqa: E402

inject_css()
sidebar_brand()

PAGES = [
    st.Page("views/command_center.py", title="Command Center", default=True),
    st.Page("views/product_intelligence.py", title="Product Intelligence"),
    st.Page("views/price_predictor.py", title="AI Price Predictor"),
    st.Page("views/market_analytics.py", title="Market Analytics"),
    st.Page("views/product_comparison.py", title="Product Comparison"),
    st.Page("views/ai_insights.py", title="AI Insights"),
    st.Page("views/model_lab.py", title="Model Lab"),
    st.Page("views/hypothesis_testing.py", title="Hypothesis Testing"),
]


def setup_page(problem: str) -> None:
    """Shown when data/ has no usable CSV. Saving a file here is a one-time step."""
    page_header("Add the project dataset",
                "BigBasket AI reads its data from the data/ folder. Add the cleaned CSV from the SI notebook once and the app loads it automatically from then on.")
    st.warning(problem)
    st.markdown(
        f"**Option 1.** Copy `{DEFAULT_SAVE_NAME}` (saved by section 10 of the notebook) into "
        f"`{DATA_DIR.name}/`, then refresh this page.\n\n"
        "**Option 2.** Upload it below; it is saved into the data folder for you."
    )
    st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS)
               + ". Rating, Item_ID and Price_Segment are optional. The notebook's raw column names "
                 "(product_name, sale_price, market_price…) are also accepted.")
    uploaded = st.file_uploader("Dataset CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            clean, _ = clean_dataset(raw)
        except DatasetSchemaError as exc:
            st.error(str(exc))
            return
        except Exception:
            st.error("That file could not be read as a CSV. Check that it is a comma-separated text file.")
            return
        st.success(f"Read {len(raw):,} rows; {len(clean):,} are usable after cleaning.")
        if st.button("Save to data folder and open the app", type="primary"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / DEFAULT_SAVE_NAME).write_bytes(uploaded.getvalue())
            st.cache_data.clear()
            st.rerun()


def error_page(message: str, detail: str = "") -> None:
    page_header("Something needs attention", "The app could not finish loading.")
    st.error(message)
    if detail:
        with st.expander("Technical details"):
            st.code(detail)


# ---------------------------------------------------------------------------
# Load data and model once; every page reuses the cached results.
# ---------------------------------------------------------------------------
df = info = bundle = None
try:
    df, info = get_dataset()
except (DatasetNotFoundError, DatasetSchemaError) as exc:
    problem = str(exc)

    def dataset_setup():
        setup_page(problem)
    navigation = st.navigation([st.Page(dataset_setup, title="Set up dataset")])
except Exception as exc:  # unreadable file, encoding problem, etc.
    data_error = repr(exc)

    def dataset_error():
        error_page("The dataset file exists but could not be read.", data_error)
    navigation = st.navigation([st.Page(dataset_error, title="Dataset error")])
else:
    try:
        bundle = get_model(df, info)
        navigation = st.navigation(PAGES)
    except Exception as exc:
        model_error = repr(exc)

        def model_failure():
            error_page("The price model could not be trained on this dataset.", model_error)
        navigation = st.navigation([st.Page(model_failure, title="Model error")])

sidebar_status(info.path.name if info else None, info.clean_rows if info else None,
               bundle["metrics"]["R2"] if bundle else None)

try:
    navigation.run()
except ScriptControlException:
    raise  # st.stop() / st.rerun() must pass through
except Exception as exc:
    st.error("This page hit an unexpected problem. Try changing the filters or reloading the page.")
    with st.expander("Technical details"):
        st.exception(exc)
