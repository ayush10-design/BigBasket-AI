"""Project-wide settings: paths, feature definitions and the colour palette.

Every other module imports from here, so changing a path or a feature list
only needs to happen in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project folder, so the app runs from anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# The notebook saves its cleaned output as `bigbasket_cleaned.csv`.
# The loader looks for these names first, then falls back to any CSV in data/.
PREFERRED_DATA_FILES = [
    "bigbasket_cleaned.csv",
    "dataset.csv",
    "bigbasket_sample - data.csv",
    "BigBasket Products.csv",
]
DEFAULT_SAVE_NAME = "bigbasket_cleaned.csv"

MODEL_FILE = MODELS_DIR / "price_model_bundle.joblib"

# ---------------------------------------------------------------------------
# Machine-learning set-up (identical to the SI notebook, section 11)
# ---------------------------------------------------------------------------
TARGET = "Price"
NUMERIC_FEATURES = ["MRP"]
CATEGORICAL_FEATURES = ["Category", "Subcategory", "Brand", "Product_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Price_Segment is NOT a feature: it may be derived from Price (target leakage).
LEAKAGE_COLUMNS = ["Price_Segment", "Discount"]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyper-parameters copied from the notebook's GradientBoostingRegressor
GB_PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)

# Prediction range: 10th and 90th percentile quantile-loss models (an 80% range)
INTERVAL_LOW_Q = 0.10
INTERVAL_HIGH_Q = 0.90

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
COLORS = {
    "ink": "#0F1626",
    "panel": "#172238",
    "panel_2": "#1D2A44",
    "text": "#E6ECF5",
    "muted": "#93A3BE",
    "basil": "#5BC489",
    "turmeric": "#F2B33D",
    "sky": "#6CB4EE",
    "rose": "#EE7D8F",
    "lavender": "#A99CF0",
    "grid": "rgba(147,163,190,0.14)",
}
CHART_SEQUENCE = [
    COLORS["basil"], COLORS["turmeric"], COLORS["sky"],
    COLORS["rose"], COLORS["lavender"], "#4FD1C5", "#F59E6B", "#9BD36A",
]

APP_TITLE = "BigBasket AI"
APP_SUBTITLE = "Smart Product Intelligence & Price Prediction"
