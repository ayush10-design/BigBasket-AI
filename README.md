# BigBasket AI — Smart Product Intelligence & Price Prediction

AI-powered e-commerce analytics and product price intelligence platform, built with
Python, Streamlit, Pandas, Scikit-learn and Plotly.

SI Lab project, B.Tech. CSE (Artificial Intelligence and Data Science), Ramdeobaba University, Nagpur.

---

## 1. Project overview

The app turns the SI notebook (`notebook/SI_Project.ipynb`) into a multi-page analytics product.
It explores 1,249 BigBasket products and uses a **Gradient Boosting Regressor** to predict a
product's **selling price (`Price`)** from its **MRP, Category, Subcategory, Brand and Product Type**.

Every number in the app (KPIs, insights, metrics, feature importance, predictions) is calculated
from the dataset and the trained model when the app runs. Nothing is hard-coded.

## 2. Features

| Page | What it does |
|---|---|
| 🏠 Command Center | KPIs, a data-generated market snapshot, price, discount and rating distributions |
| 🔎 Product Intelligence | Search, cascading filters, sorting, table or card view, CSV export, product detail with peer comparison |
| 🤖 AI Price Predictor | Estimated selling price, saving, discount, likely range, input checks, MRP sensitivity curve |
| 📊 Market Analytics | Interactive charts: MRP vs price, category/brand/type pricing, discounts, ratings, top products |
| ⚔️ Product Comparison | Compare 2–4 products side by side with a transparent, adjustable Value Score |
| 🧠 AI Insights | Rule-based insight cards, category scorecard, price anomalies, deep discounts, model surprises |
| 🧪 Model Lab | Model card, 3-model comparison, cross-validation, actual vs predicted, residuals, feature importance |
| ℹ️ About Project | Problem statement, objectives, dataset, method, stack, limitations, future scope, team |

## 3. Tech stack

Python 3.10+, Streamlit (GUI), Pandas and NumPy (data), Scikit-learn (pipeline, models, metrics),
Plotly (interactive charts), Joblib (model saving).

## 4. Project structure

```
BigBasket_AI/
├── app.py                    # Entry point: navigation, data/model loading, error handling
├── requirements.txt
├── README.md
├── .streamlit/config.toml    # Dark theme
├── data/
│   ├── bigbasket_cleaned.csv # The project dataset (1,249 rows)
│   └── README.md             # Accepted file names and columns
├── models/
│   └── price_model_bundle.joblib  # Trained models + evaluation results (auto-created)
├── views/                    # One file per page
│   ├── command_center.py
│   ├── product_intelligence.py
│   ├── price_predictor.py
│   ├── market_analytics.py
│   ├── product_comparison.py
│   ├── ai_insights.py
│   ├── model_lab.py
│   └── about.py
├── src/
│   ├── config.py             # Paths, features, hyper-parameters, colours
│   ├── data_loader.py        # Find, load and clean the dataset
│   ├── preprocessing.py      # sklearn ColumnTransformer, feature/target split
│   ├── model.py              # Training, evaluation, saving, prediction
│   ├── analytics.py          # KPIs, group summaries, filters, Value Score
│   ├── insights.py           # Rule-based insight generation
│   ├── charts.py             # Plotly theme and chart helpers
│   ├── ui.py                 # HTML components (KPI cards, price tag…)
│   ├── state.py              # Streamlit caching
│   ├── utils.py              # Formatting helpers
│   └── project_info.py       # ← edit your team details here
├── assets/style.css          # Custom styling
├── scripts/train_model.py    # Train and print metrics from the terminal
├── tests/smoke_test.py       # Runs every page headlessly
└── notebook/SI_Project.ipynb # Original Colab notebook
```

The page files live in `views/` rather than `pages/` on purpose: Streamlit automatically turns a
folder called `pages/` into its own navigation, which would bypass `app.py` (and its styling and
error handling) when a page is opened directly by URL.

## 5. Dataset

* **File:** `data/bigbasket_cleaned.csv`, 1,249 rows × 11 columns, produced by the SI notebook.
* **Source:** a sample of the public Kaggle "BigBasket Entire Product List" dataset. Live scraping of
  BigBasket returned HTTP 403, so the project uses the supplied CSV.
* **Columns:** Item_ID, Product, Category, Subcategory, Brand, Price (selling price), MRP,
  Product_Type, Rating, Price_Segment, Discount.
* **Summary:** 11 categories, 83 subcategories, 689 brands, 289 product types. Rating is missing for
  394 products (31.55%); no other column has missing values and there are no duplicate rows.

## 6. Machine learning methodology

1. **Cleaning** (same as the notebook): trim text, drop rows without a product name, drop exact
   duplicates, fill missing labels with "Unknown", treat negative prices as missing, recalculate
   `Discount = (MRP − Price) / MRP × 100`.
2. **Features:** `MRP` (numeric); `Category`, `Subcategory`, `Brand`, `Product_Type` (categorical).
   **Target:** `Price`.
3. **Leakage prevention:** `Price_Segment` and `Discount` are derived from `Price`, so they are never
   used as inputs.
4. **Pipeline:** `ColumnTransformer` with median imputation for MRP and most-frequent imputation +
   `OneHotEncoder(handle_unknown="ignore")` for the categorical features. Unseen brands or
   categories therefore never crash a prediction.
5. **Split:** 80/20 train/test, `random_state=42` → 999 training and 250 testing rows.
6. **Models:** Gradient Boosting Regressor (`n_estimators=150, learning_rate=0.05, max_depth=2`,
   the notebook's settings), plus Linear Regression and Random Forest (200 trees) as baselines.
7. **Evaluation:** MAE, RMSE and R² on the test set, and 5-fold cross-validation on all rows.
8. **Prediction range:** two extra Gradient Boosting models with quantile loss (10th and 90th
   percentiles). Their real coverage on the test set is measured and shown.

## 7. Installation

You need Python 3.10 or newer.

```bash
cd BigBasket_AI
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## 8. How to run

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. A trained model is included in `models/`, so the app starts
immediately. If the dataset changes, or your scikit-learn version differs, the app retrains
automatically on first load, which takes about a minute.

## 9. Example commands

```bash
# Train all models and print the evaluation table (useful for the report)
python scripts/train_model.py

# Run every page headlessly to check nothing is broken
python tests/smoke_test.py
```

## 10. Model evaluation

Results measured on this dataset with scikit-learn 1.8.0 (`python scripts/train_model.py`).
Your numbers may differ in the last decimal with a different scikit-learn version; the Model Lab
always shows the values from your own run.

| Model | Test MAE (₹) | Test RMSE (₹) | Test R² | CV R² (5-fold) |
|---|---|---|---|---|
| Random Forest Regressor | 49.830 | 165.960 | 0.837 | 0.879 ± 0.031 |
| **Gradient Boosting Regressor** | **57.434** | **183.705** | **0.800** | 0.889 ± 0.047 |
| Linear Regression | 70.359 | 189.224 | 0.788 | 0.888 ± 0.066 |

* The Gradient Boosting results exactly match the original notebook (MAE 57.434, RMSE 183.705, R² 0.800).
* On the single test split, Random Forest has the lowest RMSE. Across 5-fold cross-validation, Gradient
  Boosting has the highest mean R². The app keeps Gradient Boosting as the prediction model, consistent
  with the notebook, and reports the comparison exactly as measured.
* Feature importance: MRP 0.984, Brand 0.009, Product Type 0.004, Subcategory 0.003, Category 0.0002.
* The 10th–90th percentile prediction range covered 74.4% of test prices (nominal target: 80%).

## 11. Screenshots

Add screenshots for the report here, for example:

* `screenshots/command_center.png`: dashboard and KPIs
* `screenshots/price_predictor.png`: GUI input and prediction output
* `screenshots/model_lab.png`: model comparison and evaluation charts

## 12. Future scope

* Train on the full BigBasket product list for wider brand and product-type coverage.
* Add text features from product names and descriptions (TF-IDF or embeddings).
* Tune hyper-parameters with grid or randomised search.
* Track prices over time to forecast discounts.
* Deploy on Streamlit Community Cloud.

## 13. Limitations

* The data is a static 1,249-product sample, not live BigBasket prices.
* MRP explains almost all of the prediction; labels only fine-tune the discount.
* Many brands and product types appear only once or twice, so the model learns little about them.
* About a third of products have no rating.
* The prediction range is an empirical estimate, not a statistical guarantee.

## Team

Edit `src/project_info.py` to add names, roll numbers and enrollment numbers. They appear on the
About Project page.
