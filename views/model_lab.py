"""🧪 Model Lab: how the price model was built and how well it performs on unseen data."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import show
from src.config import (CATEGORICAL_FEATURES, CV_FOLDS, INTERVAL_HIGH_Q, INTERVAL_LOW_Q,
                        NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, COLORS)
from src.data_loader import data_quality_report
from src.model import MAIN_MODEL_NAME
from src.state import get_dataset, get_model, retrain_model
from src.ui import kpi_grid, key_value_rows, page_header
from src.utils import inr

df, info = get_dataset()
bundle = get_model(df, info)
m = bundle["metrics"]
comparison = bundle["comparison"]
results = bundle["test_results"]
meta = bundle["meta"]

page_header("Model Lab",
            f"How the {MAIN_MODEL_NAME} was trained, which features it uses and how accurately it predicts "
            "prices for products it has never seen.")

kpi_grid([
    ("R² score (test)", f"{m['R2']:.3f}", f"Train R² {m['Train R2']:.3f}"),
    ("MAE (test)", inr(m["MAE"]), "Average absolute error"),
    ("RMSE (test)", inr(m["RMSE"]), "Penalises large errors"),
    ("Training samples", f"{bundle['n_train']:,}", f"{100 - TEST_SIZE * 100:.0f}% of {bundle['n_total']:,} rows"),
    ("Testing samples", f"{bundle['n_test']:,}", f"{TEST_SIZE * 100:.0f}% held out"),
], lead_first=True)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown("### Model card")
    params = ", ".join(f"{k}={v}" for k, v in meta["gb_params"].items())
    key_value_rows([
        ("Model", MAIN_MODEL_NAME),
        ("Target", f"{TARGET} (selling price, ₹)"),
        ("Numeric feature", ", ".join(NUMERIC_FEATURES)),
        ("Categorical features", ", ".join(c.replace("_", " ") for c in CATEGORICAL_FEATURES)),
        ("Encoding", "One-hot, unknown values ignored"),
        ("Missing values", "Median (numeric), most frequent (categorical)"),
        ("Split", f"{100 - TEST_SIZE * 100:.0f}/{TEST_SIZE * 100:.0f}, random_state={RANDOM_STATE}"),
        ("Hyper-parameters", params),
        ("Trained", f"{meta['trained_at']} with scikit-learn {meta['sklearn_version']}"),
    ])
with c2:
    st.markdown("### Data used")
    key_value_rows([
        ("Dataset file", info.path.name),
        ("Rows in file", f"{info.raw_rows:,} ({info.raw_columns} columns)"),
        ("Rows after cleaning", f"{info.clean_rows:,}"),
        ("Exact duplicates removed", f"{info.duplicates_removed:,}"),
        ("Rows available for modelling", f"{bundle['n_total']:,}"),
        ("Excluded to prevent leakage", "Price_Segment and Discount (both derived from Price)"),
    ])
    with st.expander("Missing values per column"):
        st.dataframe(data_quality_report(df), width="stretch")

# ---------------------------------------------------------------------------
st.markdown("### Model comparison")
st.caption(f"All three models use the same preprocessing and the same train/test split. "
           f"CV columns are {CV_FOLDS}-fold cross-validation on all {bundle['n_total']:,} rows.")
st.dataframe(
    comparison, hide_index=True, width="stretch",
    column_config={
        "MAE": st.column_config.NumberColumn("Test MAE", format="₹%.2f"),
        "RMSE": st.column_config.NumberColumn("Test RMSE", format="₹%.2f"),
        "R2": st.column_config.NumberColumn("Test R²", format="%.3f"),
        "Train R2": st.column_config.NumberColumn("Train R²", format="%.3f"),
        "CV R2 mean": st.column_config.NumberColumn("CV R² (mean)", format="%.3f"),
        "CV R2 std": st.column_config.NumberColumn("CV R² (std)", format="%.3f"),
        "CV MAE mean": st.column_config.NumberColumn("CV MAE (mean)", format="₹%.2f"),
    },
)

c1, c2 = st.columns(2, gap="medium")
with c1:
    fig = px.bar(comparison.sort_values("R2"), x="R2", y="Model", orientation="h", text="R2",
                 title="Test R² by model (higher is better)")
    fig.update_traces(marker_color=COLORS["basil"], texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="R²", yaxis_title=None, xaxis_range=[min(0, comparison["R2"].min()), 1.05])
    show(fig, height=300)
with c2:
    long = comparison.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="₹")
    fig = px.bar(long, x="₹", y="Model", color="Metric", barmode="group", orientation="h",
                 title="Test errors by model (lower is better)",
                 color_discrete_map={"MAE": COLORS["turmeric"], "RMSE": COLORS["rose"]})
    fig.update_layout(xaxis_title="Error (₹)", yaxis_title=None, legend=dict(orientation="h", y=-0.3))
    show(fig, height=300)

best = bundle["best_by_rmse"]
row = comparison.set_index("Model")
gb, top = row.loc[MAIN_MODEL_NAME], row.loc[best]
if best == MAIN_MODEL_NAME:
    st.success(f"**{MAIN_MODEL_NAME}** has the lowest test RMSE ({inr(gb['RMSE'])}) and is the model used by the "
               "AI Price Predictor.")
else:
    st.info(
        f"On this split **{best}** has the lowest test RMSE ({inr(top['RMSE'])} vs {inr(gb['RMSE'])} for Gradient "
        f"Boosting). Cross-validated R² is {top['CV R2 mean']:.3f} ± {top['CV R2 std']:.3f} for {best} and "
        f"{gb['CV R2 mean']:.3f} ± {gb['CV R2 std']:.3f} for Gradient Boosting. The app keeps {MAIN_MODEL_NAME} "
        "as the prediction model to stay consistent with the SI notebook; the comparison is reported as measured."
    )
st.caption("RMSE is the selection metric because it is in rupees and penalises large pricing mistakes. "
           "On a single test set, ranking by RMSE and by R² always agree.")

# ---------------------------------------------------------------------------
st.markdown("### Prediction quality on held-out products")
log_axes = st.toggle("Logarithmic axes", value=True, key="lab_log")
c1, c2 = st.columns(2, gap="medium")
with c1:
    fig = px.scatter(results, x="Actual", y="Predicted", hover_data=["Category", "Brand", "MRP"],
                     title="Actual vs predicted selling price", log_x=log_axes, log_y=log_axes, opacity=0.8)
    fig.update_traces(marker=dict(color=COLORS["sky"], size=7))
    lim = [max(1, min(results["Actual"].min(), results["Predicted"].min())),
           max(results["Actual"].max(), results["Predicted"].max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                             line=dict(color=COLORS["turmeric"], dash="dash"), hoverinfo="skip"))
    fig.update_layout(xaxis_title="Actual price (₹)", yaxis_title="Predicted price (₹)",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, height=420)
with c2:
    fig = px.histogram(results, x="Residual", nbins=50, title="Residual distribution (actual − predicted)")
    fig.update_traces(marker_color=COLORS["lavender"], marker_line_width=0)
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["turmeric"])
    fig.update_layout(xaxis_title="Residual (₹)", yaxis_title="Number of test products")
    show(fig, height=420)

fig = px.scatter(results, x="Predicted", y="Residual", hover_data=["Category", "Brand", "Actual"],
                 title="Residuals vs predicted price", log_x=log_axes, opacity=0.8)
fig.update_traces(marker=dict(color=COLORS["basil"], size=7))
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["turmeric"])
fig.update_layout(xaxis_title="Predicted price (₹)", yaxis_title="Residual (₹)")
show(fig, height=360)
within10 = (results["Absolute error"] <= results["Actual"] * 0.10).mean() * 100
st.caption(f"Median absolute error {inr(results['Absolute error'].median())}. "
           f"{within10:.0f}% of test predictions are within 10% of the real price. "
           "Errors grow with price: a few expensive items account for most of the RMSE.")

# ---------------------------------------------------------------------------
st.markdown("### What the model relies on")
c1, c2 = st.columns(2, gap="medium")
with c1:
    imp = bundle["feature_importance"].sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", text="Importance",
                 title="Feature importance (one-hot columns summed per feature)")
    fig.update_traces(marker_color=COLORS["turmeric"], texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Importance (sums to 1)", yaxis_title=None, xaxis_range=[0, 1.12])
    show(fig, height=330)
with c2:
    top_imp = bundle["feature_importance_top"].sort_values("Importance")
    fig = px.bar(top_imp, x="Importance", y="Encoded feature", orientation="h",
                 title="Top 15 individual encoded columns", log_x=True)
    fig.update_traces(marker_color=COLORS["sky"])
    fig.update_layout(xaxis_title="Importance (log scale)", yaxis_title=None)
    show(fig, height=330)
mrp_share = float(bundle["feature_importance"].set_index("Feature").loc["MRP", "Importance"])
st.caption(f"MRP accounts for {mrp_share * 100:.1f}% of the total importance: the selling price is mostly MRP "
           "minus a discount, and the labels fine-tune that discount.")

# ---------------------------------------------------------------------------
st.markdown("### Prediction range")
width = (results["Range high"] - results["Range low"]).median()
st.markdown(
    f"The Predictor's range comes from two extra Gradient Boosting models trained with quantile loss at the "
    f"{INTERVAL_LOW_Q * 100:.0f}th and {INTERVAL_HIGH_Q * 100:.0f}th percentiles. Ideally about "
    f"{(INTERVAL_HIGH_Q - INTERVAL_LOW_Q) * 100:.0f}% of real prices fall inside it. On the test set, "
    f"**{bundle['interval_coverage'] * 100:.1f}%** did, with a median width of {inr(width)}. "
    "This is an empirical estimate, not a statistical guarantee."
)

# ---------------------------------------------------------------------------
st.markdown("### Export and retrain")
c1, c2, c3 = st.columns(3)
c1.download_button("Download test predictions", results.to_csv(index=False).encode("utf-8"),
                   file_name="test_predictions.csv", mime="text/csv", width="stretch")
c2.download_button("Download model comparison", comparison.to_csv(index=False).encode("utf-8"),
                   file_name="model_comparison.csv", mime="text/csv", width="stretch")
if c3.button("Retrain all models", width="stretch",
             help="Deletes the saved model and trains again from the current dataset."):
    retrain_model()
    st.rerun()
st.caption(f"Last training run took {meta['training_seconds']} s. The model is saved in models/ and reused "
           "until the dataset changes.")

