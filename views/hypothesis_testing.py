"""Statistical hypothesis testing for BigBasket AI."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from src.state import get_dataset
from src.ui import page_header

df, info = get_dataset()

page_header(
    "Hypothesis Testing",
    "Use statistical tests to investigate price differences, category associations and numerical relationships."
)

st.markdown("### How hypothesis testing works")
st.markdown(
    "- **Null hypothesis (H₀):** No significant difference or association exists.\n"
    "- **Alternative hypothesis (H₁):** A significant difference or association exists.\n"
    "- **Significance level (α):** 0.05 by default.\n"
    "- If p-value < α, reject H₀. Otherwise, there is insufficient evidence to reject H₀."
)

alpha = st.select_slider(
    "Significance level (α)",
    options=[0.01, 0.05, 0.10],
    value=0.05,
    format_func=lambda x: f"{x:.2f}"
)

st.caption(
    f"Dataset: {info.path.name} · {len(df):,} cleaned rows. "
    "Results describe this dataset and do not establish causation."
)

def show_result(test_name, h0, h1, statistic, p_value):
    st.markdown(f"#### {test_name}")
    st.write(f"**H₀:** {h0}")
    st.write(f"**H₁:** {h1}")
    c1, c2 = st.columns(2)
    c1.metric("Test statistic", f"{statistic:.4f}")
    c2.metric("P-value", f"{p_value:.6g}")
    if p_value < alpha:
        st.success(
            f"Reject H₀ at α = {alpha:.2f}. The result is statistically significant."
        )
    else:
        st.info(
            f"Fail to reject H₀ at α = {alpha:.2f}. "
            "There is insufficient evidence of a statistically significant result."
        )

tab1, tab2, tab3 = st.tabs([
    "One-Way ANOVA",
    "Chi-Square Test",
    "Spearman Correlation"
])

with tab1:
    st.markdown("### Compare average prices across categories")
    st.write(
        "ANOVA tests whether the mean selling price differs across product categories."
    )
    if "Category" not in df.columns or "Price" not in df.columns:
        st.warning("This test requires Category and Price columns.")
    else:
        min_group_size = st.slider(
            "Minimum products per category",
            min_value=2,
            max_value=20,
            value=5
        )
        work = df[["Category", "Price"]].copy()
        work["Price"] = pd.to_numeric(work["Price"], errors="coerce")
        work = work.dropna()
        counts = work["Category"].value_counts()
        eligible = counts[counts >= min_group_size].index.tolist()
        groups = [
            work.loc[work["Category"] == category, "Price"].to_numpy()
            for category in eligible
        ]
        if len(groups) < 2:
            st.warning(
                "At least two categories must meet the minimum group size. "
                "Lower the threshold or check the dataset."
            )
        else:
            summary = (
                work[work["Category"].isin(eligible)]
                .groupby("Category", as_index=False)
                .agg(Mean_Price=("Price", "mean"), Product_Count=("Price", "size"))
                .sort_values("Mean_Price", ascending=False)
            )
            st.plotly_chart(
                px.bar(
                    summary,
                    x="Category",
                    y="Mean_Price",
                    hover_data=["Product_Count"],
                    title="Mean selling price by category",
                    labels={"Mean_Price": "Mean price (₹)", "Category": "Category"}
                ),
                use_container_width=True
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)
            if st.button("Run One-Way ANOVA", key="run_anova"):
                statistic, p_value = stats.f_oneway(*groups)
                show_result(
                    "One-Way ANOVA",
                    "All included categories have equal mean selling prices.",
                    "At least one included category has a different mean selling price.",
                    statistic,
                    p_value
                )
                st.caption(
                    "ANOVA identifies whether a difference exists, not which specific "
                    "categories differ. Consider post-hoc testing for that question."
                )

with tab2:
    st.markdown("### Test association between categorical variables")
    st.write(
        "The Chi-Square test of independence checks whether two categorical variables "
        "are associated."
    )
    categorical = [
        col for col in ["Category", "Price_Segment", "Product_Type", "Brand"]
        if col in df.columns and df[col].nunique(dropna=True) > 1
    ]
    if len(categorical) < 2:
        st.warning("At least two suitable categorical columns are required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            row_col = st.selectbox("First variable", categorical, index=0)
        available_second = [col for col in categorical if col != row_col]
        with c2:
            col_col = st.selectbox(
                "Second variable", available_second, index=0
            )
        min_expected = st.checkbox(
            "Show expected frequencies", value=False
        )
        if st.button("Run Chi-Square Test", key="run_chi"):
            work = df[[row_col, col_col]].dropna()
            contingency = pd.crosstab(work[row_col], work[col_col])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                st.warning(
                    "Each selected variable must have at least two observed categories."
                )
            else:
                statistic, p_value, dof, expected = stats.chi2_contingency(
                    contingency
                )
                st.markdown("**Observed frequency table**")
                st.dataframe(contingency, use_container_width=True)
                if min_expected:
                    expected_df = pd.DataFrame(
                        expected,
                        index=contingency.index,
                        columns=contingency.columns
                    )
                    st.markdown("**Expected frequencies under H₀**")
                    st.dataframe(expected_df.round(2), use_container_width=True)
                show_result(
                    "Chi-Square Test of Independence",
                    f"{row_col} and {col_col} are independent.",
                    f"{row_col} and {col_col} are associated.",
                    statistic,
                    p_value
                )
                st.write(f"**Degrees of freedom:** {dof}")
                if (expected < 5).any():
                    st.warning(
                        "Some expected cell frequencies are below 5. "
                        "The Chi-Square approximation may be unreliable; "
                        "consider combining justified categories or another method."
                    )

with tab3:
    st.markdown("### Measure a monotonic relationship")
    st.write(
        "Spearman correlation measures the strength and direction of a monotonic "
        "relationship between two numerical variables. It does not prove causation."
    )
    numerical = [
        col for col in ["Price", "MRP", "Discount", "Rating"]
        if col in df.columns
    ]
    if len(numerical) < 2:
        st.warning("At least two numerical columns are required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("First numerical variable", numerical, index=0)
        with c2:
            y_col = st.selectbox(
                "Second numerical variable",
                [col for col in numerical if col != x_col],
                index=0
            )
        if st.button("Run Spearman Correlation", key="run_spearman"):
            work = df[[x_col, y_col]].copy()
            for col in [x_col, y_col]:
                work[col] = pd.to_numeric(work[col], errors="coerce")
            work = work.replace([np.inf, -np.inf], np.nan).dropna()
            if len(work) < 3 or work[x_col].nunique() < 2 or work[y_col].nunique() < 2:
                st.warning(
                    "At least three complete rows and variation in both variables are required."
                )
            else:
                result = stats.spearmanr(work[x_col], work[y_col])
                show_result(
                    "Spearman Rank Correlation",
                    f"There is no monotonic association between {x_col} and {y_col} "
                    "(population Spearman correlation is zero).",
                    f"There is a monotonic association between {x_col} and {y_col}.",
                    result.statistic,
                    result.pvalue
                )
                st.metric("Spearman correlation (ρ)", f"{result.statistic:.4f}")
                st.caption(
                    "ρ near +1 indicates a strong positive relationship; "
                    "ρ near −1 indicates a strong negative relationship; "
                    "ρ near 0 indicates little or no monotonic relationship."
                )
                st.plotly_chart(
                    px.scatter(
                        work,
                        x=x_col,
                        y=y_col,
                        opacity=0.55,
                        title=f"{x_col} vs {y_col}"
                    ),
                    use_container_width=True
                )

st.divider()
st.caption(
    "Academic note: statistical significance is not the same as practical importance. "
    "Check assumptions, data quality, sample size and effect sizes before drawing conclusions."
)
