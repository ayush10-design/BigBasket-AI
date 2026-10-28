"""Plotly theme and a few chart builders reused on several pages."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.config import CHART_SEQUENCE, COLORS

TEMPLATE_NAME = "bigbasket_ai"

pio.templates[TEMPLATE_NAME] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="DM Sans, Segoe UI, system-ui, sans-serif", color=COLORS["text"], size=13),
        title=dict(font=dict(family="Outfit, Segoe UI, sans-serif", size=16, color=COLORS["text"]), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_SEQUENCE,
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"],
                   title_font=dict(color=COLORS["muted"]), tickfont=dict(color=COLORS["muted"])),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"],
                   title_font=dict(color=COLORS["muted"]), tickfont=dict(color=COLORS["muted"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
        hoverlabel=dict(bgcolor=COLORS["panel_2"], bordercolor=COLORS["grid"],
                        font=dict(color=COLORS["text"], family="DM Sans, sans-serif")),
        margin=dict(l=10, r=16, t=56, b=10),
        bargap=0.18,
    )
)
pio.templates.default = TEMPLATE_NAME

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show(fig: go.Figure, height: int = 380, key: str | None = None) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key)


def bar_by_group(summary: pd.DataFrame, group: str, value: str, title: str, x_label: str,
                 color: str = COLORS["basil"], text_format: str = ",.0f", top: int | None = None,
                 hover_extra: list[str] | None = None) -> go.Figure:
    """Horizontal bar chart, largest at the top."""
    data = summary.sort_values(value, ascending=False)
    if top:
        data = data.head(top)
    data = data.iloc[::-1]
    fig = px.bar(data, x=value, y=group, orientation="h", title=title,
                 hover_data=hover_extra or ["Products"], text=value)
    fig.update_traces(marker_color=color, texttemplate="%{text:" + text_format + "}",
                      textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=x_label, yaxis_title=None,
                      xaxis_range=[0, max(float(data[value].max()), 0) * 1.18 or 1])
    return fig


def histogram(series: pd.Series, title: str, x_label: str, color: str = COLORS["basil"],
              bins: int = 40, clip_quantile: float | None = None) -> tuple[go.Figure, int]:
    """Histogram; optionally hides the extreme tail. Returns (figure, hidden_count)."""
    values = series.dropna()
    hidden = 0
    if clip_quantile and len(values) > 20:
        limit = values.quantile(clip_quantile)
        hidden = int((values > limit).sum())
        values = values[values <= limit]
    fig = px.histogram(values, nbins=bins, title=title)
    fig.update_traces(marker_color=color, marker_line_width=0, hovertemplate="%{x}<br>%{y} products<extra></extra>")
    fig.update_layout(xaxis_title=x_label, yaxis_title="Number of products", showlegend=False)
    return fig, hidden
