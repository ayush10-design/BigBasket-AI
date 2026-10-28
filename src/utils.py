"""Small formatting helpers shared by every page."""
from __future__ import annotations

import html
import math


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def inr(value, decimals: int = 2) -> str:
    """₹1,234.50 style currency; an em dash for missing values."""
    if is_missing(value):
        return "—"
    return f"₹{value:,.{decimals}f}"


def pct(value, decimals: int = 1) -> str:
    if is_missing(value):
        return "—"
    return f"{value:.{decimals}f}%"


def num(value, decimals: int = 0) -> str:
    if is_missing(value):
        return "—"
    return f"{value:,.{decimals}f}"


def rating(value) -> str:
    if is_missing(value):
        return "Not rated"
    return f"{value:.1f} / 5"


def esc(value) -> str:
    """Escape text from the dataset before placing it inside HTML."""
    return html.escape("" if is_missing(value) else str(value))


def clean_html(markup: str) -> str:
    """Remove indentation so Streamlit's Markdown does not turn HTML into code blocks."""
    return "".join(line.strip() for line in markup.splitlines())


def short_labels(frame, max_len: int = 34):
    """Add a readable, unique 'Label' column for product names on chart axes."""
    labels = frame["Product"].astype(str).map(lambda p: p if len(p) <= max_len else p[: max_len - 2] + "…")
    dupes = labels.duplicated(keep=False)
    labels[dupes] = labels[dupes] + " (" + frame.loc[dupes, "Brand"].astype(str) + ")"
    labels = labels.where(~labels.duplicated(keep=False), labels + " #" + frame["Item_ID"].astype(str))
    return frame.assign(Label=labels)
