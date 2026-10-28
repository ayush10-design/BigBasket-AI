"""Reusable HTML/Streamlit components so every page looks consistent."""
from __future__ import annotations

import streamlit as st

from src.config import ASSETS_DIR
from src.utils import clean_html, esc, inr, pct, rating

INSIGHT_ACCENTS = {
    "price": "var(--turmeric)", "discount": "var(--basil)", "rating": "var(--sky)",
    "brand": "var(--lavender)", "volume": "var(--sky)", "relation": "var(--lavender)",
    "anomaly": "var(--rose)",
}


def html(markup: str) -> None:
    st.markdown(clean_html(markup), unsafe_allow_html=True)


def inject_css() -> None:
    css_file = ASSETS_DIR / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    html(f'<div class="bb-header"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def kpi_grid(items: list[tuple[str, str, str | None]], lead_first: bool = False) -> None:
    """items: (label, value, optional note)."""
    cards = []
    for i, (label, value, note) in enumerate(items):
        cls = "bb-kpi lead" if lead_first and i == 0 else "bb-kpi"
        note_html = f'<div class="bb-kpi-note">{esc(note)}</div>' if note else ""
        cards.append(f'<div class="{cls}"><div class="bb-kpi-label">{esc(label)}</div>'
                     f'<div class="bb-kpi-value">{esc(value)}</div>{note_html}</div>')
    html('<div class="bb-kpis">' + "".join(cards) + "</div>")


def insight_cards(insights: list[dict]) -> None:
    cards = []
    for item in insights:
        accent = INSIGHT_ACCENTS.get(item.get("kind"), "var(--basil)")
        cards.append(
            f'<div class="bb-insight" style="--accent:{accent}">'
            f'<div class="bb-insight-title">{esc(item["title"])}</div>'
            f'<div class="bb-insight-value">{esc(item["value"])}</div>'
            f'<div class="bb-insight-detail">{esc(item["detail"])}</div></div>'
        )
    html('<div class="bb-insights">' + "".join(cards) + "</div>")


def empty_state(title: str, message: str) -> None:
    html(f'<div class="bb-empty"><h4>{esc(title)}</h4><p>{esc(message)}</p></div>')


def key_value_rows(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    html(f'<table class="bb-rows">{body}</table>')


def price_tag(result: dict, model_name: str, coverage: float) -> None:
    """The AI estimate, drawn as a supermarket shelf tag."""
    mrp = result["inputs"]["MRP"]
    if result["above_mrp"]:
        badge = '<span class="bb-tag-badge">Above MRP</span>'
    else:
        badge = f'<span class="bb-tag-badge">{pct(result["discount_pct"])} off</span>'
    html(f"""
    <div class="bb-tag">
      <div class="bb-tag-label">Estimated selling price</div>
      <div class="bb-tag-price">{inr(result['prediction'])}</div>
      <div class="bb-tag-mrp">MRP <s>{inr(mrp)}</s>{badge}</div>
      <div class="bb-tag-range">Likely range {inr(result['low'])} to {inr(result['high'])}</div>
      <div class="bb-tag-foot">Range from 10th and 90th percentile models, which covered {coverage * 100:.0f}% of test-set prices.<br>
      {esc(model_name)}. A model estimate, not a live BigBasket price.</div>
    </div>
    """)


def product_cards(rows) -> None:
    cards = []
    for _, r in rows.iterrows():
        disc = r["Discount"]
        chip = (f'<span class="bb-chip basil">{pct(disc, 0)} off</span>' if disc and disc > 0
                else '<span class="bb-chip muted">No discount</span>')
        rate = f'<span class="bb-chip turmeric">★ {rating(r["Rating"])}</span>'
        strike = f"<s>{inr(r['MRP'])}</s>" if r["MRP"] > r["Price"] else ""
        cards.append(
            f'<div class="bb-pcard"><div class="bb-pcard-name">{esc(r["Product"])}</div>'
            f'<div class="bb-pcard-meta">{esc(r["Brand"])}, {esc(r["Product_Type"])}</div>'
            f'<div class="bb-pcard-price">{inr(r["Price"])}{strike}</div>'
            f'<div>{chip}{rate}</div>'
            f'<div class="bb-pcard-meta">{esc(r["Category"])} › {esc(r["Subcategory"])}</div></div>'
        )
    html('<div class="bb-pcards">' + "".join(cards) + "</div>")


def sidebar_brand() -> None:
    with st.sidebar:
        html("""
        <div class="bb-brand"><div class="bb-brand-row">
          <span class="bb-brand-mark">b</span>
          <div><div class="bb-brand-name">BigBasket AI</div>
          <div class="bb-brand-sub">Product intelligence & price prediction</div></div>
        </div></div>
        """)


def sidebar_status(dataset_name: str | None, rows: int | None, r2: float | None) -> None:
    with st.sidebar:
        st.divider()
        if dataset_name:
            data_line = f'<span class="bb-dot"></span>Dataset <b>{esc(dataset_name)}</b><br><span class="bb-dot" style="visibility:hidden"></span>{rows:,} products'
        else:
            data_line = '<span class="bb-dot off"></span>No dataset loaded'
        model_line = (f'<span class="bb-dot"></span>Model ready, test R² <b>{r2:.3f}</b>' if r2 is not None
                      else '<span class="bb-dot off"></span>Model not trained')
        html(f'<div class="bb-status">{data_line}<br>{model_line}</div>')
