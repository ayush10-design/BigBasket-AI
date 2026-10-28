"""Cached access to the dataset and the trained model.

@st.cache_data keeps the cleaned DataFrame in memory (re-read only if the
CSV changes). @st.cache_resource keeps the trained model in memory, and the
model is also saved to models/ so later runs skip training entirely.
"""
from __future__ import annotations

import streamlit as st

from src.config import MODEL_FILE
from src.data_loader import DatasetInfo, find_dataset_file, load_dataset
from src.model import get_or_train


@st.cache_data(show_spinner="Loading the dataset…")
def _load(path_str: str, modified: float):
    df, info = load_dataset(path_str)
    return df, info


def get_dataset():
    """Return (df, info). Raises DatasetNotFoundError / DatasetSchemaError."""
    path = find_dataset_file()
    return _load(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner="Loading the price model. The first run trains it, which takes about a minute…")
def _bundle(fingerprint: str, _df):
    bundle, _ = get_or_train(_df, fingerprint)
    return bundle


def get_model(df, info: DatasetInfo) -> dict:
    return _bundle(info.fingerprint, df)


def retrain_model() -> None:
    """Delete the saved model and clear the in-memory cache."""
    _bundle.clear()
    try:
        MODEL_FILE.unlink(missing_ok=True)
    except OSError:
        pass
