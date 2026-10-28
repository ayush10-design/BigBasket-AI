"""Locate, load and clean the BigBasket dataset.

The cleaning steps reproduce section 3 of the SI notebook:
rename columns, convert numerics, trim text, drop rows without a product name,
drop exact duplicates, fill missing labels with 'Unknown', invalidate negative
prices and recalculate Discount (%) from MRP and sale price.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PREFERRED_DATA_FILES

# Raw column names used by the notebook's source file -> clean names
NOTEBOOK_COLUMN_MAP = {
    "index_items": "Item_ID",
    "product_name": "Product",
    "category_name": "Category",
    "sub_category": "Subcategory",
    "brand_name": "Brand",
    "sale_price": "Price",
    "market_price": "MRP",
    "type_product": "Product_Type",
    "rating_product": "Rating",
    "price_segment": "Price_Segment",
}
# Column names of the original public Kaggle file ("BigBasket Products.csv")
KAGGLE_COLUMN_MAP = {
    "index": "Item_ID",
    "product": "Product",
    "category": "Category",
    "brand": "Brand",
    "type": "Product_Type",
    "rating": "Rating",
}

REQUIRED_COLUMNS = ["Product", "Category", "Subcategory", "Brand", "Price", "MRP", "Product_Type"]
TEXT_COLUMNS = ["Product", "Category", "Subcategory", "Brand", "Product_Type", "Price_Segment"]


class DatasetNotFoundError(FileNotFoundError):
    """No CSV file was found in the data/ folder."""


class DatasetSchemaError(ValueError):
    """The CSV does not contain the columns the project needs."""


@dataclass
class DatasetInfo:
    path: Path
    raw_rows: int
    raw_columns: int
    clean_rows: int
    duplicates_removed: int
    fingerprint: str


def find_dataset_file(data_dir: Path = DATA_DIR) -> Path:
    """Return the dataset path, preferring the notebook's file names."""
    if not data_dir.exists():
        raise DatasetNotFoundError(f"The folder '{data_dir.name}/' does not exist.")
    for name in PREFERRED_DATA_FILES:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    raise DatasetNotFoundError(
        f"No CSV file found in '{data_dir.name}/'. Copy bigbasket_cleaned.csv there."
    )


def file_fingerprint(path: Path) -> str:
    """Short content hash, used to know when the saved model is out of date."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:16]


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map any supported column naming scheme to the project's clean names."""
    out = df.rename(columns=lambda c: str(c).strip())
    out = out.rename(columns=NOTEBOOK_COLUMN_MAP)
    # Kaggle names overlap with cleaned names only by case, so map them separately
    kaggle_present = {k: v for k, v in KAGGLE_COLUMN_MAP.items() if k in out.columns and v not in out.columns}
    out = out.rename(columns=kaggle_present)
    # Remove unnamed index columns created by `to_csv(index=True)`
    out = out.loc[:, [c for c in out.columns if not str(c).startswith("Unnamed")]]
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise DatasetSchemaError(
            "The dataset is missing required column(s): " + ", ".join(missing)
        )
    return out


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the notebook's cleaning steps. Returns (clean_df, duplicates_removed)."""
    df = standardise_columns(df).copy()

    if "Item_ID" not in df.columns:
        df.insert(0, "Item_ID", np.arange(1, len(df) + 1))
    if "Rating" not in df.columns:
        df["Rating"] = np.nan

    for col in ["Item_ID", "Price", "MRP", "Rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=["Product"])
    df = df[df["Product"] != ""]
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    for col in ["Category", "Subcategory", "Brand", "Product_Type"]:
        df[col] = df[col].fillna("Unknown").replace("", "Unknown")
    if "Price_Segment" in df.columns:
        df["Price_Segment"] = df["Price_Segment"].fillna("Unknown").replace("", "Unknown")

    for col in ["Price", "MRP"]:
        df.loc[df[col] < 0, col] = np.nan

    # Ratings must be on the 0-5 scale; anything else is treated as missing
    df.loc[~df["Rating"].between(0, 5), "Rating"] = np.nan

    valid = df["MRP"].gt(0) & df["Price"].notna()
    df["Discount"] = np.nan
    df.loc[valid, "Discount"] = (
        (df.loc[valid, "MRP"] - df.loc[valid, "Price"]) / df.loc[valid, "MRP"] * 100
    ).clip(0, 100)

    # Plain Python strings make filtering and Plotly behave predictably
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object)

    keep = ["Item_ID", "Product", "Category", "Subcategory", "Brand", "Price", "MRP",
            "Product_Type", "Rating", "Discount"]
    if "Price_Segment" in df.columns:
        keep.insert(-1, "Price_Segment")
    return df[keep].reset_index(drop=True), duplicates_removed


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, DatasetInfo]:
    """Find, read and clean the dataset."""
    path = Path(path) if path else find_dataset_file()
    try:
        raw = pd.read_csv(path)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="latin-1")
    if raw.empty:
        raise DatasetSchemaError(f"'{path.name}' contains no rows.")
    clean, dups = clean_dataset(raw)
    if clean.empty:
        raise DatasetSchemaError(f"'{path.name}' has no usable rows after cleaning.")
    info = DatasetInfo(
        path=path,
        raw_rows=len(raw),
        raw_columns=raw.shape[1],
        clean_rows=len(clean),
        duplicates_removed=dups,
        fingerprint=file_fingerprint(path),
    )
    return clean, info


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value table used by the Model Lab and About pages."""
    report = df.isna().sum().to_frame("Missing values")
    report["Missing (%)"] = (report["Missing values"] / max(len(df), 1) * 100).round(2)
    report["Data type"] = df.dtypes.astype(str)
    return report
