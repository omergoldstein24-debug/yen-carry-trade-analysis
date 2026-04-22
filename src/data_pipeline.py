"""
data_pipeline.py
----------------
Builds the unified monthly dataset `usd_jpy_interest_rates_clean.csv`
from three raw sources:

    1. Japan 10Y government bond yield          (FRED series IRLTLT01JPM156N)
    2. US    10Y Treasury constant-maturity     (FRED series GS10)
    3. USD/JPY monthly spot close                (Investing.com export)

All three files cover Jan-2015 to Dec-2025 at monthly frequency.

The script performs the following steps:
    - loads each source into a typed DataFrame
    - normalises column names and date formats
    - merges on the month-start timestamp
    - computes the US-minus-Japan interest-rate differential
    - writes the tidy output to data/processed/

Usage
-----
    python -m src.data_pipeline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

JAPAN_10Y_FILE = RAW_DIR / "japan_10y_yield.csv"
US_10Y_FILE = RAW_DIR / "us_10y_yield.csv"
USD_JPY_FILE = RAW_DIR / "usd_jpy_historical.csv"
OUTPUT_FILE = PROCESSED_DIR / "usd_jpy_interest_rates_clean.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_japan_10y(path: Path = JAPAN_10Y_FILE) -> pd.DataFrame:
    """Load Japan 10-year government bond yield (monthly, %)."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"IRLTLT01JPM156N": "japan_10y_yield"})
    return df.sort_values("observation_date").reset_index(drop=True)


def load_us_10y(path: Path = US_10Y_FILE) -> pd.DataFrame:
    """Load US 10-year Treasury constant-maturity yield (monthly, %)."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"GS10": "us_10y_yield"})
    return df.sort_values("observation_date").reset_index(drop=True)


def load_usd_jpy(path: Path = USD_JPY_FILE) -> pd.DataFrame:
    """Load USD/JPY monthly close (end-of-month, nominal)."""
    df = pd.read_csv(path)

    # The Investing.com export uses MM/DD/YYYY and is sorted newest-first.
    df["observation_date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns={"Price": "usd_jpy"})
    df = df[["observation_date", "usd_jpy"]]

    # The provider dates fall on the 1st of the month but represent that
    # month's close — align to the month-start to match the FRED convention.
    df["observation_date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()

    return df.sort_values("observation_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def build_clean_dataset() -> pd.DataFrame:
    """Merge all three sources into one tidy monthly DataFrame."""
    japan = load_japan_10y()
    us = load_us_10y()
    fx = load_usd_jpy()

    merged = (
        japan
        .merge(us, on="observation_date", how="inner", validate="1:1")
        .merge(fx, on="observation_date", how="inner", validate="1:1")
    )

    # Derived column: US minus Japan 10Y yield differential (percentage points)
    merged["rate_differential"] = merged["us_10y_yield"] - merged["japan_10y_yield"]

    # Final presentation-ready layout
    merged["Year"] = merged["observation_date"].dt.year
    merged["Month"] = merged["observation_date"].dt.strftime("%B")

    cols = [
        "observation_date",
        "Year",
        "Month",
        "japan_10y_yield",
        "us_10y_yield",
        "rate_differential",
        "usd_jpy",
    ]
    return merged[cols].sort_values("observation_date").reset_index(drop=True)


def save_clean_dataset(df: pd.DataFrame, path: Path = OUTPUT_FILE) -> Path:
    """Persist the merged dataset as a clean CSV (ISO dates, dot-decimals)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.3f")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = build_clean_dataset()
    out_path = save_clean_dataset(df)
    print(f"[data_pipeline] {len(df)} rows written to {out_path.relative_to(PROJECT_ROOT)}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
