"""
analysis.py
-----------
Statistical and financial calculations on top of the clean monthly dataset.

Exposed helpers:
    - load_clean_data      : read the processed CSV as a tidy DataFrame
    - monthly_fx_returns   : %-change series for USD/JPY
    - regression_stats     : Pearson r and OLS slope for differential -> FX
    - bucketed_weakening   : share of months where JPY weakened, by diff bucket
    - carry_trade_simulation : $100 borrow-JPY / lend-USD monthly simulation

All computations assume month-end observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "usd_jpy_interest_rates_clean.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_clean_data(path: Path = CLEAN_FILE) -> pd.DataFrame:
    """Load the processed monthly dataset with proper dtypes."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    return df.sort_values("observation_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Monthly FX returns
# ---------------------------------------------------------------------------
def monthly_fx_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the month-over-month %-change in USD/JPY."""
    out = df.copy()
    out["usd_jpy_pct_change"] = out["usd_jpy"].pct_change() * 100.0
    return out


# ---------------------------------------------------------------------------
# Regression: rate differential → FX move
# ---------------------------------------------------------------------------
@dataclass
class RegressionResult:
    correlation: float
    slope: float
    intercept: float

    @property
    def r_squared(self) -> float:
        return self.correlation ** 2


def regression_stats(df: pd.DataFrame) -> RegressionResult:
    """Simple OLS of USD/JPY monthly %-change on the rate differential."""
    data = monthly_fx_returns(df).dropna(subset=["usd_jpy_pct_change"])
    x = data["rate_differential"].to_numpy()
    y = data["usd_jpy_pct_change"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    return RegressionResult(correlation=correlation, slope=float(slope), intercept=float(intercept))


# ---------------------------------------------------------------------------
# Bucketed analysis: does a bigger differential mean more yen weakness?
# ---------------------------------------------------------------------------
def bucketed_weakening(
    df: pd.DataFrame,
    edges: tuple[float, ...] = (-np.inf, 1.5, 2.5, 3.5, np.inf),
    labels: tuple[str, ...] = (
        "Low\n(<1.5%)",
        "Medium\n(1.5-2.5%)",
        "High\n(2.5-3.5%)",
        "Very High\n(>3.5%)",
    ),
) -> pd.DataFrame:
    """Share of months in which USD/JPY rose (JPY weakened), by differential bucket."""
    data = monthly_fx_returns(df).dropna(subset=["usd_jpy_pct_change"]).copy()
    data["bucket"] = pd.cut(data["rate_differential"], bins=edges, labels=labels, right=False)

    grouped = (
        data.groupby("bucket", observed=True)["usd_jpy_pct_change"]
        .agg(
            months="count",
            pct_jpy_weakened=lambda s: (s > 0).mean() * 100.0,
            avg_move="mean",
        )
        .reset_index()
    )
    return grouped


# ---------------------------------------------------------------------------
# Carry-trade simulation
# ---------------------------------------------------------------------------
def carry_trade_simulation(df: pd.DataFrame, initial_capital: float = 100.0) -> pd.DataFrame:
    """
    Simulate a simple long-USD / short-JPY carry trade funded at Japan 10Y
    and invested at US 10Y, marked-to-market each month through the FX rate.

    Monthly total return =
        (1 + (us_10y - japan_10y) / 12 / 100)
        * (usd_jpy_t / usd_jpy_{t-1})           # FX P&L (higher USD/JPY = JPY weaker = gain)

    A "pure carry" benchmark series is also produced — the interest-rate
    differential alone, with no FX impact.
    """
    data = df.sort_values("observation_date").reset_index(drop=True).copy()

    carry_monthly = data["rate_differential"] / 12.0 / 100.0
    fx_monthly = data["usd_jpy"].pct_change().fillna(0.0)

    total_return = (1.0 + carry_monthly) * (1.0 + fx_monthly) - 1.0

    # The first row has no realised FX move yet — anchor to initial capital
    # with no month-zero return so the cumulative path starts at exactly $100.
    total_return.iloc[0] = 0.0

    data["carry_trade_value"] = initial_capital * (1.0 + total_return).cumprod()
    data["rate_only_value"] = initial_capital * (1.0 + carry_monthly).cumprod()
    data.loc[0, "rate_only_value"] = initial_capital

    return data[
        ["observation_date", "usd_jpy", "rate_differential",
         "carry_trade_value", "rate_only_value"]
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_clean_data()

    reg = regression_stats(df)
    print(f"[analysis] Pearson r = {reg.correlation:+.2f}, "
          f"slope = {reg.slope:+.2f}%/pp, R² = {reg.r_squared:.3f}")

    buckets = bucketed_weakening(df)
    print("[analysis] JPY weakening frequency by differential bucket:")
    print(buckets.to_string(index=False))

    sim = carry_trade_simulation(df)
    print(f"[analysis] Carry trade: start $100 -> "
          f"peak ${sim['carry_trade_value'].max():.0f}, "
          f"trough ${sim['carry_trade_value'].min():.0f}, "
          f"end ${sim['carry_trade_value'].iloc[-1]:.0f}")


if __name__ == "__main__":
    main()
