"""
visualization.py
----------------
Reproduces the five figures that accompany the yen carry-trade write-up:

    1. japan_10y_yield.png             — annual Japan 10Y yield line chart
    2. rate_differential_scatter.png   — monthly FX move vs. rate differential
    3. jpy_weakening_by_bucket.png     — JPY-weakening frequency by diff bucket
    4. carry_trade_cumulative.png      — $100 carry-trade cumulative return
    5. carry_trade_diagram.png         — illustrative borrow-JPY / lend-USD card

Every function takes a clean DataFrame (from src.data_pipeline) and returns
the output Path. Styling is deliberately close to the versions used in the
final report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import analysis as ana

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 — Japan 10Y government-bond yield (annual averages)
# ---------------------------------------------------------------------------
def plot_japan_10y_yield(df: pd.DataFrame) -> Path:
    """Annual mean of the Japan 10Y yield, 2015–2025."""
    _ensure_figures_dir()

    # Use the December observation (year-end) to summarise each year
    year_end = (
        df.assign(year=df["observation_date"].dt.year,
                  month=df["observation_date"].dt.month)
          .loc[lambda d: d["month"] == 12, ["year", "japan_10y_yield"]]
          .reset_index(drop=True)
    )
    annual = year_end.round(2)

    teal = "#2f4f4a"
    salmon = "#b97a6a"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.plot(annual["year"], annual["japan_10y_yield"],
            color=salmon, linewidth=2.5, marker="o", markersize=7, zorder=3)

    for _, row in annual.iterrows():
        ax.annotate(f"{row['japan_10y_yield']:.2f}".replace(".", ","),
                    (row["year"], row["japan_10y_yield"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=teal, fontweight="bold")

    ax.set_title("Japan 10-Year Government Bond Yield (%)",
                 color=teal, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(annual["year"])
    ax.tick_params(axis="x", colors=teal, labelsize=11)
    ax.tick_params(axis="y", colors=teal, labelsize=11)

    # y-axis auto-extends to fit the peak with a small headroom
    y_max = float(annual["japan_10y_yield"].max()) + 0.2
    ax.set_ylim(-0.2, max(1.8, round(y_max * 5) / 5))
    ax.set_yticks(np.arange(-0.2, ax.get_ylim()[1] + 0.01, 0.2))
    ax.set_yticklabels([f"{v:.2f}".replace(".", ",") for v in ax.get_yticks()])

    ax.yaxis.grid(True, linestyle="-", color="#cfd8d6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(teal)

    fig.tight_layout()
    out = FIGURES_DIR / "01_japan_10y_yield.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Rate differential vs. monthly USD/JPY change (scatter + OLS)
# ---------------------------------------------------------------------------
def plot_rate_differential_scatter(df: pd.DataFrame) -> Path:
    _ensure_figures_dir()
    data = ana.monthly_fx_returns(df).dropna(subset=["usd_jpy_pct_change"])

    reg = ana.regression_stats(df)
    x_line = np.linspace(data["rate_differential"].min(), data["rate_differential"].max(), 100)
    y_line = reg.slope * x_line + reg.intercept

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    ax.scatter(data["rate_differential"], data["usd_jpy_pct_change"],
               color="#6ea8d6", s=50, alpha=0.75, edgecolor="#3a6fa0", linewidth=0.4)
    ax.plot(x_line, y_line, color="#e76f51", linewidth=2.2,
            label=f"Trend line (slope={reg.slope:.2f})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.text(0.02, 0.95, f"r = {reg.correlation:.2f}",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                      edgecolor="#d4a017", linewidth=1))

    ax.set_title("Interest Rate Differential (US − Japan) vs. Monthly % Change in USD/JPY",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Interest Rate Differential: US Rate − Japan Rate [%]", fontsize=11)
    ax.set_ylabel("Monthly % Change in USD/JPY", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out = FIGURES_DIR / "02_rate_differential_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — JPY-weakening frequency by differential bucket
# ---------------------------------------------------------------------------
def plot_jpy_weakening_by_bucket(df: pd.DataFrame) -> Path:
    _ensure_figures_dir()
    buckets = ana.bucketed_weakening(df)

    colors = ["#3fa9f5", "#f7a93c", "#ef5a5a", "#8a7dfb"]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120, facecolor="black")
    ax.set_facecolor("black")
    bars = ax.bar(buckets["bucket"].astype(str), buckets["pct_jpy_weakened"],
                  color=colors, edgecolor="white", linewidth=0.6, width=0.6)

    for bar, value in zip(bars, buckets["pct_jpy_weakened"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                f"{value:.0f}%", ha="center", va="bottom",
                color="white", fontsize=13, fontweight="bold")

    ax.axhline(50, color="white", linestyle="--", linewidth=1.1, alpha=0.6)
    ax.text(len(buckets) - 0.5, 52, "50% baseline",
            color="white", fontsize=10, alpha=0.8)

    fig.suptitle("% of months JPY weakened (USD/JPY up)",
                 color="white", fontsize=15, fontweight="bold",
                 x=0.08, y=0.97, ha="left")
    fig.text(0.08, 0.92, "by interest rate differential bucket (US minus Japan)",
             color="#bfbfbf", fontsize=11, ha="left")

    ax.set_xlabel("Interest rate differential (US minus Japan)", color="#dcdcdc", fontsize=11)
    ax.set_ylabel("% of months where JPY weakened", color="#dcdcdc", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis="x", colors="#dcdcdc", labelsize=10)
    ax.tick_params(axis="y", colors="#dcdcdc", labelsize=10)

    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.yaxis.grid(True, color="#333333", linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = FIGURES_DIR / "03_jpy_weakening_by_bucket.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 — Carry-trade cumulative return on $100
# ---------------------------------------------------------------------------
def plot_carry_trade_cumulative(df: pd.DataFrame) -> Path:
    _ensure_figures_dir()
    sim = ana.carry_trade_simulation(df, initial_capital=100.0)

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=120)

    dates = sim["observation_date"]
    value = sim["carry_trade_value"]
    baseline = sim["rate_only_value"]

    ax.plot(dates, value, color="#2d6cb0", linewidth=2.2,
            label="Carry Trade (Interest + FX effect)")
    ax.plot(dates, baseline, color="#555555", linewidth=1.5, linestyle="--",
            label="Interest Rate Differential Only (no FX)")

    ax.axhline(100, color="#888888", linestyle=":", linewidth=1.0)

    ax.fill_between(dates, 100, value, where=value >= 100,
                    color="#c8e6c9", alpha=0.55, label="Profit zone")
    ax.fill_between(dates, 100, value, where=value < 100,
                    color="#ffcdd2", alpha=0.55, label="Loss zone")

    trough_idx = value.idxmin()
    peak_idx = value.idxmax()
    final_idx = value.index[-1]

    ax.annotate(f"Trough: ${value.loc[trough_idx]:.0f}",
                xy=(dates.loc[trough_idx], value.loc[trough_idx]),
                xytext=(dates.loc[trough_idx] + pd.Timedelta(days=120),
                        value.loc[trough_idx] - 4),
                color="#c0392b", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#c0392b"))

    ax.annotate(f"Peak: ${value.loc[peak_idx]:.0f}",
                xy=(dates.loc[peak_idx], value.loc[peak_idx]),
                xytext=(dates.loc[peak_idx] - pd.Timedelta(days=300),
                        value.loc[peak_idx] - 6),
                color="#1e7a36", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#1e7a36"))

    ax.annotate(f"Final: ${value.loc[final_idx]:.0f}",
                xy=(dates.loc[final_idx], value.loc[final_idx]),
                xytext=(dates.loc[final_idx] - pd.Timedelta(days=400),
                        value.loc[final_idx] + 8),
                color="#2d6cb0", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2d6cb0"))

    ax.set_title("JPY/USD Carry Trade — Cumulative Return on $100 Investment (2015–2025)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    out = FIGURES_DIR / "04_carry_trade_cumulative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5 — Illustrative carry-trade diagram
# ---------------------------------------------------------------------------
def plot_carry_trade_diagram() -> Path:
    """Simple schematic showing the borrow-JPY / lend-USD mechanics."""
    _ensure_figures_dir()

    fig, ax = plt.subplots(figsize=(12, 7), dpi=120, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def card(x, y, title, flag, amount_text, subtext, pill_text, pill_color):
        box = mpatches.FancyBboxPatch(
            (x, y), 3.2, 3.0, boxstyle="round,pad=0.1,rounding_size=0.15",
            linewidth=0, facecolor="white")
        ax.add_patch(box)
        ax.text(x + 1.6, y + 2.7, title, ha="center", va="top",
                color="#888", fontsize=10, fontweight="bold")
        ax.text(x + 1.6, y + 2.25, flag, ha="center", va="center", fontsize=22)
        ax.text(x + 1.6, y + 1.75, "Borrow in Japan" if "JP" in title or flag == "JP"
                else "Invest in USA",
                ha="center", va="center", color="#222", fontsize=12, fontweight="bold")
        ax.text(x + 1.6, y + 1.15, amount_text, ha="center", va="center",
                color="#2d6cb0" if "¥" in amount_text else "#1e7a36",
                fontsize=18, fontweight="bold")
        ax.text(x + 1.6, y + 0.75, subtext, ha="center", va="center",
                color="#666", fontsize=9)
        pill = mpatches.FancyBboxPatch(
            (x + 0.6, y + 0.2), 2.0, 0.35,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=0, facecolor=pill_color)
        ax.add_patch(pill)
        ax.text(x + 1.6, y + 0.375, pill_text, ha="center", va="center",
                color="#333", fontsize=9)

    # Step 1 — borrow JPY
    ax.text(1.6 + 0.6, 6.4, "STEP 1", ha="center", color="#777", fontsize=10, fontweight="bold")
    card(0.6, 3.0, "", "JP", "¥15,000", "≈ $100 USD", "1% interest/yr", "#fde0e0")

    # Step 2 — invest USD
    ax.text(6.2 + 0.6, 6.4, "STEP 2", ha="center", color="#777", fontsize=10, fontweight="bold")
    card(6.2, 3.0, "", "US", "$100", "e.g. US Treasuries", "5% return/yr", "#e0f2e0")

    # Arrow + "Convert JPY → USD"
    ax.annotate("", xy=(6.1, 4.5), xytext=(3.9, 4.5),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    ax.text(5.0, 4.8, "Convert\nJPY → USD", ha="center", va="center",
            color="#aaa", fontsize=9)

    # P&L band
    band = mpatches.FancyBboxPatch(
        (0.6, 0.6), 8.8, 1.6, boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=0, facecolor="#f7f0dc")
    ax.add_patch(band)
    ax.text(2.0, 1.7, "YOU PAY (JPY LOAN)", ha="center", color="#a23", fontsize=9, fontweight="bold")
    ax.text(2.0, 1.3, "−$1.00", ha="center", color="#a23", fontsize=17, fontweight="bold")
    ax.text(2.0, 0.9, "1% on $100", ha="center", color="#888", fontsize=9)

    ax.text(5.0, 1.7, "YOU EARN (USD INVEST)", ha="center", color="#1e7a36",
            fontsize=9, fontweight="bold")
    ax.text(5.0, 1.3, "+$5.00", ha="center", color="#1e7a36", fontsize=17, fontweight="bold")
    ax.text(5.0, 0.9, "5% on $100", ha="center", color="#888", fontsize=9)

    ax.text(8.0, 1.7, "NET PROFIT", ha="center", color="#333", fontsize=9, fontweight="bold")
    ax.text(8.0, 1.25, "$4.00", ha="center", color="#111", fontsize=22, fontweight="bold")
    ax.text(8.0, 0.9, "4% carry spread", ha="center", color="#888", fontsize=9)

    # Operator symbols
    ax.text(3.5, 1.3, "+", ha="center", va="center", color="#555", fontsize=22)
    ax.text(6.5, 1.3, "|", ha="center", va="center", color="#bbb", fontsize=22)

    ax.text(5.0, 0.2, "\u26A0 Risk: if the yen strengthens sharply, your $100 buys fewer yen when you repay — wiping out the profit.",
            ha="center", va="center", color="#999", fontsize=9)

    out = FIGURES_DIR / "05_carry_trade_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def generate_all_figures(df: pd.DataFrame) -> list[Path]:
    paths = [
        plot_japan_10y_yield(df),
        plot_rate_differential_scatter(df),
        plot_jpy_weakening_by_bucket(df),
        plot_carry_trade_cumulative(df),
        plot_carry_trade_diagram(),
    ]
    return paths


def main() -> None:
    df = ana.load_clean_data()
    paths = generate_all_figures(df)
    for p in paths:
        print(f"[visualization] wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
