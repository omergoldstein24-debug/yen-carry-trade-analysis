"""
main.py
-------
End-to-end driver for the yen carry-trade analysis.

Running this script will:
    1. Rebuild the clean monthly dataset from the raw sources.
    2. Print the regression and bucket summary statistics.
    3. Regenerate all five figures under `figures/`.

Usage:
    python main.py
"""

from __future__ import annotations

from src import analysis, data_pipeline, visualization


def main() -> None:
    print("=" * 72)
    print("YEN CARRY-TRADE ANALYSIS")
    print("=" * 72)

    # Step 1 — build the clean monthly dataset from raw sources
    print("\n[1/3] Building clean dataset...")
    clean_df = data_pipeline.build_clean_dataset()
    out_path = data_pipeline.save_clean_dataset(clean_df)
    print(f"      -> {len(clean_df)} monthly rows written to {out_path.name}")

    # Step 2 — descriptive statistics
    print("\n[2/3] Computing descriptive statistics...")
    reg = analysis.regression_stats(clean_df)
    print(f"      Rate-differential vs. monthly FX move:")
    print(f"         Pearson r = {reg.correlation:+.2f}")
    print(f"         OLS slope = {reg.slope:+.2f}%/pp")
    print(f"         R-squared = {reg.r_squared:.3f}")

    buckets = analysis.bucketed_weakening(clean_df)
    print("\n      JPY weakening frequency by differential bucket:")
    for _, row in buckets.iterrows():
        label = str(row["bucket"]).replace("\n", " ")
        print(f"         {label:<22} -> {row['pct_jpy_weakened']:5.1f}%  "
              f"(n = {int(row['months'])})")

    sim = analysis.carry_trade_simulation(clean_df, initial_capital=100.0)
    print(f"\n      Carry trade $100 start ->  peak ${sim['carry_trade_value'].max():.0f}, "
          f"trough ${sim['carry_trade_value'].min():.0f}, "
          f"end ${sim['carry_trade_value'].iloc[-1]:.0f}")

    # Step 3 — figures
    print("\n[3/3] Generating figures...")
    paths = visualization.generate_all_figures(clean_df)
    for p in paths:
        print(f"      -> {p.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
