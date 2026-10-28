"""SVIX from the OptionMetrics surface, against the series Martin (2017) publishes.

Martin publishes the one-month SVIX² he computed from S&P 500 index options,
daily from January 1996 to January 2012 (`data_manual/martin_2017/svix2.csv`).
This module computes the same quantity from the OptionMetrics 30-day S&P 500
surface with `svix.svix_squared`, one day at a time, and lines the two up.
It is the check that the SVIX integral is right: CME computes the same
simple-variance construction as its CVOL indexes, but only for options on
interest-rate, currency, energy, metals, and agricultural futures, so there
is no exchange-published SVIX for the S&P 500 to compare against. Martin's
own numbers are the benchmark.

Only the priced grid enters the integral, so the smile is priced with
`rnd.priced_grid` and wrapped in a `RiskNeutralCDF` without its CDF values:
the check exercises `svix_squared` and nothing else.

Run by `doit single_names` (it needs the OptionMetrics pull); the results
feed guide 04, and `test_svix.py::test_svix_matches_martin` runs the same
functions at month-ends.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rnd import RiskNeutralCDF, priced_grid
from settings import config
from svix import svix_squared

MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
SPX = 108105
MATURITY_DAYS = 30
PARQUET = "svix_vs_martin.parquet"


def load_martin_svix(manual_data_dir=MANUAL_DATA_DIR):
    """Martin's published one-month SVIX², as [date, svix2_martin]."""
    m = pd.read_csv(Path(manual_data_dir) / "martin_2017" / "svix2.csv", parse_dates=["date"])
    return m[["date", "m1"]].rename(columns={"m1": "svix2_martin"})


def spx_surface(raw_surface, prices, days=MATURITY_DAYS):
    """The cleaned S&P 500 surface at one maturity (Martin and Shi's Appendix D filters)."""
    import single_name_bounds as snb

    raw = raw_surface[(raw_surface["secid"] == SPX) & (raw_surface["days"] == days)]
    return snb.clean_surface(raw, prices[prices["secid"] == SPX])


def svix_from_surface(clean, rates, dates=None, days=MATURITY_DAYS):
    """SVIX² from each day's smile on the surface, as [date, svix2].

    ``rates`` maps (date, days) to a decimal rate, as
    ``single_name_bounds.rates_by_maturity`` returns. ``dates`` restricts the
    computation to those dates.
    """
    if dates is not None:
        clean = clean[clean["date"].isin(pd.DatetimeIndex(dates))]
    rows = []
    for date, smile in clean.groupby("date"):
        rate = rates.get((date, days))
        if rate is None:
            continue
        try:
            grid, call_grid, tau = priced_grid(smile, rate)
        except ValueError:
            continue
        cdf = RiskNeutralCDF(grid, np.zeros_like(grid), call_grid=call_grid, maturity_years=tau, rate=rate)
        rows.append((date, svix_squared(cdf)))
    return pd.DataFrame(rows, columns=["date", "svix2"])


def compare_with_martin(ours, martin):
    """Merge on date and express both in VIX units (annualized volatility, percent)."""
    merged = ours.merge(martin, on="date", how="inner")
    merged["svix"] = 100 * np.sqrt(merged["svix2"])
    merged["svix_martin"] = 100 * np.sqrt(merged["svix2_martin"])
    merged["gap"] = merged["svix"] - merged["svix_martin"]
    return merged


def load_svix_vs_martin(output_dir=OUTPUT_DIR):
    return pd.read_parquet(Path(output_dir) / PARQUET)


if __name__ == "__main__":
    import single_name_bounds as snb
    from pull_optionmetrics import load_security_prices, load_vol_surface, load_zero_curve

    martin = load_martin_svix()
    clean = spx_surface(load_vol_surface(), load_security_prices())
    clean = clean[clean["date"] <= martin["date"].max()]
    ours = svix_from_surface(clean, snb.rates_by_maturity(load_zero_curve()))
    merged = compare_with_martin(ours, martin)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_DIR / PARQUET)
    print(
        f"SVIX vs Martin (2017), {len(merged):,} days "
        f"({merged['date'].min().date()} to {merged['date'].max().date()}): "
        f"correlation {np.corrcoef(merged['svix'], merged['svix_martin'])[0, 1]:.4f}, "
        f"median absolute gap {merged['gap'].abs().median():.2f} vol points"
    )
