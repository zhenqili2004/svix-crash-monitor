"""Crash bounds for single names from the OptionMetrics surface, Martin and Shi (2025).

The pipeline for the single-name part:

1. ``clean_surface``: the paper's Appendix D filters on the raw surface, and
   moneyness = impl_strike / close with the spot from the price file.
2. ``rates_by_maturity``: the zero curve interpolated to each surface
   maturity, in decimals.
3. ``compute_bounds``: for each month-end, name, and maturity, the
   risk-neutral CDF of the name and of the index, and the Frechet-Hoeffding
   bounds of Result 3 on the probability of a 20% drop. This is the paper's
   Figure 1 for Apple and AIG.
4. ``compute_index_crash_probabilities``: for the index alone, every day,
   the risk-neutral probability and the log-utility and power-utility
   probabilities of a 20% drop. Averaged by month, the log-utility series
   is what Martin (2017) publishes as ``crashprob.xls``, which is the
   check on this half of the pipeline.

Written from Riley Haas and Marija Jovicic's ``forecasting_crashes_smile``
(FINM 32800, Summer 2026), used with their permission.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bounds import crash_bounds
from crash_prob import crash_probability, crash_probability_power, risk_neutral_crash_prob
from pull_optionmetrics import MATURITIES_DAYS, SECIDS
from rnd import risk_neutral_cdf
from settings import config

OUTPUT_DIR = config("OUTPUT_DIR")
THRESHOLDS = [0.80, 0.90]
SPX = SECIDS["SPX"]
BOUNDS_PARQUET = "single_name_bounds.parquet"
INDEX_PARQUET = "index_crash_probabilities.parquet"


def clean_surface(raw_surface, prices):
    """Appendix D of Martin and Shi: the filters, then moneyness.

    1. a spot price exists for the security on that date (inner join),
    2. strike > 0 (OptionMetrics marks a failed fit with -99.99),
    3. dispersion strictly inside (0, 0.05),
    4. more than 10 distinct strikes per security, date, maturity.
    """
    df = raw_surface.copy()
    df = df[(df["impl_strike"] > 0) & (df["dispersion"] > 0) & (df["dispersion"] < 0.05)]
    spot = prices.rename(columns={"close": "spot_price"})[["secid", "date", "spot_price"]]
    df = df.merge(spot, on=["secid", "date"], how="inner")
    df["moneyness"] = df["impl_strike"] / df["spot_price"]
    df = df.rename(columns={"days": "days_to_maturity", "impl_volatility": "implied_vol"})
    n_strikes = df.groupby(["date", "secid", "days_to_maturity"])["impl_strike"].transform("nunique")
    df = df[n_strikes > 10]
    columns = ["date", "secid", "days_to_maturity", "moneyness", "implied_vol", "spot_price", "cp_flag"]
    return df[columns].sort_values(["date", "secid", "days_to_maturity", "moneyness"]).reset_index(drop=True)


def rates_by_maturity(zero_curve, maturities=MATURITIES_DAYS):
    """Interpolate the zero curve to the surface maturities. Returns a dict {(date, days): rate as a decimal}."""
    out = {}
    zc = zero_curve.copy()
    zc["date"] = pd.to_datetime(zc["date"])
    for date, g in zc.sort_values("days").groupby("date"):
        days = g["days"].to_numpy(dtype=float)
        rate = g["rate"].to_numpy(dtype=float)
        for m in maturities:
            out[(date, int(m))] = float(np.interp(m, days, rate)) / 100.0
    return out


def month_ends(dates):
    """The last date of each calendar month among ``dates``."""
    s = pd.Series(pd.DatetimeIndex(dates).unique()).sort_values()
    return s.groupby([s.dt.year, s.dt.month]).max().to_numpy()


def _cdfs_for(day, rates):
    """Risk-neutral CDFs for every (secid, maturity) smile of one date."""
    cdfs = {}
    for (secid, days), smile in day.groupby(["secid", "days_to_maturity"]):
        rate = rates.get((smile["date"].iloc[0], int(days)))
        if rate is None:
            continue
        try:
            cdfs[(int(secid), int(days))] = risk_neutral_cdf(smile, rate)
        except ValueError:
            continue
    return cdfs


def compute_bounds(clean, rates, thresholds=THRESHOLDS, names=None):
    """Result 3 bounds for each name against the index, on month-ends."""
    names = names or [s for s in SECIDS.values() if s != SPX]
    ticker = {v: k for k, v in SECIDS.items()}
    sel = clean[clean["date"].isin(month_ends(clean["date"]))]
    rows = []
    for date, day in sel.groupby("date"):
        cdfs = _cdfs_for(day, rates)
        for days in MATURITIES_DAYS:
            market = cdfs.get((SPX, days))
            if market is None:
                continue
            for secid in names:
                cdf_i = cdfs.get((secid, days))
                if cdf_i is None:
                    continue
                for q in thresholds:
                    lower, p_star, upper = crash_bounds(cdf_i, market, q)
                    rows.append(
                        dict(date=date, secid=secid, ticker=ticker[secid], days_to_maturity=days,
                             threshold=q, bound_lower=lower, prob_riskneutral=p_star, bound_upper=upper)
                    )
    return pd.DataFrame(rows)


def compute_index_crash_probabilities(clean, rates, thresholds=THRESHOLDS):
    """Daily index crash probabilities: risk-neutral, log utility, power utility."""
    spx = clean[clean["secid"] == SPX]
    rows = []
    for (date, days), smile in spx.groupby(["date", "days_to_maturity"]):
        rate = rates.get((date, int(days)))
        if rate is None:
            continue
        try:
            cdf = risk_neutral_cdf(smile, rate)
        except ValueError:
            continue
        for q in thresholds:
            rows.append(
                dict(date=date, days_to_maturity=int(days), threshold=q,
                     prob_riskneutral=risk_neutral_crash_prob(cdf, q),
                     prob_log_utility=crash_probability(cdf, q),
                     prob_power_utility=crash_probability_power(cdf, q))
            )
    return pd.DataFrame(rows)


def load_single_name_bounds(output_dir=OUTPUT_DIR):
    return pd.read_parquet(Path(output_dir) / BOUNDS_PARQUET)


def load_index_crash_probabilities(output_dir=OUTPUT_DIR):
    return pd.read_parquet(Path(output_dir) / INDEX_PARQUET)


if __name__ == "__main__":
    from pull_optionmetrics import load_security_prices, load_vol_surface, load_zero_curve

    clean = clean_surface(load_vol_surface(), load_security_prices())
    rates = rates_by_maturity(load_zero_curve())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = compute_index_crash_probabilities(clean, rates)
    index.to_parquet(OUTPUT_DIR / INDEX_PARQUET)
    print(f"Saved {len(index):,} index rows ({index['date'].min().date()} to {index['date'].max().date()})")
    bounds = compute_bounds(clean, rates)
    bounds.to_parquet(OUTPUT_DIR / BOUNDS_PARQUET)
    print(f"Saved {len(bounds):,} single-name rows")
