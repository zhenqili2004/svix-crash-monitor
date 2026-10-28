"""Tests for the single-name part: the cleaning on synthetic rows, and the
check of the index crash probability against Martin's published series."""

from pathlib import Path

import numpy as np
import pandas as pd

import single_name_bounds as snb
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))


def test_clean_surface_filters():
    date = pd.Timestamp("2024-06-28")
    rows = []
    for i, delta in enumerate(range(-90, 0, 5)):  # 18 strikes for SPX, all valid
        rows.append(dict(secid=108105, date=date, days=30, delta=delta, impl_volatility=0.15,
                         impl_strike=5000 + 20 * i, dispersion=0.01, cp_flag="P"))
    rows.append(dict(secid=108105, date=date, days=30, delta=-95, impl_volatility=np.nan,
                     impl_strike=-99.99, dispersion=np.nan, cp_flag="P"))  # failed fit
    rows.append(dict(secid=108105, date=date, days=30, delta=50, impl_volatility=0.15,
                     impl_strike=5500, dispersion=0.08, cp_flag="C"))  # dispersion too high
    for i in range(5):  # a name with too few strikes
        rows.append(dict(secid=101594, date=date, days=30, delta=-50 + i, impl_volatility=0.3,
                         impl_strike=200 + i, dispersion=0.01, cp_flag="P"))
    raw = pd.DataFrame(rows)
    prices = pd.DataFrame({"secid": [108105, 101594], "date": [date, date], "close": [5460.0, 210.0]})
    clean = snb.clean_surface(raw, prices)
    assert set(clean["secid"]) == {108105}
    assert len(clean) == 18
    assert np.allclose(clean["moneyness"], (5000 + 20 * np.arange(18)) / 5460.0)


def test_index_probabilities_computed():
    """`doit single_names` ran the chain on the pulled surface: the index
    table exists, covers the sample, and its three probabilities are ordered
    the way risk aversion orders them."""
    index = pd.read_parquet(OUTPUT_DIR / "index_crash_probabilities.parquet")
    assert {"date", "days_to_maturity", "threshold", "prob_riskneutral", "prob_log_utility", "prob_power_utility"} <= set(index.columns)
    assert index["date"].min().year == 1996 and index["date"].max().year >= 2024
    assert set(index["days_to_maturity"].unique()) == set(snb.MATURITIES_DAYS)
    one_month = index[(index["days_to_maturity"] == 30) & (index["threshold"] == 0.80)]
    assert len(one_month) > 7000
    assert (one_month["prob_power_utility"] <= one_month["prob_log_utility"] + 1e-9).all()
    assert (one_month["prob_log_utility"] <= one_month["prob_riskneutral"] + 1e-9).all()


def test_spx_crash_probability_matches_martin():
    """The log-investor probability of a 20% drop at one month, averaged by
    month, against the series Martin (2017) publishes, on the months where
    they overlap."""
    ours = pd.read_parquet(OUTPUT_DIR / "index_crash_probabilities.parquet")
    ours = ours[(ours["days_to_maturity"] == 30) & (ours["threshold"] == 0.80)].copy()
    ours["year"] = ours["date"].dt.year
    ours["month"] = ours["date"].dt.month
    monthly = ours.groupby(["year", "month"])["prob_log_utility"].mean().reset_index()
    martin = pd.read_csv(MANUAL_DATA_DIR / "martin_2017" / "crashprob.csv")
    merged = monthly.merge(martin[["year", "month", "m1"]], on=["year", "month"])
    assert len(merged) > 100
    gap = (merged["prob_log_utility"] - merged["m1"]).abs()
    assert gap.median() < 0.005
    assert np.corrcoef(merged["prob_log_utility"], merged["m1"])[0, 1] > 0.9
