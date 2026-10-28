"""Pull the OptionMetrics volatility surface, prices, and zero curve from WRDS.

This is the data for the single-name part: Martin and Shi (2025) build their
crash bounds from the IvyDB **standardized volatility surface**, which
OptionMetrics fits each day for each security at fixed maturities (30, 60,
91, ... days) and fixed deltas (10 to 90, puts and calls). Their Appendix D
filters on the surface's ``dispersion`` (a goodness-of-fit measure) and keys
moneyness off ``impl_strike``, both of which exist only in the surface table,
so the surface is the paper's specification, not a shortcut.

Three WRDS tables in the ``optionm_all`` library:

- ``vsurfd{YYYY}``: the surface, one table per year. Columns secid, date,
  days, delta, impl_volatility, impl_strike, impl_premium, dispersion,
  cp_flag.
- ``secprd{YYYY}``: daily security prices, one table per year. We need
  ``close`` for the spot.
- ``zerocd``: the zero-coupon yield curve (date, days, rate in percent).

The universe is scoped to the S&P 500 index and two names, Apple and AIG,
the two names in the paper's Figure 1, at the 30- and 91-day maturities,
so the pull is a few minutes rather than the hour the full constituent panel
takes. The query helpers return SQL strings so that they can be tested
against a small in-memory database without a WRDS connection.

Adapted from Riley Haas and Marija Jovicic's ``forecasting_crashes_smile``
(FINM 32800, Summer 2026), used with their permission.

HOMEWORK INSTRUCTIONS
---------------------
``build_surface_query`` has had its ``WHERE`` clause removed. Its docstring
says what the query must select. Check your work, offline, with::

    pytest -vv ./src/test_pull_optionmetrics.py
"""

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = config("DATA_DIR")
WRDS_USERNAME = config("WRDS_USERNAME", default="")
SURFACE_START_DATE = config("SURFACE_START_DATE")
SURFACE_END_DATE = config("SURFACE_END_DATE")

OPTIONM_LIB = "optionm_all"
SECIDS = {"SPX": 108105, "AAPL": 101594, "AIG": 101397}
MATURITIES_DAYS = [30, 91]

SURFACE_PARQUET = "optionm_vol_surface.parquet"
PRICES_PARQUET = "optionm_security_prices.parquet"
ZERO_CURVE_PARQUET = "optionm_zero_curve.parquet"


def _sql_int_list(values):
    """Render ints as the body of a SQL ``IN`` list: ``1, 2, 3``."""
    return ", ".join(str(int(v)) for v in values)


def build_surface_query(
    year, secids=tuple(SECIDS.values()), maturities=tuple(MATURITIES_DAYS),
    start_date=SURFACE_START_DATE, end_date=SURFACE_END_DATE,
):
    """SQL for one year of the volatility surface.

    Select ``secid, date, days, cp_flag, delta, impl_volatility, impl_strike,
    dispersion`` from ``optionm_all.vsurfd{year}`` for the given securities
    (``secid IN ...``), the given maturities (``days IN ...``), and dates
    between ``start_date`` and ``end_date`` inclusive. Keep every delta and
    both calls and puts: the bound integrates across the whole smile.

    ``start_date`` and ``end_date`` are ``YYYY-MM-DD`` strings;
    ``_sql_int_list`` renders the two ``IN`` lists.
    """
    return f"""
        SELECT secid, date, days, cp_flag, delta,
               impl_volatility, impl_strike, dispersion
        FROM {OPTIONM_LIB}.vsurfd{year}
        WHERE TODO
    """
    ## TODO: Replace `WHERE TODO` above with the three conditions the docstring
    ## describes. `build_price_query`, below, shows two of them on another table.
    ## Check your work, offline, with `pytest ./src/test_pull_optionmetrics.py`.


def build_price_query(year, secids=tuple(SECIDS.values()), start_date=SURFACE_START_DATE, end_date=SURFACE_END_DATE):
    """SQL for one year of daily closing prices."""
    return f"""
        SELECT secid, date, close
        FROM {OPTIONM_LIB}.secprd{year}
        WHERE secid IN ({_sql_int_list(secids)})
          AND date BETWEEN '{start_date}' AND '{end_date}'
    """


def build_zero_curve_query(start_date=SURFACE_START_DATE, end_date=SURFACE_END_DATE):
    """SQL for the zero-coupon curve over the window."""
    return f"""
        SELECT date, days, rate
        FROM {OPTIONM_LIB}.zerocd
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date, days
    """


def _years(start_date, end_date):
    return range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1)


def _connect(wrds_username=WRDS_USERNAME):
    import wrds

    return wrds.Connection(wrds_username=wrds_username)


def pull_vol_surface(start_date=SURFACE_START_DATE, end_date=SURFACE_END_DATE, db=None):
    """Pull the surface year by year and concatenate."""
    own = db is None
    db = db or _connect()
    frames = []
    try:
        for year in _years(start_date, end_date):
            try:
                df = db.raw_sql(build_surface_query(year, start_date=start_date, end_date=end_date), date_cols=["date"])
            except Exception as error:  # a year's table may not exist yet
                print(f"[pull_vol_surface] {year}: skipped ({str(error).splitlines()[0][:80]})")
                continue
            print(f"[pull_vol_surface] {year}: {len(df):,} rows", flush=True)
            frames.append(df)
    finally:
        if own:
            db.close()
    df = pd.concat(frames, ignore_index=True)
    df["secid"] = df["secid"].astype("int64")
    df["days"] = df["days"].astype("int64")
    return df


def pull_security_prices(start_date=SURFACE_START_DATE, end_date=SURFACE_END_DATE, db=None):
    """Pull daily closes year by year and concatenate."""
    own = db is None
    db = db or _connect()
    frames = []
    try:
        for year in _years(start_date, end_date):
            try:
                df = db.raw_sql(build_price_query(year, start_date=start_date, end_date=end_date), date_cols=["date"])
            except Exception as error:
                print(f"[pull_security_prices] {year}: skipped ({str(error).splitlines()[0][:80]})")
                continue
            frames.append(df)
    finally:
        if own:
            db.close()
    df = pd.concat(frames, ignore_index=True)
    df["secid"] = df["secid"].astype("int64")
    return df


def pull_zero_curve(start_date=SURFACE_START_DATE, end_date=SURFACE_END_DATE, db=None):
    own = db is None
    db = db or _connect()
    try:
        df = db.raw_sql(build_zero_curve_query(start_date, end_date), date_cols=["date"])
    finally:
        if own:
            db.close()
    return df


def load_vol_surface(data_dir=DATA_DIR):
    return pd.read_parquet(Path(data_dir) / SURFACE_PARQUET)


def load_security_prices(data_dir=DATA_DIR):
    return pd.read_parquet(Path(data_dir) / PRICES_PARQUET)


def load_zero_curve(data_dir=DATA_DIR):
    return pd.read_parquet(Path(data_dir) / ZERO_CURVE_PARQUET)


if __name__ == "__main__":
    if not WRDS_USERNAME:
        raise SystemExit("WRDS_USERNAME is not set. Copy .env.example to .env and add it.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = _connect()
    try:
        zero = pull_zero_curve(db=db)
        zero.to_parquet(DATA_DIR / ZERO_CURVE_PARQUET)
        print(f"Saved {len(zero):,} zero-curve rows")
        prices = pull_security_prices(db=db)
        prices.to_parquet(DATA_DIR / PRICES_PARQUET)
        print(f"Saved {len(prices):,} price rows")
        surface = pull_vol_surface(db=db)
        surface.to_parquet(DATA_DIR / SURFACE_PARQUET)
        print(
            f"Saved {len(surface):,} surface rows "
            f"({surface['date'].min().date()} to {surface['date'].max().date()}) "
            f"to {DATA_DIR / SURFACE_PARQUET}"
        )
    finally:
        db.close()
