"""Pull E-mini S&P 500 option settlement prices from Databento's CME Globex feed.

What is pulled
--------------
End-of-month E-mini S&P 500 options (CME root ``EW``: European-style, PM-settled
at 4:00 p.m. ET, exercising into the nearest quarterly ES future). Two
schemas, both on the ``GLBX.MDP3`` dataset:

- ``statistics``: the exchange's daily statistics for every listed contract.
  We keep ``stat_type == 3``, the **settlement price**, which CME publishes
  for every listed strike whether or not it traded that day. (Daily bars,
  ``ohlcv-1d``, exist only for strikes that traded, which is fewer than
  half of them on a typical day.) Each contract carries a preliminary and a
  final settlement; the last one by ``ts_event`` is the final.
- ``definition``: the exchange's contract specifications, which say what each
  ``instrument_id`` is: expiration date, strike, call or put, underlying
  future. The symbol ``EWV6 C7920`` names a series (EW, October, 2026), a
  right, and a strike, but not the expiration date; the definition record
  does. Parent-symbol queries also return user-defined spreads, which the
  definition flags and we drop.

Both are pulled one session at a time, because Databento's streaming endpoint
is happiest with small requests and because the daily monitor only ever needs
the latest session. The ES futures' own settlements (``ES.FUT``) are pulled
too, so the parity-implied forward can be checked against the exchange's
price for the underlying future.

Cost
----
Everything on ``GLBX.MDP3`` is covered by the course's Databento
subscription, so every query here costs $0.00. Each query is priced with the
free ``metadata.get_cost`` endpoint first, and the pull refuses to download
anything whose estimate is not $0.00, so running this script never spends
money.

Conventions
-----------
``pull_*`` hits the network and, as a script, caches to ``DATA_DIR``;
``load_*`` reads the cache. Refresh with ``doit forget pull && doit``. As a
script, ``--latest`` pulls the most recent session only (the daily monitor's
window, ``latest_window``) instead of START_DATE to END_DATE, and replaces
the cache with it; the default ``doit`` pulls the whole window.
"""

from pathlib import Path

import pandas as pd

import source_status
from settings import config

DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
DATABENTO_API_KEY = config("DATABENTO_API_KEY", default="")

DATASET = "GLBX.MDP3"
SYMBOLS = ["EW.OPT"]  # parent symbology: every listed end-of-month E-mini option
FUTURES_SYMBOLS = ["ES.FUT"]  # every listed E-mini S&P 500 future
STYPE_IN = "parent"
SETTLEMENT_STAT_TYPE = 3  # Databento statistics schema: 3 = settlement price

OPTIONS_PARQUET = "es_options_settlements.parquet"
FUTURES_PARQUET = "es_futures_settlements.parquet"


def build_queries(start_date, end_date):
    """The three query dicts for one window.

    The same dict feeds ``metadata.get_cost`` and ``timeseries.get_range``,
    so the cost guard always prices exactly the query that is pulled.
    """
    options = dict(
        dataset=DATASET,
        symbols=SYMBOLS,
        stype_in=STYPE_IN,
        start=start_date,
        end=end_date,
    )
    futures = dict(
        dataset=DATASET,
        symbols=FUTURES_SYMBOLS,
        stype_in=STYPE_IN,
        schema="statistics",
        start=start_date,
        end=end_date,
    )
    return {
        "option_statistics": {**options, "schema": "statistics"},
        "option_definitions": {**options, "schema": "definition"},
        "futures_statistics": futures,
    }


def clamp_end_date(client, end_date):
    """Clamp the requested end date to what the dataset has.

    Historical data lags real time by less than a day; asking past the
    available end raises, so clamp instead of guessing.
    """
    available_end = pd.Timestamp(with_retry(client.metadata.get_dataset_range, DATASET)["end"])
    if available_end.tzinfo is not None:
        available_end = available_end.tz_convert("UTC").tz_localize(None)
    end = min(pd.Timestamp(end_date), available_end)
    return end.strftime("%Y-%m-%d")


def assert_query_is_free(cost, what=""):
    """Abort (SystemExit) unless Databento's estimate for the query is $0.00.

    DO NOT MODIFY OR BYPASS THIS GUARD. Everything this project pulls is
    covered by the course's CME Globex subscription, so the estimate should
    always be $0.00. If it is not, the query was edited or the API key
    belongs to an account without the subscription, and we stop before
    downloading anything.
    """
    print(f"Databento cost estimate {what}: ${cost:,.4f}")
    if cost > 0:
        print(
            f"\nABORTING: the estimate is ${cost:,.4f}, but everything this "
            "project pulls should be free ($0.00) under the course's Databento "
            "subscription. Nothing was downloaded and nothing was charged.\n"
            "Check that your API key belongs to an account covered by the "
            "course subscription and that the query has not been modified."
        )
        raise SystemExit(1)


def sessions_between(start_date, end_date):
    """Weekdays in [start_date, end_date), one query window per session."""
    days = pd.bdate_range(start_date, end_date, inclusive="left")
    return [d.strftime("%Y-%m-%d") for d in days]


LATEST_DAYS = 4  # calendar days back from today; see latest_window


def latest_window(today=None):
    """[today - LATEST_DAYS, today): the window that holds the most recent session.

    The daily monitor pulls this window (``--latest``) instead of the
    START_DATE..END_DATE window from the settings. Four calendar days reach
    Friday's session from a Monday run, and the previous Friday's from the
    Tuesday after a Monday holiday. Sessions already in the history are
    skipped by ``monitor.py``, so a window with two sessions costs one extra
    pull and changes nothing else.
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(days=LATEST_DAYS)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def final_settlements(stats):
    """The final settlement price of each contract on each session.

    Parameters
    ----------
    stats : DataFrame
        A ``statistics`` pull, as returned by ``DBNStore.to_df()``, reset so
        that ``ts_event`` is a column.

    Returns
    -------
    DataFrame with columns date, instrument_id, symbol, settle, ts_event.
    ``date`` is the session the statistic refers to (``ts_ref``), as a
    tz-naive date. A session's final settlement is re-broadcast when Globex
    reopens in the evening, after midnight UTC, so a query window for one
    calendar day also returns the previous session's settlements; the
    caller drops those duplicates across windows by ``ts_event``.
    """
    s = stats[stats["stat_type"] == SETTLEMENT_STAT_TYPE].copy()
    # A settlement of zero is a placeholder on a contract the exchange has
    # not valued (a strike listed that day, typically), not a price.
    s = s[s["price"] > 0]
    s["date"] = (
        pd.to_datetime(s["ts_ref"], utc=True).dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    )
    s = s.sort_values("ts_event").drop_duplicates(["date", "instrument_id"], keep="last")
    s = s.rename(columns={"price": "settle"})
    return s[["date", "instrument_id", "symbol", "settle", "ts_event"]].reset_index(drop=True)


def option_contracts(definitions):
    """One row per outright call or put in a ``definition`` pull.

    Drops user-defined spreads and other multi-leg instruments (anything whose
    ``instrument_class`` is not C or P). Returns instrument_id, symbol,
    expiration (tz-naive date), strike, cp_flag, underlying.
    """
    d = definitions[definitions["instrument_class"].isin(["C", "P"])].copy()
    d = d.drop_duplicates("instrument_id", keep="last")
    d["expiration"] = (
        pd.to_datetime(d["expiration"], utc=True).dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    )
    d = d.rename(columns={"strike_price": "strike", "instrument_class": "cp_flag"})
    return d[["instrument_id", "symbol", "expiration", "strike", "cp_flag", "underlying"]].reset_index(
        drop=True
    )


def outright_futures(settlements):
    """Keep the outright ES contracts (drop calendar spreads such as ESZ6-ESH7)."""
    s = settlements[~settlements["symbol"].str.contains("-", regex=False)]
    return s.reset_index(drop=True)


def is_transient_databento_error(error):
    """Server errors, dropped connections, and network errors: worth retrying.

    Databento's gateway times out now and then (HTTP 500 and 504), and a
    download can end early ("Response ended prematurely"), on the free cost
    check as well as on the download. A client error (HTTP 4xx), such as a
    rejected key or a bad request, is not transient.
    """
    import databento as db
    import requests

    if isinstance(error, db.common.error.BentoClientError):
        return False
    return isinstance(error, (db.common.error.BentoError, requests.exceptions.RequestException))


def with_retry(call, *args, **kwargs):
    """Call a Databento client method, retrying transient errors.

    Retrying is safe: an error says nothing about cost, and the cost guard
    still runs on every query. See ``source_status.retry``.
    """
    return source_status.retry(call, *args, source="Databento", transient=is_transient_databento_error, **kwargs)


def get_range_with_retry(client, query):
    """``timeseries.get_range`` as a DataFrame, with retries."""
    return with_retry(client.timeseries.get_range, **query).to_df().reset_index()


def pull_es_options(start_date=START_DATE, end_date=END_DATE, client=None):
    """Pull settlement prices and contract definitions, one session at a time.

    Returns (options, futures): the option settlements joined to their
    definitions, and the outright ES futures settlements.
    """
    import databento as db

    if client is None:
        client = db.Historical(key=DATABENTO_API_KEY)
    end_date = clamp_end_date(client, end_date)

    options, futures = [], []
    for session in sessions_between(start_date, end_date):
        next_day = (pd.Timestamp(session) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        queries = build_queries(session, next_day)
        frames = {}
        for name, query in queries.items():
            cost = with_retry(client.metadata.get_cost, **query)
            assert_query_is_free(cost, what=f"({name}, {session})")
            frames[name] = get_range_with_retry(client, query)
        settled = final_settlements(frames["option_statistics"])
        contracts = option_contracts(frames["option_definitions"])
        if settled.empty or contracts.empty:
            print(f"  {session}: no settlements (holiday?)")
            continue
        merged = settled.merge(contracts.drop(columns="symbol"), on="instrument_id", how="inner")
        merged = merged.drop_duplicates(["date", "instrument_id"], keep="last")
        merged["days_to_expiry"] = (merged["expiration"] - merged["date"]).dt.days
        options.append(merged)
        futures.append(outright_futures(final_settlements(frames["futures_statistics"])))
        print(f"  {session}: {len(merged):,} settled options, {len(futures[-1]):,} futures")

    if not options:
        raise ValueError(f"No sessions with settlements between {start_date} and {end_date}")
    # The evening re-broadcast of a session's settlements lands in the next
    # calendar day's window; keep the last record per session and contract.
    options = pd.concat(options, ignore_index=True)
    options = options.sort_values("ts_event").drop_duplicates(["date", "instrument_id"], keep="last")
    futures = pd.concat(futures, ignore_index=True)
    futures = futures.sort_values("ts_event").drop_duplicates(["date", "instrument_id"], keep="last")
    futures = futures.drop(columns="ts_event").sort_values(["date", "symbol"]).reset_index(drop=True)
    columns = [
        "date", "symbol", "instrument_id", "expiration", "days_to_expiry",
        "strike", "cp_flag", "settle", "underlying",
    ]
    return options.sort_values(["date", "expiration", "strike", "cp_flag"])[columns].reset_index(drop=True), futures


def load_es_options(data_dir=DATA_DIR):
    """Load the cached option settlements (one row per contract per session)."""
    return pd.read_parquet(Path(data_dir) / OPTIONS_PARQUET)


def load_es_futures(data_dir=DATA_DIR):
    """Load the cached ES futures settlements."""
    return pd.read_parquet(Path(data_dir) / FUTURES_PARQUET)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pull E-mini option settlements from Databento into DATA_DIR.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help=f"pull the most recent session only (the last {LATEST_DAYS} calendar days), "
        "as the daily monitor does, instead of START_DATE to END_DATE",
    )
    args = parser.parse_args()
    source_status.clear()
    if not DATABENTO_API_KEY:
        raise SystemExit(
            "DATABENTO_API_KEY is not set. Copy .env.example to .env and add "
            "your key (https://databento.com, Account > API keys)."
        )
    start_date, end_date = latest_window() if args.latest else (START_DATE, END_DATE)
    options, futures = pull_es_options(start_date, end_date)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    options.to_parquet(DATA_DIR / OPTIONS_PARQUET)
    futures.to_parquet(DATA_DIR / FUTURES_PARQUET)
    print(
        f"Saved {len(options):,} option settlements over "
        f"{options['date'].nunique()} sessions "
        f"({options['date'].min().date()} to {options['date'].max().date()}) "
        f"to {DATA_DIR / OPTIONS_PARQUET}"
    )
