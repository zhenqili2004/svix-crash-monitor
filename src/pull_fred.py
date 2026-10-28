"""Pull the VIX and a one-month Treasury yield from FRED (no API key).

Two daily series, both from FRED's public ``fredgraph.csv`` endpoint:

- ``VIXCLS``: the Cboe VIX, the benchmark the monitor plots SVIX against.
- ``DGS1MO``: the one-month Treasury constant-maturity yield, in percent,
  used as the risk-free rate when discounting one-month options.

Follows the project convention: ``pull_*`` hits the network and, as a
script, caches to ``DATA_DIR``; ``load_*`` reads the cache.
"""

import io
from pathlib import Path

import pandas as pd
import requests

import source_status
from settings import config

DATA_DIR = config("DATA_DIR")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = {"VIXCLS": "vix", "DGS1MO": "dgs1mo"}
FRED_START = "1990-01-02"
PARQUET_NAME = "fred_vix_rates.parquet"


def parse_fred_csv(text, name):
    """Parse a fredgraph.csv payload into a tidy DataFrame [date, name]."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.dropna(subset=[name]).reset_index(drop=True)


class FredServerError(Exception):
    """FRED answered with a 5xx status: a server problem, worth retrying."""


def _get_csv(params):
    """One request to FRED's CSV endpoint."""
    response = requests.get(FRED_CSV_URL, params=params, timeout=30)
    if response.status_code >= 500:
        raise FredServerError(f"HTTP {response.status_code} from FRED")
    response.raise_for_status()  # a 4xx is a bad request, not an outage
    return response.text


def pull_fred_series(series_id, name, start_date=FRED_START):
    """Pull one FRED series, retrying server and network errors."""
    params = {"id": series_id, "cosd": start_date}
    transient = (FredServerError, requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    text = source_status.retry(_get_csv, params, source="FRED", transient=transient)
    df = parse_fred_csv(text, name)
    if df.empty:
        raise ValueError(f"FRED returned no observations for {series_id}")
    return df


def pull_fred(start_date=FRED_START):
    """Pull VIXCLS and DGS1MO and merge them on date (outer join)."""
    frames = [pull_fred_series(sid, name, start_date) for sid, name in SERIES.items()]
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="date", how="outer")
    return df.sort_values("date").reset_index(drop=True)


def load_fred(data_dir=DATA_DIR):
    """Load the cached FRED series pulled by this module."""
    return pd.read_parquet(Path(data_dir) / PARQUET_NAME)


def rate_on(fred, date, column="dgs1mo"):
    """The latest yield on or before ``date``, as a decimal (0.045 = 4.5%)."""
    s = fred.dropna(subset=[column]).set_index("date")[column]
    s = s[s.index <= pd.Timestamp(date)]
    if s.empty:
        raise ValueError(f"no {column} observation on or before {date}")
    return float(s.iloc[-1]) / 100.0


if __name__ == "__main__":
    df = pull_fred()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / PARQUET_NAME)
    print(
        f"Saved {len(df):,} rows of VIX and DGS1MO "
        f"({df['date'].min().date()} to {df['date'].max().date()}) "
        f"to {DATA_DIR / PARQUET_NAME}"
    )
