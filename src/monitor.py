"""The daily monitor: SVIX and the probability of a crash, one row per session.

``compute_session`` turns one session's settlement prices into one row:
SVIX at a constant 30-day horizon, interpolated between the two end-of-month
expirations that bracket it, and the probabilities of a 20% and a 10% drop
over the expiration closest to one month. ``compute_history`` does that for
every session in the settlement table, and ``append_history`` merges the new
rows into the history CSV, ``_output/svix_monitor_history.csv``, one row per
session, oldest first.

The GitHub Actions workflow runs ``doit latest`` every morning: restore the
history from the ``monitor-history`` branch, pull the latest session, append
its row if the CSV does not have it, rebuild the page, record the CSV on
that branch again, publish the page. So the chart on the page grows by one
point a day without anyone touching it, and ``main`` holds only source code. The default
``doit`` runs the same computation over the window in the settings, which is
how the history is filled in.

History columns
---------------
date; near_expiry, far_expiry, days_near, days_far; forward_near (the
parity-implied price of the underlying future); rate (DGS1MO, decimal);
svix2_near, svix2_far (annualized); svix2_30, svix_30 (the 30-day index in
VIX units); days_used (the expiry the probabilities are read from);
p_star_20, p_log_20, p_power_20 (risk-neutral, log-utility, and
power-utility probabilities of a 20% drop); p_star_10, p_log_10; vix (the
Cboe VIX close from FRED, when available).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import crash_prob
import smile
import svix
from rnd import risk_neutral_cdf
from settings import config

OUTPUT_DIR = config("OUTPUT_DIR")
HISTORY_CSV = Path(OUTPUT_DIR) / "svix_monitor_history.csv"
LATEST_CSV = Path(OUTPUT_DIR) / "svix_monitor_latest.csv"

THRESHOLDS = {"20": 0.80, "10": 0.90}


def compute_session(session, rate):
    """One row of the monitor from one session's settlement table.

    ``session`` holds every settled option of one date (all expirations);
    ``rate`` is the risk-free rate for that date as a decimal.
    """
    date = pd.Timestamp(session["date"].iloc[0])
    near, far = smile.bracketing_expiries(session)
    cdfs = {}
    for expiry in (near, far):
        cdfs[expiry] = risk_neutral_cdf(smile.smile_for(session, date, expiry, rate), rate)
    s2 = {e: svix.svix_squared(c) for e, c in cdfs.items()}
    tau = {e: c.maturity_years for e, c in cdfs.items()}
    svix2_30 = svix.interpolate_variance(s2[near], tau[near], s2[far], tau[far])

    closest = smile.closest_expiry(session)
    cdf = cdfs.get(closest) or risk_neutral_cdf(smile.smile_for(session, date, closest, rate), rate)
    row = {
        "date": date,
        "near_expiry": pd.Timestamp(near),
        "far_expiry": pd.Timestamp(far),
        "days_near": int(round(tau[near] * 365)),
        "days_far": int(round(tau[far] * 365)),
        "forward_near": float(smile.smile_for(session, date, near, rate)["forward"].iloc[0]),
        "rate": float(rate),
        "svix2_near": s2[near],
        "svix2_far": s2[far],
        "svix2_30": svix2_30,
        "svix_30": svix.svix_index(svix2_30),
        "days_used": int(round(cdf.maturity_years * 365)),
    }
    for label, alpha in THRESHOLDS.items():
        row[f"p_star_{label}"] = crash_prob.risk_neutral_crash_prob(cdf, alpha)
        row[f"p_log_{label}"] = crash_prob.crash_probability(cdf, alpha)
    row["p_power_20"] = crash_prob.crash_probability_power(cdf, THRESHOLDS["20"])
    return row


def compute_history(options, fred, dates=None):
    """Monitor rows for every session in ``options`` (or only ``dates``)."""
    from pull_fred import rate_on

    rows = []
    for date, session in options.groupby("date"):
        if dates is not None and pd.Timestamp(date) not in set(pd.to_datetime(dates)):
            continue
        try:
            rate = rate_on(fred, date)
            rows.append(compute_session(session, rate))
        except ValueError as error:
            print(f"  {pd.Timestamp(date).date()}: skipped ({error})")
    history = pd.DataFrame(rows)
    if history.empty:
        return history
    vix = fred.dropna(subset=["vix"])[["date", "vix"]]
    return history.merge(vix, on="date", how="left").sort_values("date").reset_index(drop=True)


def load_history(path=HISTORY_CSV):
    """The history so far, or an empty frame if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date", "near_expiry", "far_expiry"])
    return df.sort_values("date").reset_index(drop=True)


def append_history(new_rows, path=HISTORY_CSV):
    """Merge new rows into the history CSV, one row per session, oldest first.

    A session already in the file is replaced by its new row (a late
    correction wins), so running the monitor twice on the same day is
    harmless.
    """
    history = load_history(path)
    merged = pd.concat([history, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    merged = merged.reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, float_format="%.8g", date_format="%Y-%m-%d")
    return merged


def fill_missing_vix(history, fred):
    """Fill a blank VIX in the history from FRED, where FRED now has it.

    FRED posts the VIX close the next morning, sometimes later than the
    monitor runs, so a session can enter the history before its VIX does.
    Each run fills such gaps. Returns the history and the number filled.
    """
    if history.empty or "vix" not in history:
        return history, 0
    missing = history["vix"].isna()
    if not missing.any():
        return history, 0
    vix = fred.dropna(subset=["vix"]).set_index("date")["vix"]
    filled = history.loc[missing, "date"].map(vix)
    history = history.copy()
    history.loc[missing, "vix"] = filled
    return history, int(filled.notna().sum())


def write_history(history, path=HISTORY_CSV):
    history.to_csv(path, index=False, float_format="%.8g", date_format="%Y-%m-%d")


if __name__ == "__main__":
    from pull_es_options import load_es_options
    from pull_fred import load_fred

    options = load_es_options()
    fred = load_fred()
    known = set(load_history()["date"]) if HISTORY_CSV.exists() else set()
    todo = sorted(d for d in options["date"].unique() if pd.Timestamp(d) not in known)
    print(f"Sessions in the pull: {options['date'].nunique()}; new to the history: {len(todo)}")
    new_rows = compute_history(options, fred, dates=todo) if todo else pd.DataFrame()
    history = append_history(new_rows) if not new_rows.empty else load_history()
    history, n_filled = fill_missing_vix(history, fred)
    if n_filled:
        write_history(history)
        print(f"Filled the VIX for {n_filled} session(s) that entered the history before FRED posted it")
    latest = history.tail(1)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    latest.to_csv(LATEST_CSV, index=False)
    if not latest.empty:
        r = latest.iloc[0]
        vix = "not on FRED yet" if pd.isna(r["vix"]) else f"{r['vix']:.2f}"
        print(
            f"Latest session {r['date'].date()}: SVIX30 = {r['svix_30']:.2f}, "
            f"VIX = {vix}, P*(20% drop, {int(r['days_used'])}d) = {100 * r['p_star_20']:.2f}%, "
            f"log-investor = {100 * r['p_log_20']:.2f}%, gamma=2: {100 * r['p_power_20']:.2f}%"
        )
    print(f"History: {len(history)} session(s) in {HISTORY_CSV}")
