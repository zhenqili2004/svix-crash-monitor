"""Tests for the Databento pull: the queries and the settlement extraction, offline."""

import pandas as pd

import pull_es_options as peo


def test_build_queries():
    q = peo.build_queries("2026-10-15", "2026-10-16")
    assert set(q) == {"option_statistics", "option_definitions", "futures_statistics"}
    for query in q.values():
        assert query["dataset"] == "GLBX.MDP3" and query["stype_in"] == "parent"
        assert query["start"] == "2026-10-15" and query["end"] == "2026-10-16"
    assert q["option_statistics"]["schema"] == "statistics"
    assert q["option_definitions"]["schema"] == "definition"
    assert q["option_statistics"]["symbols"] == ["EW.OPT"]
    assert q["futures_statistics"]["symbols"] == ["ES.FUT"]


def test_final_settlements_keeps_the_last_per_contract():
    t0 = pd.Timestamp("2026-10-15 20:17", tz="UTC")
    stats = pd.DataFrame(
        {
            "ts_event": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=3), t0],
            "ts_ref": [pd.Timestamp("2026-10-15", tz="UTC")] * 4,
            "instrument_id": [1, 1, 1, 2],
            "symbol": ["EWV6 P6700"] * 3 + ["EWV6 C6700"],
            "price": [120.0, 120.0, 121.0, 80.0],
            "stat_type": [3, 3, 3, 9],  # the last row is open interest, not a settlement
        }
    )
    out = peo.final_settlements(stats)
    assert list(out.columns) == ["date", "instrument_id", "symbol", "settle", "ts_event"]
    assert len(out) == 1 and out["settle"].iloc[0] == 121.0
    assert out["date"].iloc[0] == pd.Timestamp("2026-10-15")


def test_option_contracts_drops_spreads():
    defs = pd.DataFrame(
        {
            "instrument_id": [1, 2, 3],
            "symbol": ["EWV6 C7920", "EWV6 P7920", "UD:1V: VT 2500195"],
            "instrument_class": ["C", "P", "T"],
            "expiration": [pd.Timestamp("2026-10-30 20:00", tz="UTC")] * 3,
            "strike_price": [7920.0, 7920.0, float("nan")],
            "underlying": ["ESZ6", "ESZ6", ""],
        }
    )
    out = peo.option_contracts(defs)
    assert sorted(out["cp_flag"]) == ["C", "P"]
    assert out["expiration"].iloc[0] == pd.Timestamp("2026-10-30")
    assert "strike" in out.columns


def test_which_databento_errors_are_retried():
    import databento as db
    import requests

    err = db.common.error
    assert peo.is_transient_databento_error(err.BentoServerError(504, message="The remote gateway timed out."))
    assert peo.is_transient_databento_error(err.BentoError("Error streaming response: Response ended prematurely"))
    assert peo.is_transient_databento_error(requests.exceptions.ConnectionError("connection reset"))
    assert not peo.is_transient_databento_error(err.BentoClientError(401, message="auth_authentication_failed"))
    assert not peo.is_transient_databento_error(KeyError("a bug"))


def test_latest_window_reaches_the_previous_session():
    from pull_es_options import latest_window, sessions_between

    # The Tuesday after a Monday holiday still has Friday's session in its window.
    start, end = latest_window("2026-10-27")
    assert (start, end) == ("2026-10-23", "2026-10-27")
    assert sessions_between(start, end) == ["2026-10-23", "2026-10-26"]
    # A Monday run reaches back to Thursday and Friday.
    assert sessions_between(*latest_window("2026-10-26")) == ["2026-10-22", "2026-10-23"]
