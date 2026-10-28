"""Tests for the monitor's history bookkeeping, offline."""

import pandas as pd

import monitor


def test_append_history_dedups_and_sorts(tmp_path):
    path = tmp_path / "history.csv"
    first = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-14"]), "svix_30": [16.0, 15.5],
                          "near_expiry": pd.Timestamp("2026-10-30"), "far_expiry": pd.Timestamp("2026-11-30")})
    out = monitor.append_history(first, path)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-10-14", "2026-10-15"]
    again = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"]), "svix_30": [16.1, 17.0],
                          "near_expiry": pd.Timestamp("2026-10-30"), "far_expiry": pd.Timestamp("2026-11-30")})
    out = monitor.append_history(again, path)
    assert len(out) == 3
    assert out.loc[out["date"] == pd.Timestamp("2026-10-15"), "svix_30"].iloc[0] == 16.1  # the later row wins
    reloaded = monitor.load_history(path)
    assert len(reloaded) == 3 and reloaded["date"].is_monotonic_increasing


def test_fill_missing_vix():
    history = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-21"]), "vix": [17.31, float("nan")]})
    fred = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-21"]), "vix": [17.31, 16.68]})
    filled, n = monitor.fill_missing_vix(history, fred)
    assert n == 1 and filled["vix"].tolist() == [17.31, 16.68]
    still_missing, n = monitor.fill_missing_vix(history, fred.iloc[:1])
    assert n == 0 and still_missing["vix"].isna().sum() == 1
