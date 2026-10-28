"""Tests for the OptionMetrics SQL in pull_optionmetrics.py.

These tests do NOT connect to WRDS. They run the module's query against a
small in-memory SQLite database with the layout of the WRDS tables, so they
check that the WHERE clause selects exactly the rows the single-name part
needs: the chosen securities, the chosen maturities, the date window.
"""

import sqlite3

import pandas as pd

import pull_optionmetrics as pom

SURFACE_COLUMNS = ["secid", "date", "days", "delta", "impl_volatility", "impl_strike", "impl_premium", "dispersion", "cp_flag"]


def _connection():
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS optionm_all")
    conn.execute(f"CREATE TABLE optionm_all.vsurfd2024 ({', '.join(SURFACE_COLUMNS)})")
    return conn


def _row(secid, date, days, delta=-50.0, cp_flag="P"):
    return (secid, date, days, delta, 0.20, 100.0, 5.0, 0.01, cp_flag)


def test_query_selects_secids():
    conn = _connection()
    conn.executemany(
        f"INSERT INTO optionm_all.vsurfd2024 VALUES ({', '.join('?' * len(SURFACE_COLUMNS))})",
        [
            _row(108105, "2024-06-28", 30),  # SPX, keep
            _row(101594, "2024-06-28", 30),  # AAPL, keep
            _row(101397, "2024-06-28", 30),  # AIG, keep
            _row(102265, "2024-06-28", 30),  # another name, drop
        ],
    )
    query = pom.build_surface_query(2024, start_date="2024-01-01", end_date="2024-12-31")
    result = pd.read_sql_query(query, conn)
    conn.close()
    assert sorted(result["secid"].astype(int)) == sorted(pom.SECIDS.values())
    assert set(result.columns) == {"secid", "date", "days", "cp_flag", "delta", "impl_volatility", "impl_strike", "dispersion"}


def test_query_selects_maturities_and_dates():
    conn = _connection()
    conn.executemany(
        f"INSERT INTO optionm_all.vsurfd2024 VALUES ({', '.join('?' * len(SURFACE_COLUMNS))})",
        [
            _row(108105, "2024-06-28", 30),  # keep
            _row(108105, "2024-06-28", 91),  # keep
            _row(108105, "2024-06-28", 60),  # wrong maturity
            _row(108105, "2024-06-28", 182),  # wrong maturity
            _row(108105, "2023-12-29", 30),  # before the window
            _row(108105, "2024-07-01", 30, delta=50.0, cp_flag="C"),  # after the window
            _row(108105, "2024-01-01", 30),  # keep (boundary)
            _row(108105, "2024-06-30", 91),  # keep (boundary)
        ],
    )
    query = pom.build_surface_query(2024, secids=[108105], maturities=[30, 91], start_date="2024-01-01", end_date="2024-06-30")
    result = pd.read_sql_query(query, conn)
    conn.close()
    assert sorted(result["days"].astype(int)) == [30, 30, 91, 91]
    assert result["date"].min() == "2024-01-01" and result["date"].max() == "2024-06-30"
