"""Tests for svix.svix_squared and the constant-maturity interpolation."""

import numpy as np
import pandas as pd
import pytest

from rnd import RiskNeutralCDF, priced_grid
from svix import interpolate_variance, svix_index, svix_squared
from test_rnd import RATE, smile_frame


def priced(smile, rate=RATE):
    """The priced grid of a smile, without the Breeden-Litzenberger step:
    SVIX needs only the option prices, so its tests do not depend on rnd.py's
    blank."""
    grid, call_grid, tau = priced_grid(smile, rate)
    return RiskNeutralCDF(grid, np.zeros_like(grid), call_grid=call_grid, maturity_years=tau, rate=rate)


def test_svix_flat_smile():
    """Under a flat smile SVIX^2 = (exp(sigma^2 tau) - 1) / tau."""
    for vol, days in [(0.20, 30), (0.35, 30), (0.20, 91)]:
        tau = days / 365.0
        cdf = priced(smile_frame(np.linspace(0.4, 1.8, 29), vol, days=days))
        expected = (np.exp(vol**2 * tau) - 1.0) / tau
        assert abs(svix_squared(cdf) - expected) < 2e-4 * max(1.0, expected / 0.04), (vol, days)


def test_interpolate_variance_is_exact_at_the_expiries():
    s_near, s_far = 0.04, 0.05
    t_near, t_far = 20 / 365, 50 / 365
    assert abs(interpolate_variance(s_near, t_near, s_far, t_far, t_near) - s_near) < 1e-12
    assert abs(interpolate_variance(s_near, t_near, s_far, t_far, t_far) - s_far) < 1e-12
    mid = interpolate_variance(s_near, t_near, s_far, t_far, 30 / 365)
    assert s_near < mid < s_far


def test_svix_index_units():
    assert svix_index(0.04) == pytest.approx(20.0)


def test_svix_matches_martin():
    """Your SVIX on the OptionMetrics S&P 500 surface against the one-month
    SVIX that Martin (2017) publishes, at each month-end from 1996 to 2012.

    Needs the OptionMetrics pull (`doit single_names`). CME publishes no CVOL
    for the S&P 500, so Martin's own series is the benchmark.
    """
    import calc_svix_vs_martin as calc
    import single_name_bounds as snb
    from pull_optionmetrics import load_security_prices, load_vol_surface, load_zero_curve

    martin = calc.load_martin_svix()
    clean = calc.spx_surface(load_vol_surface(), load_security_prices())
    clean = clean[clean["date"] <= martin["date"].max()]
    month_ends = snb.month_ends(clean["date"])
    ours = calc.svix_from_surface(clean, snb.rates_by_maturity(load_zero_curve()), dates=month_ends)
    merged = calc.compare_with_martin(ours, martin)
    assert len(merged) > 150, "expected about 190 month-ends between 1996 and 2012"
    assert merged["gap"].abs().median() < 0.6, (
        f"median gap to Martin's SVIX is {merged['gap'].abs().median():.2f} vol points"
    )
    assert np.corrcoef(merged["svix"], merged["svix_martin"])[0, 1] > 0.99
