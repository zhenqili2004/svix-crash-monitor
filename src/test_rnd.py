"""Tests for rnd.risk_neutral_cdf.

The correctness check: a FLAT implied-vol smile is exactly the Black-Scholes
assumption, so Breeden-Litzenberger must recover the closed-form lognormal
CDF of the gross return. A skewed smile has no closed form, so it is checked
for the properties a CDF must have.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from rnd import risk_neutral_cdf

RATE = 0.03
DAYS = 30
TAU = DAYS / 365.0


def smile_frame(moneyness, implied_vol, days=DAYS, rate=RATE):
    moneyness = np.asarray(moneyness, dtype=float)
    forward = np.exp(rate * days / 365.0)
    return pd.DataFrame(
        {
            "days_to_maturity": days,
            "moneyness": moneyness,
            "implied_vol": np.broadcast_to(implied_vol, moneyness.shape),
            "cp_flag": np.where(moneyness <= forward, "P", "C"),
        }
    )


def lognormal_cdf(k, vol, tau=TAU, rate=RATE):
    """Closed form P*[R <= k] under geometric Brownian motion."""
    mu = (rate - 0.5 * vol**2) * tau
    return norm.cdf((np.log(k) - mu) / (vol * np.sqrt(tau)))


def test_cdf_flat_smile_is_lognormal():
    cdf = risk_neutral_cdf(smile_frame(np.linspace(0.5, 1.5, 21), 0.25), RATE)
    for k in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10]:
        assert abs(cdf(k) - lognormal_cdf(k, 0.25)) < 2e-3, k


def test_cdf_is_valid():
    """Monotone, in [0, 1], and the smile's skew shows up as a fatter left tail."""
    moneyness = np.linspace(0.6, 1.4, 17)
    skewed = 0.20 + 0.5 * np.clip(1.0 - moneyness, 0, None)  # vol rises below the money
    cdf = risk_neutral_cdf(smile_frame(moneyness, skewed), RATE)
    assert np.all(np.diff(cdf.values) >= 0)
    assert cdf.values.min() >= 0.0 and cdf.values.max() <= 1.0
    flat = risk_neutral_cdf(smile_frame(moneyness, 0.20), RATE)
    assert cdf(0.85) > flat(0.85)
