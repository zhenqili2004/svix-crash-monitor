"""SVIX: the simple variance swap rate, Martin (2017).

Martin defines SVIX^2 as the risk-neutral variance of the market's gross
return, normalized by the risk-free rate and annualized,

    SVIX^2 = var*(R_T / R_f) / tau,

and shows (his equation 12) that it is an integral of option prices:

    SVIX^2 = 2 / (tau R_f S^2) * [ integral_0^F put(K) dK + integral_F^inf call(K) dK ],

out-of-the-money puts below the forward F = R_f S and out-of-the-money calls
above it. Unlike the VIX, which weights each strike by 1/K^2, SVIX weights
every strike equally. CME computes the same simple-variance construction
every day as its CVOL indexes, for options on its interest-rate, currency,
energy, metals, and agricultural futures; it publishes none for the S&P 500.

The integral is evaluated on the fine grid that ``rnd.risk_neutral_cdf``
prices, with the spot normalized to 1 (so S = 1, F = R_f, and option prices
are fractions of the spot). Andrew Heekin and Chandler Bird's replication of
Martin (2017), ``p06_martin_2025`` (FINM 32800, Summer 2026), computes the
same integral from raw SPX quotes.

HOMEWORK INSTRUCTIONS
---------------------
``svix_squared`` has had its implementation removed. Its docstring states
the formula; the guide notebook ``02_svix_and_crash_probability_ipynb.py``
derives it. Check your work, offline, with::

    pytest -vv ./src/test_svix.py
"""

import numpy as np

HORIZON_DAYS = 30


def svix_squared(cdf):
    """SVIX^2 from the priced grid of a ``RiskNeutralCDF``, annualized.

    With the spot normalized to 1 the formula reads::

        SVIX^2 = 2 / (tau R_f) * [ integral_{k <= R_f} put(k) dk + integral_{k > R_f} call(k) dk ]

    where ``cdf.grid`` is the moneyness grid k, ``cdf.put_grid`` and
    ``cdf.call_grid`` are the discounted option prices on it,
    ``cdf.forward`` is R_f, and ``cdf.maturity_years`` is tau. Use
    ``np.trapezoid`` for the integral.

    Examples
    --------
    Under a flat smile the return is lognormal and var*(R / R_f) is
    exp(sigma^2 tau) - 1, so SVIX^2 is (exp(sigma^2 tau) - 1) / tau, a
    little above sigma^2:

    >>> import numpy as np, pandas as pd
    >>> from rnd import risk_neutral_cdf
    >>> moneyness = np.linspace(0.5, 1.5, 9)
    >>> smile = pd.DataFrame({"days_to_maturity": 30, "moneyness": moneyness,
    ...     "implied_vol": 0.20, "cp_flag": np.where(moneyness <= 1.0, "P", "C")})
    >>> round(svix_squared(risk_neutral_cdf(smile, rate=0.03)), 4)
    0.0401
    """
    k = cdf.grid
    R_f = cdf.forward
    tau = cdf.maturity_years
    # OTM price at every grid point: put below the forward, call above it.
    # put(R_f) == call(R_f) by parity, so no interval is dropped at the splice.
    otm = np.where(k <= R_f, cdf.put_grid, cdf.call_grid)
    return float(2.0 / (tau * R_f) * np.trapezoid(otm, k))


def interpolate_variance(svix2_near, tau_near, svix2_far, tau_far, horizon_years=HORIZON_DAYS / 365.0):
    """Constant-maturity SVIX^2 from two bracketing expirations.

    Total variance (SVIX^2 times tau) is interpolated linearly in time to
    expiry, the convention the VIX uses, and divided by the horizon. If the
    horizon lies outside [tau_near, tau_far] this extrapolates, as the VIX
    does.
    """
    if tau_far == tau_near:
        raise ValueError("the two expirations must differ")
    w = (tau_far - horizon_years) / (tau_far - tau_near)
    total_variance = w * svix2_near * tau_near + (1.0 - w) * svix2_far * tau_far
    return float(total_variance / horizon_years)


def svix_index(svix2):
    """SVIX in the units of the VIX: annualized volatility in percent."""
    return float(100.0 * np.sqrt(svix2))
