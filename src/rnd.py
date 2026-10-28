"""The risk-neutral distribution of the return, by Breeden and Litzenberger.

Builds the risk-neutral CDF of a gross return R = S_T / S_0 from an
implied-volatility smile: keep the out-of-the-money side of each strike,
interpolate the smile onto a fine grid of moneyness k = K / S_0, price a
call at every grid point with Black-Scholes, differentiate the call curve in
the strike, and fit an isotonic regression so the result is a valid CDF.

Units. Everything is normalized by the spot: a strike and a gross-return
level are the same number k, option prices are fractions of the spot, and
the forward is R_f = exp(r * tau). For options on a future, pass the
discounted forward S* = F exp(-r * tau) as the spot; that is the Black-76
identity the ``black_scholes`` module explains.

This is the engine of Riley Haas and Marija Jovicic's replication of Martin
and Shi (2025), ``forecasting_crashes_smile`` (FINM 32800, Summer 2026), used
with their permission. The isotonic step uses SciPy instead of scikit-learn.

HOMEWORK INSTRUCTIONS
---------------------
One step of ``risk_neutral_cdf`` has been removed: the Breeden-Litzenberger
formula that turns the call curve into the raw CDF. Its docstring and the
guide notebook ``02_svix_and_crash_probability_ipynb.py`` state the formula.
Check your work, offline, with::

    pytest -vv ./src/test_rnd.py
"""

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import norm

N_GRID = 2000  # steps in the fine moneyness grid, per Martin and Shi, Appendix D


def _grid_half_width(days_to_maturity):
    """Grid spans k in [1/L, L]. Appendix D: L=3 up to 6 months, L=5 at 12."""
    return 5.0 if days_to_maturity >= 365 else 3.0


class RiskNeutralCDF:
    """A monotone risk-neutral CDF Q(.) of the gross return R = S_T / S_0.

    Stored on a fixed grid of gross-return levels with matching CDF values.
    Call it like a function to evaluate Q(k); use ``inverse`` for Q^-1(p).
    ``call`` and ``put`` evaluate the (discounted, spot-normalized) option
    price curves the CDF was built from; ``maturity_years`` and ``rate``
    are what the price-based moments in ``utility_correction`` need.
    """

    def __init__(self, grid, values, call_grid=None, maturity_years=None, rate=None):
        self.grid = np.asarray(grid, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.call_grid = None if call_grid is None else np.asarray(call_grid, dtype=float)
        self.maturity_years = maturity_years
        self.rate = rate

    def __call__(self, k):
        return np.interp(k, self.grid, self.values)

    def inverse(self, p):
        """Q^-1(p): the gross-return level where the CDF equals p."""
        values, first_idx = np.unique(self.values, return_index=True)
        grid = self.grid[first_idx]
        return np.interp(p, values, grid)

    @property
    def forward(self):
        """R_f = exp(r * tau), the forward with the spot normalized to 1."""
        return float(np.exp(self.rate * self.maturity_years))

    @property
    def put_grid(self):
        """Put prices on the grid, from the calls by put-call parity."""
        return self.call_grid - 1.0 + self.grid / self.forward

    def call(self, k):
        return np.interp(k, self.grid, self.call_grid)

    def put(self, k):
        return np.interp(k, self.grid, self.put_grid)


def _black_scholes_call(moneyness, vol, maturity_years, rate):
    """Black-Scholes call price with the spot normalized to 1."""
    k = np.asarray(moneyness, dtype=float)
    sqrt_t = np.sqrt(maturity_years)
    d1 = (-np.log(k) + (rate + 0.5 * vol**2) * maturity_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return norm.cdf(d1) - k * np.exp(-rate * maturity_years) * norm.cdf(d2)


def otm_smile(surface_slice, rate):
    """Keep the out-of-the-money side of each strike, sorted by moneyness.

    Puts below the forward and calls above it. A surface can quote both a put
    and a call at the same moneyness with different vols, so without this the
    smile is not even single-valued.
    """
    days = int(surface_slice["days_to_maturity"].iloc[0])
    forward = np.exp(rate * days / 365.0)
    is_otm = ((surface_slice["cp_flag"] == "P") & (surface_slice["moneyness"] <= forward)) | (
        (surface_slice["cp_flag"] == "C") & (surface_slice["moneyness"] > forward)
    )
    return surface_slice.loc[is_otm].sort_values("moneyness")


def priced_grid(surface_slice, rate, n_grid=N_GRID):
    """Interpolate the smile onto the fine grid and price calls on it.

    Returns (grid, call_grid, maturity_years). Linear interpolation of the
    implied vol between observed strikes, flat outside the observed range.
    """
    days = int(surface_slice["days_to_maturity"].iloc[0])
    maturity_years = days / 365.0
    slice_ = otm_smile(surface_slice, rate)
    moneyness = slice_["moneyness"].to_numpy(dtype=float)
    implied_vol = slice_["implied_vol"].to_numpy(dtype=float)
    if len(moneyness) < 2:
        raise ValueError("need at least two out-of-the-money quotes to build a smile")
    L = _grid_half_width(days)
    grid = np.linspace(1.0 / L, L, n_grid)
    vol_grid = np.interp(grid, moneyness, implied_vol)
    call_grid = _black_scholes_call(grid, vol_grid, maturity_years, rate)
    return grid, call_grid, maturity_years


def risk_neutral_cdf(surface_slice, rate, n_grid=N_GRID):
    """Return the risk-neutral CDF Q(.) for one date, one name, one maturity.

    Parameters
    ----------
    surface_slice : DataFrame
        One smile: columns days_to_maturity, moneyness (K / S_0), implied_vol
        (decimal), cp_flag ('P' or 'C'); both sides may be present.
    rate : float
        Continuously compounded risk-free rate for this maturity, as a
        decimal (0.03 = 3%).

    Returns a RiskNeutralCDF: monotone, non-decreasing, in [0, 1].

    The Breeden-Litzenberger step. A call struck at k pays (R - k)^+, so its
    discounted price is c(k) = E*[(R - k)^+] / R_f and its derivative in the
    strike is dc/dk = -P*(R > k) / R_f. Rearranging gives the risk-neutral
    CDF on the grid::

        Q(k) = 1 + R_f * dc/dk,     R_f = exp(rate * tau)

    (``np.gradient(call_grid, grid)`` differentiates the call curve.) Then
    the isotonic regression below forces the raw Q to be non-decreasing
    and clips it to [0, 1].

    Examples
    --------
    A flat 25% vol smile over 30 days: near-zero probability of a 20% drop,
    roughly a coin flip of ending below today's price.

    >>> import numpy as np
    >>> import pandas as pd
    >>> moneyness = np.linspace(0.5, 1.5, 9)
    >>> surface = pd.DataFrame({
    ...     "days_to_maturity": 30,
    ...     "moneyness": moneyness,
    ...     "implied_vol": 0.25,
    ...     "cp_flag": np.where(moneyness <= 1.0, "P", "C"),
    ... })
    >>> cdf = risk_neutral_cdf(surface, rate=0.03)
    >>> round(float(cdf(1.00)), 4)
    0.5006
    >>> round(float(cdf(0.80)), 4)
    0.0009
    """
    if abs(rate) >= 1.0:
        raise ValueError(
            f"rate must be a decimal fraction (e.g. 0.03 for 3%), got {rate!r}"
        )
    grid, call_grid, maturity_years = priced_grid(surface_slice, rate, n_grid)

    ## TODO: Breeden-Litzenberger. Compute `q_raw`, the raw risk-neutral CDF on
    ## the grid, from the slope of the call curve in the strike. The docstring
    ## above states the formula; `np.gradient(call_grid, grid)` is the slope.
    q_raw = None
    if q_raw is None:
        raise NotImplementedError(
            "Compute q_raw in risk_neutral_cdf (see HOMEWORK INSTRUCTIONS at the top of this file)"
        )

    # A valid CDF is non-decreasing and lives in [0, 1]; the raw one need not be.
    q_fit = np.clip(isotonic_regression(q_raw, increasing=True).x, 0.0, 1.0)

    return RiskNeutralCDF(
        grid, q_fit, call_grid=call_grid, maturity_years=maturity_years, rate=rate
    )
