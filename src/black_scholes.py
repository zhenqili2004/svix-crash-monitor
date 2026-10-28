"""Black-Scholes pricing, greeks, and implied volatility for S&P 500 options.

Purpose
-------
This module provides pure scalar functions to compute Black-Scholes prices,
greeks, and implied volatilities for European options. It exists because
Databento market data ships option *prices only*, unlike WRDS OptionMetrics,
which ships implied volatilities and greeks precomputed. When working with
the E-mini S&P 500 options pulled from Databento's CME Globex feed, we must
therefore back out implied volatility (via root-finding) and compute greeks
ourselves.

The Black-76 identity
---------------------
Our Databento contracts are options on ES futures, and even SPX index
options are naturally priced off the forward rather than the spot (the
S&P 500 pays dividends, and the feed supplies no dividend yield). Rather
than adding a dividend-yield input to every function, this module relies on
the Black-76 identity used throughout the case study:

    Black76(F, K, r, sigma, T) = BlackScholes(S*, K, r, sigma, T)
    where S* = F * exp(-r * T)

That is, feeding the ordinary spot-form Black-Scholes formulas the discounted
underlying S* = F * exp(-r*T) reproduces the Black-76 forward-based price
exactly, with no dividend-yield input needed. In practice F is implied from
put-call parity on liquid near-the-money option pairs
(F = K + exp(r*T) * (C - P)) -- for options on futures, that parity F is
the underlying futures price itself.

Conventions
-----------
- ``sigma`` is annualized volatility (e.g. 0.20 for 20% per year).
- ``T`` is time to expiration in years.
- ``bs_theta`` is per YEAR (divide by 365 for a per-calendar-day theta).
- ``bs_vega`` is per 1.00 of volatility (divide by 100 for per vol point).
- Put delta is negative: ``bs_delta(..., option_type="put")`` = N(d1) - 1.

Edge behavior
-------------
All pricing/greek functions return ``np.nan`` when ``T <= 0`` or
``sigma <= 0`` (guarded at the top of each function, before any math).
``implied_volatility`` returns ``np.nan`` (and never raises) on unpriceable
inputs: expired options, non-positive prices, prices outside the
no-arbitrage bounds, or non-finite inputs.

Imports are intentionally minimal (numpy, scipy.stats.norm,
scipy.optimize.brentq) so the module stays pure and importable anywhere —
no plotly, no IPython, no project settings.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    """Return (d1, d2) for the Black-Scholes formulas.

    d1 = (ln(S/K) + (r + sigma^2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, r, sigma, T, option_type="call"):
    """European Black-Scholes option price.

    Formula
    -------
    d1 = (ln(S/K) + (r + sigma^2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call: S * N(d1) - K * exp(-r*T) * N(d2)
    put:  K * exp(-r*T) * N(-d2) - S * N(-d1)

    To price off a parity-implied forward F (Black-76), pass
    ``S = F * exp(-r*T)``.

    Parameters
    ----------
    S : float
        Spot price (or discounted forward F * exp(-r*T)).
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate (annualized).
    sigma : float
        Annualized volatility.
    T : float
        Time to expiration in years.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Option price. ``np.nan`` if ``T <= 0`` or ``sigma <= 0``.

    Raises
    ------
    ValueError
        If ``option_type`` is not "call" or "put".
    """
    if option_type not in ("call", "put"):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    if T <= 0 or sigma <= 0:
        return np.nan
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(S, K, r, sigma, T, option_type="call"):
    """Black-Scholes delta, the sensitivity of price to the underlying.

    Formula
    -------
    call: N(d1)
    put:  N(d1) - 1   (negative)

    Returns
    -------
    float
        Delta. ``np.nan`` if ``T <= 0`` or ``sigma <= 0``.

    Raises
    ------
    ValueError
        If ``option_type`` is not "call" or "put".
    """
    if option_type not in ("call", "put"):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    if T <= 0 or sigma <= 0:
        return np.nan
    d1, _ = _d1_d2(S, K, r, sigma, T)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def bs_gamma(S, K, r, sigma, T):
    """Black-Scholes gamma, the second derivative of price wrt the underlying.

    Formula
    -------
    gamma = phi(d1) / (S * sigma * sqrt(T))

    where phi is the standard normal pdf. Identical for calls and puts.

    Returns
    -------
    float
        Gamma. ``np.nan`` if ``T <= 0`` or ``sigma <= 0``.
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(S, K, r, sigma, T):
    """Black-Scholes vega, the sensitivity of price to volatility.

    Formula
    -------
    vega = S * phi(d1) * sqrt(T)

    Expressed per 1.00 of volatility (divide by 100 for per vol point).
    Identical for calls and puts.

    Returns
    -------
    float
        Vega. ``np.nan`` if ``T <= 0`` or ``sigma <= 0``.
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return float(S * norm.pdf(d1) * np.sqrt(T))


def bs_theta(S, K, r, sigma, T, option_type="call"):
    """Black-Scholes theta, the sensitivity of price to the passage of time.

    Expressed per YEAR (divide by 365 for per calendar day). Theta is the
    negative of the derivative of price with respect to time to expiration:
    theta = -dV/dT.

    Formula
    -------
    call: -S * phi(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r*T) * N(d2)
    put:  -S * phi(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r*T) * N(-d2)

    Returns
    -------
    float
        Theta per year. ``np.nan`` if ``T <= 0`` or ``sigma <= 0``.

    Raises
    ------
    ValueError
        If ``option_type`` is not "call" or "put".
    """
    if option_type not in ("call", "put"):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    if T <= 0 or sigma <= 0:
        return np.nan
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    time_decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return float(time_decay - r * K * np.exp(-r * T) * norm.cdf(d2))
    return float(time_decay + r * K * np.exp(-r * T) * norm.cdf(-d2))


def implied_volatility(price, S, K, r, T, option_type="call", lo=1e-4, hi=5.0):
    """Black-Scholes implied volatility via Brent's root-finding method.

    Solves f(sigma) = bs_price(S, K, r, sigma, T, option_type) - price = 0
    for sigma on the bracket [lo, hi] using ``scipy.optimize.brentq``.

    Guards (returns ``np.nan``, NEVER raises)
    -----------------------------------------
    - Non-finite inputs (price, S, K, r, T).
    - ``T <= 0`` (expired option).
    - ``price <= 0``.
    - Price outside the no-arbitrage bounds, detected as f(lo) and f(hi)
      having the same sign (the bracket does not straddle a root, so brentq
      is never called): e.g. a deep-ITM call priced below its intrinsic
      lower bound S - K*exp(-r*T), or any price above the upper bound.

    Parameters
    ----------
    price : float
        Observed option price.
    S : float
        Spot price (or discounted forward F * exp(-r*T)).
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate (annualized).
    T : float
        Time to expiration in years.
    option_type : str
        "call" or "put".
    lo, hi : float
        Volatility bracket for the root search.

    Returns
    -------
    float
        Annualized implied volatility, or ``np.nan`` if unrecoverable.
    """
    if not all(np.isfinite(x) for x in (price, S, K, r, T)):
        return np.nan
    if T <= 0 or price <= 0:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, r, sigma, T, option_type=option_type) - price

    f_lo = objective(lo)
    f_hi = objective(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return np.nan
    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan

    return float(brentq(objective, lo, hi))
