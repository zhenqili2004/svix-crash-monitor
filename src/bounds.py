"""Frechet-Hoeffding bounds on the probability that a single stock crashes.

Martin and Shi (2025), Result 3: with the crash threshold q fixed,

    q_l = Q_m^{-1}(Q_i(q)),   q_u = Q_m^{-1}(1 - Q_i(q)),
    P^L = E*[R_m^gamma I(R_m <= q_l)] / E*[R_m^gamma],
    P^U = E*[R_m^gamma I(R_m >= q_u)] / E*[R_m^gamma],

and the risk-neutral probability P* = Q_i(q) always lies between them. The
lower bound is the paper's forecaster.

Written by Riley Haas and Marija Jovicic (``forecasting_crashes_smile``, FINM
32800, Summer 2026), used with their permission.
"""

from utility_correction import GAMMA, market_moment, weighted_tail_expectation


def crash_bounds(cdf_i, cdf_m, threshold_q, gamma=GAMMA):
    """Return (bound_lower, prob_riskneutral, bound_upper) for one observation.

    Guaranteed to satisfy bound_lower <= prob_riskneutral <= bound_upper
    (Result 3).

    Examples
    --------
    A calmer market (20% vol) and a riskier name (35% vol), both 30 days out:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from rnd import risk_neutral_cdf
    >>> moneyness = np.linspace(0.5, 1.5, 9)
    >>> def _smile(vol):
    ...     return pd.DataFrame({
    ...         "days_to_maturity": 30, "moneyness": moneyness, "implied_vol": vol,
    ...         "cp_flag": np.where(moneyness <= 1.0, "P", "C"),
    ...     })
    >>> cdf_m = risk_neutral_cdf(_smile(0.20), rate=0.03)
    >>> cdf_i = risk_neutral_cdf(_smile(0.35), rate=0.03)
    >>> lower, p_star, upper = crash_bounds(cdf_i, cdf_m, 0.80)
    >>> lower <= p_star <= upper
    True
    >>> round(lower, 4), round(p_star, 4), round(upper, 4)
    (0.0104, 0.014, 0.0186)
    """
    prob_riskneutral = float(cdf_i(threshold_q))

    q_l = float(cdf_m.inverse(prob_riskneutral))
    q_u = float(cdf_m.inverse(1.0 - prob_riskneutral))

    denom = market_moment(cdf_m, gamma)
    bound_lower = weighted_tail_expectation(cdf_m, q_l, tail="lower", gamma=gamma) / denom
    bound_upper = weighted_tail_expectation(cdf_m, q_u, tail="upper", gamma=gamma) / denom

    return float(bound_lower), prob_riskneutral, float(bound_upper)
