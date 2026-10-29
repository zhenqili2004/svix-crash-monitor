"""Option-implied probabilities of a crash in the market.

Three probabilities that a gross return R = S_T / S_0 ends at or below a
threshold alpha (alpha = 0.80 is a 20% drop), from one risk-neutral CDF:

- ``risk_neutral_crash_prob``: P*[R <= alpha] = Q(alpha). What the options
  market charges for, not what it expects; it overstates crashes when risk
  aversion is high.
- ``crash_probability``: the probability as perceived by an investor with
  log utility who holds the market, Martin (2017), the series he publishes
  as ``crashprob.xls``. Because that investor's marginal utility is 1/R,
  real-world probabilities are risk-neutral expectations weighted by R:

      P[R <= alpha] = E*[R I(R <= alpha)] / R_f,

  and the numerator is itself a Breeden-Litzenberger expression in the put
  price and its slope at the threshold,

      E*[R I(R <= alpha)] = R_f [ alpha put'(alpha) - put(alpha) ],   put'(alpha) = Q(alpha) / R_f.

- ``crash_probability_power``: the same for power utility with risk
  aversion gamma, equation (7) of Martin and Shi (2025):

      P[R <= alpha] = E*[R^gamma I(R <= alpha)] / E*[R^gamma],

  computed from the option prices by their Result 5. At gamma = 1 it equals
  ``crash_probability``; they calibrate gamma = 2.

HOMEWORK INSTRUCTIONS
---------------------
``crash_probability`` has had its implementation removed. Its docstring
states the formula; the guide notebook ``02_svix_and_crash_probability_ipynb.py``
derives it. Check your work, offline, with::

    pytest -vv ./src/test_crash_prob.py
"""

from utility_correction import GAMMA, market_moment, weighted_tail_expectation


def risk_neutral_crash_prob(cdf, alpha):
    """Return P*[R <= alpha] by evaluating the risk-neutral CDF at alpha.

    Examples
    --------
    >>> import numpy as np
    >>> from rnd import RiskNeutralCDF
    >>> cdf = RiskNeutralCDF(np.array([0.5, 1.0, 1.5]), np.array([0.0, 0.5, 1.0]))
    >>> risk_neutral_crash_prob(cdf, 0.75)
    0.25
    """
    return float(cdf(alpha))


def crash_probability(cdf, alpha):
    """The log investor's probability of R <= alpha, Martin (2017).

    Combine the two displays in the module docstring::

        P[R <= alpha] = alpha Q(alpha) / R_f - put(alpha)

    where ``cdf(alpha)`` is Q(alpha), ``cdf.forward`` is R_f, and
    ``cdf.put(alpha)`` is the discounted put price at moneyness alpha (spot
    normalized to 1).

    Examples
    --------
    Under a flat smile the return is lognormal and the closed form is
    N((ln alpha - sigma^2 tau / 2) / (sigma sqrt(tau))) with the rate
    folded into alpha; see ``test_crash_prob.py``.

    >>> import numpy as np, pandas as pd
    >>> from rnd import risk_neutral_cdf
    >>> moneyness = np.linspace(0.5, 1.5, 9)
    >>> smile = pd.DataFrame({"days_to_maturity": 30, "moneyness": moneyness,
    ...     "implied_vol": 0.30, "cp_flag": np.where(moneyness <= 1.0, "P", "C")})
    >>> cdf = risk_neutral_cdf(smile, rate=0.03)
    >>> round(crash_probability(cdf, 0.90), 4) < round(risk_neutral_crash_prob(cdf, 0.90), 4)
    True
    """
    return float(alpha * cdf(alpha) / cdf.forward - cdf.put(alpha))


def crash_probability_power(cdf, alpha, gamma=GAMMA):
    """The power-utility investor's probability of R <= alpha, Martin and Shi (2025), eq. (7)."""
    return float(weighted_tail_expectation(cdf, alpha, tail="lower", gamma=gamma) / market_moment(cdf, gamma))
