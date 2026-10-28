"""Tests for the three crash probabilities, on synthetic smiles."""

import numpy as np
from scipy.stats import norm

from crash_prob import crash_probability, crash_probability_power, risk_neutral_crash_prob
from rnd import risk_neutral_cdf
from test_rnd import RATE, TAU, smile_frame


def lognormal_log_investor_prob(alpha, vol, tau=TAU, rate=RATE):
    """Closed form of E*[R I(R <= alpha)] / R_f under geometric Brownian motion.

    With R / R_f lognormal with mean 1 and log-variance s^2 = vol^2 tau,
    E*[(R / R_f) I(R / R_f <= a)] = N((ln a - s^2 / 2) / s) at a = alpha / R_f.
    """
    s = vol * np.sqrt(tau)
    a = alpha / np.exp(rate * tau)
    return norm.cdf((np.log(a) - 0.5 * s**2) / s)


def test_bound_flat_smile():
    cdf = risk_neutral_cdf(smile_frame(np.linspace(0.4, 1.8, 29), 0.30), RATE)
    for alpha in [0.80, 0.90, 0.95]:
        assert abs(crash_probability(cdf, alpha) - lognormal_log_investor_prob(alpha, 0.30)) < 2e-3, alpha


def test_log_investor_is_below_risk_neutral():
    cdf = risk_neutral_cdf(smile_frame(np.linspace(0.4, 1.8, 29), 0.30), RATE)
    for alpha in [0.80, 0.90]:
        p_star = risk_neutral_crash_prob(cdf, alpha)
        assert 0 < crash_probability(cdf, alpha) < p_star


def test_power_utility_at_gamma_one_is_the_log_investor():
    cdf = risk_neutral_cdf(smile_frame(np.linspace(0.4, 1.8, 29), 0.30), RATE)
    for alpha in [0.80, 0.90]:
        assert abs(crash_probability_power(cdf, alpha, gamma=1) - crash_probability(cdf, alpha)) < 1e-6
        assert crash_probability_power(cdf, alpha, gamma=2) < crash_probability(cdf, alpha)
