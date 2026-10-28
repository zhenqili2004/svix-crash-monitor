"""Power-utility ("fear") correction: moments of R_m^gamma under Q.

Converts risk-neutral expectations into the beliefs of an investor with
power utility who holds the market. For gamma = 1 (log utility) this is
Martin (2017); for gamma = 2 it is the calibration Martin and Shi (2025) use
throughout. The R_m^gamma weighting pulls probability mass out of the crash
tail, since R_m is small in a crash and R_m^gamma smaller still.

Follows Result 5 of Martin and Shi (Appendix D) directly: these moments come
from the index OPTION PRICES on the fine grid, by the midpoint rule, not from
re-integrating the fitted CDF. Only out-of-the-money prices matter (puts
below the forward, calls above it). Since put and call prices built from the
same vol satisfy put-call parity exactly, the put curve comes from the call
curve by parity instead of being priced separately.

Result 5, with the spot normalized to 1 (K becomes moneyness k, F becomes R_f):

    E*[R_m^gamma] = R_f^gamma + R_f * [
        integral_0^R_f   gamma(gamma-1) k^(gamma-2) put(k)  dk +
        integral_R_f^inf gamma(gamma-1) k^(gamma-2) call(k) dk
    ]

    E*[R_m^gamma I(R_m<=q_l)] = R_f q_l^gamma [put'(q_l) - gamma put(q_l)/q_l]
        + R_f * integral_0^q_l gamma(gamma-1) k^(gamma-2) put(k) dk

    E*[R_m^gamma I(R_m>=q_u)] = R_f q_u^gamma [gamma call(q_u)/q_u - call'(q_u)]
        + R_f * integral_q_u^inf gamma(gamma-1) k^(gamma-2) call(k) dk

Written by Riley Haas and Marija Jovicic (``forecasting_crashes_smile``, FINM
32800, Summer 2026), used with their permission.
"""

import numpy as np

GAMMA = 2  # Martin and Shi's calibration (their Appendix C)


def _put_call_curves(index_cdf):
    """Call and put price curves on index_cdf.grid, via put-call parity."""
    return index_cdf.call_grid, index_cdf.put_grid


def _midpoint_integral(grid, integrand, mask):
    """Midpoint-rule sum of integrand over grid points where mask is True."""
    dk = grid[1] - grid[0]
    return float(integrand[mask].sum() * dk)


def market_moment(index_cdf, gamma=GAMMA):
    """Return E*[R_m^gamma], the denominator of the correction (Result 5)."""
    grid = index_cdf.grid
    call_grid, put_grid = _put_call_curves(index_cdf)
    forward = index_cdf.forward

    kernel = gamma * (gamma - 1) * grid ** (gamma - 2)
    integrand = np.where(grid <= forward, kernel * put_grid, kernel * call_grid)
    integral = _midpoint_integral(grid, integrand, np.ones_like(grid, dtype=bool))

    return forward**gamma + forward * integral


def weighted_tail_expectation(index_cdf, k_level, tail, gamma=GAMMA):
    """Return E*[R_m^gamma I(R_m <= k)] (lower) or E*[R_m^gamma I(R_m >= k)] (upper)."""
    grid = index_cdf.grid
    call_grid, put_grid = _put_call_curves(index_cdf)
    forward = index_cdf.forward
    kernel = gamma * (gamma - 1) * grid ** (gamma - 2)

    if tail == "lower":
        put_at_k = float(np.interp(k_level, grid, put_grid))
        # The boundary derivative put'(k) is the marginal: Appendix D defines
        # Q_m(k) = R_f put'(k) on the put side and fits the isotonic regression
        # to it, so put'(k) = Q_m(k) / R_f must come from the FITTED CDF. A raw
        # gradient of the price curve is the pre-isotonic marginal, which
        # breaks P^L <= P* at short maturities.
        put_prime_at_k = float(index_cdf(k_level)) / forward
        boundary = forward * k_level**gamma * (put_prime_at_k - gamma * put_at_k / k_level)
        integral = _midpoint_integral(grid, kernel * put_grid, grid <= k_level)
        return boundary + forward * integral

    if tail == "upper":
        call_at_k = float(np.interp(k_level, grid, call_grid))
        # Symmetric: Q_m(k) = R_f call'(k) + 1 on the call side.
        call_prime_at_k = (float(index_cdf(k_level)) - 1.0) / forward
        boundary = forward * k_level**gamma * (gamma * call_at_k / k_level - call_prime_at_k)
        integral = _midpoint_integral(grid, kernel * call_grid, grid >= k_level)
        return boundary + forward * integral

    raise ValueError(f"tail must be 'lower' or 'upper', got {tail!r}")
