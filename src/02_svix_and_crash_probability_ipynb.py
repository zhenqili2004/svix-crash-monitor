# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02. Replicating SVIX and the Probability of a Crash
#
# ```{admonition} In this guide
# :class: tip
# **Option prices are a probability distribution.** The slope of the call
# price curve across strikes is the market's risk-neutral CDF (Breeden and
# Litzenberger). You write this step in `rnd.risk_neutral_cdf`.
#
# **SVIX is that distribution's variance.** SVIX² is the variance of the
# market's return under the *risk-neutral* distribution, not the real-world
# one. It is a sum of out-of-the-money option prices with the same weight on
# every strike. The VIX leans on the low strikes, so the gap between the VIX
# and SVIX measures the left tail. You write this in `svix.svix_squared`.
#
# **The market's crash probability is a price, not a forecast.** The
# risk-neutral probability of a crash overstates the real one. Martin's log
# investor shrinks it by the price of a put, and more risk aversion shrinks it
# further. You write this in `crash_prob.crash_probability`.
# ```
#
# Ian Martin's [*What is the Expected Return on the Market?*](https://academic.oup.com/qje/article/132/1/367/2724543)
# (QJE 2017; the paper is in `papers/`) builds two numbers from S&P 500
# option prices and publishes both on [his website](https://personal.lse.ac.uk/martiniw/):
# **SVIX**, a volatility index like the VIX with different weights on the
# strikes, and the **probability of a 20% decline** in the S&P 500 over the
# coming month. This guide rebuilds both for one session, from the E-mini
# settlements of guide 01. There are three formulas, the three functions
# Part B asks you to complete, and each gets a picture. The pictures are the
# part to remember.
#
# **Sections**
#
# 1. From option prices to a distribution.
# 2. SVIX: add up the option prices.
# 3. Three beliefs about a crash.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import crash_prob
import figures
import monitor
import pull_es_options
import pull_fred
import rnd
import smile
import svix
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
pd.set_option("display.width", 120)

options = pull_es_options.load_es_options()
fred = pull_fred.load_fred()
# The newest session the monitor computed a row for (a session it had to
# skip, a rare chain with too few usable strikes, is not shown here either).
computed = set(monitor.load_history()["date"]) & set(options["date"])
latest = max(computed) if computed else options["date"].max()
session = options[options["date"] == latest]
rate = pull_fred.rate_on(fred, latest)
near, far = smile.bracketing_expiries(session)
month_expiry = smile.closest_expiry(session)  # the expiration closest to 30 days
cdf_near = rnd.risk_neutral_cdf(smile.smile_for(options, latest, near, rate), rate)
cdf_far = rnd.risk_neutral_cdf(smile.smile_for(options, latest, far, rate), rate)
cdf_month = {near: cdf_near, far: cdf_far}[month_expiry]
print(f"Session {latest.date()}; expirations {pd.Timestamp(near).date()} and {pd.Timestamp(far).date()} bracket 30 days; "
      f"one-month Treasury yield {100 * rate:.2f}%")

# %% [markdown]
# ## Two kinds of probability
#
# There are two distributions of the market's return a month from now. The
# **real-world** distribution says how often each outcome will happen. The
# **risk-neutral** distribution, written $P^*$ and $E^*$, is the set of
# probabilities under which every asset's price equals its expected payoff,
# discounted at the risk-free rate. A call struck at $k$ costs
# $c(k) = E^*[(R_T - k)^+] / R_f$, where $R_T$ is the gross return of the
# index (its level at expiration over today's) and $R_f$ the gross risk-free
# rate.
#
# The two differ because a dollar is worth more in a crash than in a boom.
# A price weights each state by that worth as well as by its likelihood, so
# the risk-neutral distribution puts more weight on the bad states than the
# real world does. Option prices reveal the risk-neutral distribution
# directly. The risk-neutral CDF $Q(k) = P^*(R_T \le k)$ is the
# price-implied chance that the index ends below $k$ times today's level.
# Sections 1 and 2 read the risk-neutral distribution; section 3 is the
# step back toward real-world probabilities.

# %% [markdown]
# ## 1. From option prices to a distribution
#
# A call struck at $k$ pays $(R_T - k)^+$. Raise the strike a little and the
# payoff falls by that little in every state where the call finishes in the
# money, and not at all elsewhere. So the slope of the call price in the
# strike is minus the risk-neutral probability of finishing above $k$,
# discounted:
#
# $$\frac{dc}{dk} = -\frac{P^*(R_T > k)}{R_f}
# \qquad\Longrightarrow\qquad
# Q(k) = 1 + R_f\,\frac{dc}{dk}.$$
#
# This is Breeden and Litzenberger's result: the CDF is the slope of the
# call curve, and the density is its curvature. To take a slope we need a
# call price at every strike, not only at the listed ones, so the engine
# interpolates the smile of guide 01 onto a fine grid and prices a call at
# every point (the recipe of Martin and Shi's Appendix D). The spot is
# normalized to 1, so a strike and a gross return are the same number $k$.
#
# The figure has the three steps, one panel each. Use the menu to change the
# expiration.

# %%
figures.fig_distribution(options, session, latest, rate).show()

# %% [markdown]
# **Takeaway.** The call curve falls and flattens as the strike rises, its
# slope is the CDF, and the CDF's slope is the density. Short
# expirations give narrow distributions, long ones wide; the height of the
# CDF at 0.8 is the risk-neutral probability of a 20% drop.

# %%
p_star_20 = crash_prob.risk_neutral_crash_prob(cdf_month, 0.80)
p_star_10 = crash_prob.risk_neutral_crash_prob(cdf_month, 0.90)
print(f"Risk-neutral probability of a 20% drop by {pd.Timestamp(month_expiry).date()}: {100 * p_star_20:.2f}%; "
      f"of a 10% drop: {100 * p_star_10:.2f}%")

# %% [markdown]
# ```{admonition} Stop and discuss
# :class: note
# The density has a long, fat left tail. Is that what the market *expects*,
# or what it *pays to avoid*? From option prices alone, could you tell the
# two apart?
# ```
#
# **One cleaning step.** Why interpolate at all? Take the same slope on the
# settlement prices directly, strike by strike, and check the one property
# every CDF has: it never falls.

# %%
for expiry in [near, month_expiry]:
    spot = smile.smile_for(options, latest, expiry, rate)["spot_price"].iloc[0]
    calls = session[(session["expiration"] == expiry) & (session["cp_flag"] == "C") & (session["settle"] > smile.MIN_SETTLE)].sort_values("strike")
    k = calls["strike"].to_numpy() / spot
    tau = calls["days_to_expiry"].iloc[0] / 365
    q_from_prices = 1.0 + np.exp(rate * tau) * np.gradient(calls["settle"].to_numpy() / spot, k)
    falls = np.diff(q_from_prices) < 0
    print(f"{calls['days_to_expiry'].iloc[0]:>3} days: the 'CDF' from settlement prices falls at {falls.sum()} of {len(k)} strikes, "
          f"by up to {100 * (-np.diff(q_from_prices)).max(initial=0):.1f} percentage points")

# %% [markdown]
# **Takeaway.** Settlements are marked to the tick, strike by strike, so
# their differences are noisy, and a CDF read off them falls in places,
# which no probability can do. The engine differentiates the smooth grid
# instead, then an isotonic regression (the closest non-decreasing curve)
# and a clip to $[0, 1]$ make the result a proper CDF. A vendor's fitted
# surface hides this problem; working from the exchange's prices, you see
# it and fix it.

# %% [markdown]
# ## 2. SVIX: add up the option prices
#
# Martin defines SVIX² as the variance of the market's return under the
# **risk-neutral** distribution of section 1, scaled by the risk-free rate
# and annualized (his equation 13):
#
# $$\mathrm{SVIX}^2 = \frac{1}{\tau}\,\mathrm{var}^*\!\left(\frac{R_T}{R_f}\right).$$
#
# The star matters: this is not a forecast of how volatile the market will
# be, but the variance that option prices imply. Martin's paper uses it to
# put a lower bound on the expected return of the market, the question in
# its title.
#
# Why is a variance a sum of option prices? Any smooth payoff can be built
# from a bond, a forward, and out-of-the-money options, with the option at
# strike $K$ weighted by the payoff's curvature there (Carr and Madan 1998).
# A variance is the price of a squared return, whose curvature is the same
# at every strike, so **every strike gets the same weight** (Martin's
# equation 12, spot normalized to 1):
#
# $$\mathrm{SVIX}^2 = \frac{2}{\tau\, R_f}\left[\int_0^{R_f} p(k)\,dk + \int_{R_f}^{\infty} c(k)\,dk\right].$$
#
# Puts below the forward, calls above it. The VIX prices a log payoff
# instead, whose curvature is $1/K^2$, so it leans on the low strikes. The
# figure shows each index's weights, as a share of its total:

# %%
figures.fig_svix_weights(cdf_month, latest).show()

# %%
svix2_near, svix2_far = svix.svix_squared(cdf_near), svix.svix_squared(cdf_far)
svix2_30 = svix.interpolate_variance(svix2_near, cdf_near.maturity_years, svix2_far, cdf_far.maturity_years)
# FRED publishes the VIX close a day later, so take the latest close on or before the session.
vix_obs = fred.dropna(subset=["vix"])
vix_obs = vix_obs[vix_obs["date"] <= latest].iloc[-1]
print(f"SVIX at {int(round(cdf_near.maturity_years * 365))} days: {svix.svix_index(svix2_near):.2f}; "
      f"at {int(round(cdf_far.maturity_years * 365))} days: {svix.svix_index(svix2_far):.2f}")
print(f"Interpolated to 30 days, as the VIX is: SVIX {svix.svix_index(svix2_30):.2f}")
print(f"The VIX closed at {vix_obs['vix']:.2f} on {vix_obs['date'].date()}")

# %% [markdown]
# **Takeaway.** SVIX sits below the VIX because the VIX's $1/K^2$ weights
# lean on the deep puts, the most expensive options in volatility terms (the
# smirk of guide 01). The VIX-style number in the legend, computed from the
# same settlements, lands close to the VIX Cboe published that day: a check
# that the options are being read right. Two expirations rarely sit exactly
# 30 days out, so, like the VIX, the monitor interpolates SVIX² times $\tau$
# between the two that bracket 30 days.
#
# ```{admonition} Stop and discuss
# :class: note
# If returns were lognormal, SVIX would sit slightly *above* the VIX at a
# one-month horizon. Martin found it below the VIX in every year of his
# sample, and it is below today. What does that gap say about how the
# options market prices a fall?
# ```

# %% [markdown]
# ## 3. Three beliefs about a crash
#
# $Q(0.8)$ overstates the chance of a crash, because option prices weight a
# crash by what a dollar is worth then. Martin's way back (his Result 2) is
# to ask what an investor with log utility who holds the market must believe
# to be content holding it. Their marginal utility is $1/R_T$, which turns
# the risk-neutral probability into
#
# $$P(R_T \le \alpha) = \frac{\alpha\, Q(\alpha)}{R_f} - p(\alpha),$$
#
# the risk-neutral probability, shrunk by the price of the put struck at the
# crash level $\alpha$. Martin and Shi (2025) do the same for power utility
# with risk aversion $\gamma$, weighting each state by $R_T^\gamma$:
# $P = E^*[R_T^\gamma 1\{R_T \le \alpha\}] / E^*[R_T^\gamma]$. At
# $\gamma = 1$ it is Martin's formula, and they calibrate $\gamma = 2$.
#
# The figure reads all three off the same distribution, for every size of
# drop. Drag the slider to change the risk aversion.

# %%
figures.fig_crash_beliefs(cdf_month, latest).show()

# %%
rows = []
for label, alpha in [("20% drop", 0.80), ("10% drop", 0.90)]:
    rows.append({
        "event": f"{label} by {pd.Timestamp(month_expiry).date()}",
        "risk-neutral, P*": crash_prob.risk_neutral_crash_prob(cdf_month, alpha),
        "log investor (Martin 2017)": crash_prob.crash_probability(cdf_month, alpha),
        "power utility, gamma = 2 (Martin and Shi 2025)": crash_prob.crash_probability_power(cdf_month, alpha),
    })
pd.DataFrame(rows).set_index("event").style.format("{:.2%}")

# %% [markdown]
# **Takeaway.** The market charges for a crash at a probability well above
# what an investor who holds the market would assign to it, and the more
# risk-averse the investor, the larger the share of an option's price that
# is fear rather than likelihood. At a risk aversion of 1 the power-utility
# line lands exactly on the log investor's.
#
# ```{admonition} Stop and discuss
# :class: note
# A risk committee asks you for "the probability of a 20% drop next month."
# Which of these three numbers do you report, and what do you say about the
# other two?
# ```

# %% [markdown]
# ## The homework
#
# Part B asks for the three formulas of this guide, in `src/rnd.py`,
# `src/svix.py`, and `src/crash_prob.py`. Each function's docstring repeats
# its formula, and `uv run pytest src/test_svix.py src/test_rnd.py src/test_crash_prob.py`
# checks them offline: under a flat smile, returns are lognormal and every
# answer has a closed form. With the three in place, `uv run doit` computes
# this session, and every session in the pull, into the monitor's history,
# which is guide 03.

# %% [markdown]
# ```{admonition} Notes on the discussion questions
# **What the market expects, or what it pays to avoid?** From option prices
# alone you cannot tell. The risk-neutral density is the real-world density
# reweighted by how much a dollar is worth in each state, so a fat left tail
# can mean that crashes are likely, that investors badly want dollars in a
# crash, or both. Separating the two takes an assumption about preferences,
# which is what section 3 makes, or a long record of how often crashes
# actually happen.
#
# **SVIX below the VIX.** The two indexes add up the same option prices with
# different weights. Under a lognormal distribution the weightings nearly
# agree, and SVIX comes out slightly above. The VIX's $1/K^2$ weights magnify
# the deep puts, so the VIX ends up above SVIX when those puts cost more than
# a lognormal distribution says they should, that is, when the market prices
# a left tail fatter than the normal curve's. The gap measures how much the
# market pays for crash insurance beyond what volatility alone explains.
#
# **Which number for the risk committee?** It depends on the question they
# are asking, and a good report says so. The risk-neutral probability is the
# price of insurance: what it costs to hedge a 20% drop, and an overstatement
# of how likely the drop is. The log investor's and the power-utility
# investor's numbers are forecasts, but each rests on an assumed utility
# (Martin and Shi calibrate risk aversion at 2). Report a forecast, name the
# assumption behind it, and give the risk-neutral number as the cost of
# protection.
# ```
