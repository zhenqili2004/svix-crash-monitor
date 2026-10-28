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
# # 04. Crash Bounds for Single Names (Martin and Shi 2025)
#
# ```{admonition} Optional reading
# :class: important
# This guide is optional, and it is not on the exam. Part D of HW 3 needs
# two of its sections: **The data**, which describes the query whose
# `WHERE` clause you write, and **SVIX against Martin's published series**,
# the check your `svix_squared` must pass. The Martin-Shi bounds and the
# thirty years of crash probabilities are there for the curious.
# ```
#
# ```{admonition} In this guide
# :class: tip
# **A stock's crash probability needs more than its options.** The
# real-world probability that one stock crashes depends on how it moves
# with the market in a crash, which no option price reveals, so Martin and
# Shi bound it instead of estimating it.
#
# **A vendor's surface stops short of the crash.** The OptionMetrics
# surface ends near the 10-delta put, so every one-month tail number built
# from it rests on an extrapolation convention (guide 01, section 7).
#
# **With no exchange benchmark, the check is the published series.** No
# exchange publishes an SVIX for the S&P 500, so the pipeline is validated
# against the series Martin publishes.
# ```
#
# The first three guides worked with the market as a whole. This one asks
# the question of [Martin and Shi (2025)](https://personal.lse.ac.uk/martiniw/oib_martin_shi_latest.pdf),
# *Forecasting Crashes with a Smile*: what is the probability that **one
# stock**, Apple or AIG, falls 20% in the next month?
#
# Options on the stock give its risk-neutral distribution, exactly as index
# options gave the market's in guide 02. But a risk-neutral probability
# "cries wolf": it prices crash states at their marginal utility, not their
# frequency. The correction that worked for the index, weighting each
# state by the investor's return, now needs to know how the stock and the
# market move *together* in a crash, and no option price reveals that.
# Martin and Shi's answer is to bound it. Two marginal distributions pin
# the joint one down only so far (the Fréchet-Hoeffding theorem), and the
# two extremes, the stock crashing exactly when the market does and
# crashing exactly when it does not, give a lower and an upper bound. The
# lower bound turns out to forecast crashes out of sample with nothing
# estimated on the stocks.
#
# **Sections**
#
# - The data: the OptionMetrics volatility surface, and what it does not
#   cover.
# - The paper's Figure 1: the bounds for Apple and AIG, 1996 to 2022, and
#   extended to the latest surface.
# - Thirty years of crash fear: the index's crash probability from the
#   surface, checked against the monthly series Martin (2017) publishes.
# - SVIX from the surface, checked against Martin's published daily SVIX.
#
# The code is Riley Haas and Marija Jovicic's, from their Summer 2026
# replication of the paper (`forecasting_crashes_smile`), used with their
# permission. The computation runs in `src/single_name_bounds.py` as the
# `single_names` task, and the SVIX check in `src/calc_svix_vs_martin.py`;
# this notebook loads their output.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import figures
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

HAVE_RESULTS = (OUTPUT_DIR / "single_name_bounds.parquet").exists() and (
    OUTPUT_DIR / "index_crash_probabilities.parquet"
).exists()
HAVE_SVIX = (OUTPUT_DIR / "svix_vs_martin.parquet").exists()
if not HAVE_RESULTS:
    print(
        "The single-name results are not built. They need the OptionMetrics pull\n"
        "from WRDS, which is not part of the public monitor: run `uv run doit single_names`\n"
        "with your WRDS username in .env, then rebuild. The rest of this notebook is\n"
        "skipped until then."
    )

# %% [markdown]
# ## The data: the OptionMetrics volatility surface
#
# Guides 01 and 02 used the exchange's settlement prices, one per listed
# strike. Single-name options are not on the course's Databento license, so
# this part uses what the academic literature uses: OptionMetrics' IvyDB on
# WRDS, and specifically its **standardized volatility surface**, which the
# vendor fits every day for every security at fixed maturities (30, 60, 91,
# ... days) and fixed deltas (10 to 90, puts and calls). Martin and Shi's
# Appendix D filters on the surface's `dispersion`, a goodness-of-fit
# measure, and keys moneyness off `impl_strike`, both of which exist only in
# the surface table, so the surface is the paper's specification, not a
# shortcut.
#
# Three WRDS tables, pulled by `src/pull_optionmetrics.py` for the S&P 500,
# Apple, and AIG at 30 and 91 days: the surface (`vsurfd{year}`, one table
# per year), daily closes for the spot (`secprd{year}`), and the zero curve
# (`zerocd`). The homework's one blank in this part is the `WHERE` clause of
# the surface query.

# %%
if HAVE_RESULTS:
    surface = pd.read_parquet(DATA_DIR / "optionm_vol_surface.parquet")
    print(f"{len(surface):,} surface rows, {surface['date'].min().date()} to {surface['date'].max().date()}")
    one = surface[(surface["secid"] == 108105) & (surface["days"] == 30)]
    one = one[one["date"] == one["date"].max()].sort_values(["cp_flag", "delta"])
    display(one.head(8))

# %% [markdown]
# A surface slice is 34 points: puts from delta -10 to -90 and calls from
# 10 to 90. Note where the strikes stop. The deepest put on the surface is
# the 10-delta put, and over the sample its strike sits around 92% of the
# spot at 30 days:

# %%
if HAVE_RESULTS:
    prices = pd.read_parquet(DATA_DIR / "optionm_security_prices.parquet")
    spx = surface[(surface["secid"] == 108105) & (surface["impl_strike"] > 0)].merge(
        prices[prices["secid"] == 108105][["date", "close"]], on="date"
    )
    spx["moneyness"] = spx["impl_strike"] / spx["close"]
    deepest = spx.groupby(["date", "days"])["moneyness"].min().unstack("days")
    coverage = pd.DataFrame(
        {
            "median deepest put (K/S)": deepest.median(),
            "share of days reaching K/S <= 0.80": (deepest <= 0.80).mean(),
            "share of days reaching K/S <= 0.90": (deepest <= 0.90).mean(),
        }
    )
    display(coverage)

# %% [markdown]
# A 20% drop is at moneyness 0.80, well beyond the deepest strike on the
# 30-day surface. Everything the engine does below that strike rests on an
# extrapolation convention: Appendix D holds the implied volatility flat
# beyond the last observed point. That is faithful to the paper, and it is
# the reason guide 01 made such a point of the exchange's settlement prices,
# which exist for every listed strike down to a fraction of the spot;
# guide 01's last figure shows what the convention costs. Keep it in mind
# when reading the one-month numbers.

# %% [markdown]
# ## The paper's Figure 1: Apple and AIG
#
# For each month-end, each name, and each maturity, `single_name_bounds.py`
# builds the risk-neutral CDF of the name and of the index (the engine of
# guide 02, `rnd.risk_neutral_cdf`) and applies Result 3 of the paper. In
# words: read off the stock's risk-neutral crash probability
# $p^* = Q_i(q)$; find the market return $q_l$ that is exactly as unlikely,
# $Q_m(q_l) = p^*$; then the lower bound is the market's own probability
# of falling to $q_l$, corrected for risk aversion as in guide 02. The
# upper bound is the same construction in the market's right tail. In
# symbols:
#
# $$
# \frac{E^*[R_m^\gamma\, 1\{R_m \le q_l\}]}{E^*[R_m^\gamma]}
# \;\le\; P[R_i \le q] \;\le\;
# \frac{E^*[R_m^\gamma\, 1\{R_m \ge q_u\}]}{E^*[R_m^\gamma]},
# \qquad q_l = Q_m^{-1}(Q_i(q)),\; q_u = Q_m^{-1}(1 - Q_i(q)),
# $$
#
# with $\gamma = 2$, calibrated once on the market (the paper's Appendix
# C). The lower bound is attained when the stock and the market crash
# together; the upper bound when they move against each other. The
# risk-neutral probability $Q_i(q)$ always lies between them. Here is the
# paper's Figure 1, the bounds on a 20% drop within a month, over the
# paper's 1996 to 2022 sample:

# %%
if HAVE_RESULTS:
    bounds = pd.read_parquet(OUTPUT_DIR / "single_name_bounds.parquet")
    figures.fig_single_names(bounds).show()

# %% [markdown]
# **Takeaway.** The two names tell the paper's story. AIG's band explodes in
# the autumn of 2008, when its options priced the crash that was happening;
# Apple's spikes are the dot-com bust and 2008, and its bounds are the
# quieter of the two after 2010. The width of the band is what the options
# cannot tell you: how the stock and the market move together in a crash.
# The buttons switch between the paper's sample and the latest surface.
#
# ```{admonition} Stop and discuss
# :class: note
# At the end of September 2008, the bounds put AIG's chance of another 20%
# fall within a month between about one in three and one in two. What does
# a band that wide tell a risk manager, and what extra information would
# narrow it?
# ```
#
# The latest month-end, for both names and both maturities:

# %%
if HAVE_RESULTS:
    latest = bounds[(bounds["date"] == bounds["date"].max()) & (bounds["threshold"] == 0.80)]
    display(latest.set_index(["ticker", "days_to_maturity"])[["bound_lower", "prob_riskneutral", "bound_upper"]])

# %% [markdown]
# ## Thirty years of crash fear, against Martin's published series
#
# When the "name" is the market itself, the two returns are comonotonic and
# the lower bound holds with equality (the paper's equation 7), so the
# index's crash probability is a point, not a band. That is the number
# guide 02 computed from the exchange's settlement prices for one session.
# Here it is from the vendor's surface for every day since 1996, under
# the three beliefs of guide 02. Zoom in on 2008 and on March 2020.

# %%
if HAVE_RESULTS:
    index = pd.read_parquet(OUTPUT_DIR / "index_crash_probabilities.parquet")
    martin = pd.read_csv(MANUAL_DATA_DIR / "martin_2017" / "crashprob.csv")
    figures.fig_crash_history(index, martin).show()

# %% [markdown]
# Martin (2017) publishes exactly the green series, the log investor's
# probability of a 20% drop, averaged by month, from January 1996 to
# January 2012 (`data_manual/martin_2017/crashprob.csv`, the black dots).
# His inputs are S&P 500 index options from the same vendor, so this is the
# check on the single-name half of the pipeline:

# %%
if HAVE_RESULTS:
    m1 = index[(index["days_to_maturity"] == 30) & (index["threshold"] == 0.80)]
    ours = m1.assign(year=m1["date"].dt.year, month=m1["date"].dt.month)
    monthly = ours.groupby(["year", "month"])["prob_log_utility"].mean().reset_index()
    merged = monthly.merge(martin[["year", "month", "m1"]], on=["year", "month"])
    merged["date"] = pd.to_datetime(merged[["year", "month"]].assign(day=1))
    gap = (merged["prob_log_utility"] - merged["m1"]).abs()
    print(f"Months in the overlap: {len(merged)}")
    print(f"Correlation with Martin's series: {np.corrcoef(merged['prob_log_utility'], merged['m1'])[0, 1]:.3f}")
    print(f"Median absolute gap: {100 * gap.median():.2f} percentage points; mean: {100 * gap.mean():.2f}")
    display(merged[merged["year"].isin([2008, 2011])].set_index("date")[["prob_log_utility", "m1"]].rename(
        columns={"prob_log_utility": "ours", "m1": "Martin"}))

# %% [markdown]
# **Takeaway.** Crash fear spikes with every crisis, and ours tracks
# Martin's: October 2008, the flash crash of May 2010, August 2011. Ours runs a little below Martin's in
# calm months and above it in the worst ones, and the first of those is the
# surface again: with its strikes stopping near 92% of the spot, the flat
# extrapolation into the tail understates the price of deep puts in quiet
# markets, and the probability of a drop that starts beyond the last strike
# comes out small. Martin works from the raw option quotes, which reach
# further. The exchange's settlement prices in guides 01 and 02 are the
# cure.
#
# ```{admonition} Stop and discuss
# :class: note
# Ours runs below Martin's in calm months. Using guide 01's flat
# extrapolation figure, explain why a surface that stops near 92% would
# understate the tail most when markets are calm.
# ```

# %% [markdown]
# ## SVIX against Martin's published series
#
# The same surface gives SVIX, with the integral of guide 02. Martin
# publishes his one-month SVIX², daily from January 1996 to January 2012
# (`data_manual/martin_2017/svix2.csv`), computed from S&P 500 index
# options. No exchange publishes an SVIX for the S&P 500: CME's CVOL indexes
# use the same construction, but only for options on its interest-rate,
# currency, energy, metals, and agricultural futures. So Martin's series is
# the check on `svix_squared`. Both are shown in VIX units, 100 times the
# square root of SVIX²:

# %%
if HAVE_SVIX:
    sv = pd.read_parquet(OUTPUT_DIR / "svix_vs_martin.parquet")
    print(f"Days in the overlap: {len(sv):,} ({sv['date'].min().date()} to {sv['date'].max().date()})")
    print(f"Correlation with Martin's series: {np.corrcoef(sv['svix'], sv['svix_martin'])[0, 1]:.4f}")
    print(f"Median absolute gap: {sv['gap'].abs().median():.2f} vol points; mean gap: {sv['gap'].mean():+.2f}")
    figures.fig_svix_check(sv).show()
else:
    print("The SVIX check is not built: run `uv run doit single_names`.")

# %% [markdown]
# **Takeaway.** The two lines are hard to tell apart. What gap there is comes from the
# same seam as before: the surface's strikes stop near the 10-delta put, and
# holding the volatility flat beyond them puts a little more or a little
# less area under the out-of-the-money price curve than Martin's raw quotes
# do. One of the homework's tests runs your `svix_squared` on this surface
# at every month-end in the sample and requires it to stay within a
# fraction of a vol point of Martin's series.

# %% [markdown]
# ## Where this leaves the homework
#
# Part D asks for one thing: the `WHERE` clause of the surface query in
# `src/pull_optionmetrics.py`, which selects the three securities, the two
# maturities, and the sample window from the vendor's yearly tables. Two
# tests run it against a small in-memory database with the WRDS layout, as
# HW 1 tested its queries. Then `uv run doit single_names` pulls the
# surface and runs the chain that made every chart on this page, and the
# last test checks the index probability against Martin's series as above.
# The same pull feeds Part B's check of `svix_squared` against Martin's SVIX.

# %% [markdown]
# ```{admonition} Notes on the discussion questions
# **AIG in 2008.** At the end of September 2008 the bounds were about 34% and
# 50%. A band that wide still says something firm. The lower bound is the
# case where AIG crashes only when the market does, the case that gives the
# smallest probability, and even then the chance of another 20% fall within
# a month was at least one in three. That floor is the number a risk manager
# needs. What the options cannot say is how AIG and the market move together
# in a crash; narrowing the band takes information about that joint
# behavior, such as a model of the dependence, its record in past crises, or
# the prices of contracts that pay on both at once.
#
# **Below Martin's in calm months.** In a calm market implied volatility is
# low, so the 10-delta put, the surface's last strike, sits close to the
# money, and the whole region from there down to a 20% drop is filled by
# holding the volatility flat. The true smirk keeps rising there (guide 01's
# last figure), so the extrapolated deep puts are too cheap and the crash
# probability too small. Martin worked from the raw option quotes, which
# reach deeper strikes. In a crisis volatility is high, the 10-delta put
# moves far out of the money, and less of the tail is extrapolated, so the
# gap closes.
# ```
