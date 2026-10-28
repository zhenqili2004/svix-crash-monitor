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
# # 01. E-mini S&P 500 Options Data from Databento
#
# ```{admonition} In this guide
# :class: tip
# **One row is one contract's settlement on one session.** A strike, a
# right (call or put), an expiration, and a price the exchange assigns to
# every listed contract at every close, traded that day or not.
#
# **Each option is an option on a future.** It exercises into an E-mini
# S&P 500 future, so the chain is read against that future's price, which
# put-call parity recovers from the options themselves.
#
# **The exchange reaches the crash strikes.** Pricing insurance against a
# 20% fall takes puts struck at 80% of today's level or lower. The exchange
# settles puts that deep every day; a vendor's standard surface, the usual
# academic data, stops near 92% and fills the rest by assumption. That reach
# is why this case study uses the exchange's prices.
# ```
#
# Everything in this case study starts from one table: the price of every
# listed option on the S&P 500 at the close of one session. This guide is
# about what that table is and how to read it, for readers who have not
# worked with futures before.
#
# In HW 2 you pulled 30-Day Fed Funds futures from Databento's CME Globex
# feed. Options on futures come from the same feed, and the pull in
# `src/pull_es_options.py` is written for you this time. What is new is the
# data: options have a strike and an expiration, so one product is
# thousands of contracts, and the price we want is not a trade but a
# **settlement**.
#
# The settlements come under the course's Databento license, so these guides
# show quantities computed from them (one price relative to another, implied
# volatilities, probabilities) and never the prices themselves. Your own
# pull has every one of them, in `_data/es_options_settlements.parquet`.
#
# **Sections**
#
# 1. The future first.
# 2. Which options, and how to read a row.
# 3. What expires when.
# 4. Where the rows come from: settlements and definitions.
# 5. One expiration's chain.
# 6. The smile, and the surface.
# 7. Where the strikes stop, which is the point of the whole exercise.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import figures
import monitor
import pull_es_options
import pull_fred
import smile
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

options = pull_es_options.load_es_options()
futures = pull_es_options.load_es_futures()
fred = pull_fred.load_fred()
# The newest session the monitor computed a row for (a session it had to
# skip, a rare chain with too few usable strikes, is not shown here either).
computed = set(monitor.load_history()["date"]) & set(options["date"])
latest = max(computed) if computed else options["date"].max()
session = options[options["date"] == latest]
rate = pull_fred.rate_on(fred, latest)
curve = figures.futures_curve(futures, latest)
print(f"{len(options):,} option settlements over {options['date'].nunique()} sessions, "
      f"{options['date'].min().date()} to {latest.date()}")

# %% [markdown]
# ## 1. The future first
#
# An **E-mini S&P 500 future** (CME root `ES`) is a contract to buy the S&P
# 500 index on a fixed future date at a price agreed today, at $50 per index
# point. No money changes hands when the contract is opened. Instead, every
# evening the exchange sets a **settlement price** for the contract and
# moves the day's gain or loss from one side's account to the other's. There
# are four expirations a year, on the third Friday of March, June,
# September, and December, and a symbol names its expiration with a month
# code and the last digit of the year: `ESZ6` is December 2026. Here are
# the nearest listed contracts at the close of the latest session, each
# priced relative to the nearest one (the figure below has all of them):

# %%
curve[["symbol", "contract month", "expiration", "days to expiration", "above the nearest (%)"]].head(6).round(2)

# %%
figures.fig_futures_curve(curve, latest, underlyings=session["underlying"].unique()).show()

# %%
first, second = curve.iloc[0], curve.iloc[1]
carry = np.log(second["settlement"] / first["settlement"]) / ((second["expiration"] - first["expiration"]).days / 365)
print(f"From {first['symbol']} to {second['symbol']}, the curve rises at {100 * carry:.1f}% a year; "
      f"the one-month Treasury yield is {100 * rate:.1f}%")

# %% [markdown]
# **Takeaway.** Longer-dated futures settle higher. Agreeing today to buy
# the index later, instead of buying it now, keeps your cash earning
# interest and gives up the dividends, so a future's price is the index
# carried forward at the interest rate less the dividend yield. The options
# in this case study exercise into the first few of these contracts (in
# blue).
#
# ```{admonition} Stop and discuss
# :class: note
# The contract five years out settles well above the nearest one. Is that
# the market forecasting that stocks will rise that much? What would have
# to change for the curve to slope down?
# ```

# %% [markdown]
# ## 2. Which options, and how to read a row
#
# CME lists several families of options on the E-mini future. The quarterly
# options expire with the future itself, in the morning, and are
# American-style. The **end-of-month options** (root `EW`) are
# European-style and settle at 4:00 p.m. ET, like the SPX options the
# academic literature uses. They give an expiration at the end of every
# month about seven months out, and then quarterly, which is enough to
# bracket a 30-day horizon on every session. That is what we pull.
#
# An option symbol extends a future's: `EWX6 P6900` is root `EW`, month code
# `X` (November), year digit `6`, a **put** struck at **6,900**. A put is
# the right to sell, at the strike, and a call the right to buy. Prices are
# in index points, like the future's, at 50 dollars a point.
# Here is the decoding for a few of the latest session's contracts, with the
# expiration and the future each one exercises into, which the symbol does
# not say:

# %%
month_expiry = smile.closest_expiry(session)  # the expiration closest to 30 days
chain = session[session["expiration"] == month_expiry]
surface = smile.surface_from_chain(chain, rate)
forward = surface["forward"].iloc[0]


def nearest(expiration, flag, moneyness):
    """The listed contract closest to a given strike, as a fraction of the forward."""
    rows = session[(session["expiration"] == expiration) & (session["cp_flag"] == flag)]
    return rows.iloc[(rows["strike"] - moneyness * forward).abs().argmin()]


expirations = sorted(session["expiration"].unique())
examples = pd.DataFrame([nearest(month_expiry, "P", 0.90), nearest(month_expiry, "C", 1.03),
                         nearest(expirations[0], "P", 1.00), nearest(expirations[-1], "P", 0.80)])
decoded = pd.DataFrame([figures.decode_symbol(s, latest) for s in examples["symbol"]])
decoded[["expiration", "underlying future"]] = examples[["expiration", "underlying"]].to_numpy()
decoded

# %%
put = examples.iloc[0]
print(f"Read aloud: {put['symbol']} is the right to sell one {put['underlying']} future at {put['strike']:,.0f} "
      f"on {put['expiration']:%B %-d, %Y}.")

# %% [markdown]
# And this is what the table itself looks like: one row per contract per
# session, after the pull has joined the price to the contract's terms. The
# price is the `settle` column, left out here.

# %%
chain.drop(columns="settle").sort_values(["strike", "cp_flag"]).iloc[len(chain) // 2:len(chain) // 2 + 6]

# %% [markdown]
# ## 3. What expires when
#
# Each end-of-month option exercises into the nearest quarterly future that
# expires after it does. In the figure, each row is a future: the diamond is
# the future's own expiration, and the dots are the option expirations that
# exercise into it. Hover for the details.

# %%
figures.fig_expiration_calendar(session, curve, latest).show()

# %% [markdown]
# **Takeaway.** On every session, one option expiration falls inside 30
# days and the next one beyond it (in red). The monitor computes SVIX for
# both and interpolates to exactly 30 days, as the VIX does. The table has
# the same information, with the strike range listed for each expiration:

# %%
(session.groupby("expiration")
 .agg(days_to_expiry=("days_to_expiry", "first"), contracts=("symbol", "size"),
      strikes=("strike", "nunique"), underlying=("underlying", "first"),
      lowest_strike=("strike", "min"), highest_strike=("strike", "max")))

# %% [markdown]
# ## 4. Where the rows come from: settlements and definitions
#
# A daily bar (`ohlcv-1d`, the schema HW 2 used) exists for a contract only
# on a day it traded, and most listed strikes do not trade on most days.
# What exists for every listed contract every day is the exchange's
# settlement price: the value CME assigns at the close to mark positions,
# from the quotes near the money and a model in the wings. It is in
# Databento's `statistics` schema, as one record type among several
# (`stat_type` 3; the schema also carries open interest, the session high
# and low, and others), published twice a day, a preliminary and a final;
# the pull keeps the final. The query has the four parameters you know from
# HW 2:

# %%
pull_es_options.build_queries("2026-10-15", "2026-10-16")["option_statistics"]

# %% [markdown]
# A `statistics` record names its contract by `instrument_id` and symbol,
# not by its expiration or strike. The `definition` schema has those terms,
# republished every day for every listed contract, so the pull joins the
# same day's definitions on `instrument_id`. Parent-symbol queries also
# return user-defined spreads, multi-leg strategies traders register during
# the day; their `instrument_class` is not C or P, and they are dropped.

# %% [markdown]
# ## 5. One expiration's chain
#
# Take the expiration closest to 30 days. Buy the call and sell the put at
# the same strike $K$, and at expiration you hold the future's price less
# $K$, whatever the future does. So the call's price less the put's is the
# value today of that forward position. Plot it at every strike. Hover over
# a point to read it.

# %%
future_settle = curve.set_index("symbol").loc[chain["underlying"].iloc[0], "settlement"]
figures.fig_parity(chain, forward, rate).show()
print(f"The parity-implied forward is {forward - future_settle:+.2f} index points from the "
      f"{chain['underlying'].iloc[0]} future's own settlement")

# %% [markdown]
# **Takeaway.** The points fall on a straight line,
# $C - P = e^{-r\tau}(F - K)$: put-call parity. Calls lose value and puts
# gain value as the strike rises, and they cost the same where the strike
# equals the forward, so the line crosses zero there. The forward it implies
# from the options alone matches the future's own settlement almost to the
# point. `smile.parity_forward` computes it; the engine uses it as the
# underlying price.
#
# ```{admonition} Stop and discuss
# :class: note
# Why do calls fall and puts rise as the strike rises? If the
# parity-implied forward were 50 points away from the future's settlement,
# what would you suspect about the data?
# ```

# %% [markdown]
# ## 6. The smile, and the surface
#
# A price is hard to compare across strikes; an implied volatility is not.
# `smile.surface_from_chain` inverts each out-of-the-money settlement, puts
# below the forward and calls above it, to the volatility that makes the
# Black-76 formula return it (the `black_scholes` module explains the
# formula for options on a future). The strike becomes moneyness,
# $k = K / S$, so that a strike and a gross return are the same number: a
# 20% drop is $k = 0.8$.
#
# Not every settlement is usable. CME's smallest price increment is 0.05
# index points, and a far out-of-the-money option that settles at 0.05 is
# marked at a floor, not priced; inverting it would give a meaningless
# volatility. The pull keeps those rows, since they are what the exchange
# published, and `surface_from_chain` drops them (the grey crosses).

# %%
figures.fig_smile(chain, surface, rate).show()

# %% [markdown]
# **Takeaway.** The smile is a smirk: out-of-the-money puts are dearer, in
# volatility terms, than out-of-the-money calls, because a market fall is
# what investors insure against. And the exchange settles puts far below a
# 20% drop, in the shaded region where a vendor's standard surface has no
# strikes at all.
#
# ```{admonition} Stop and discuss
# :class: note
# Who buys a put struck 20% below the market, and who sells it? Why would
# the seller demand a price that implies a volatility twice the at-the-money
# one?
# ```
#
# Every usable expiration of the same session gives a smile; stacked by
# days to expiration, they form the volatility surface. Drag to rotate it,
# and hover for any point.

# %%
figures.fig_surface(options, session, latest, rate).show()

# %% [markdown]
# **Takeaway.** The smirk is steepest for the shortest expirations and
# flattens with time: insurance against a sudden fall is priced dearest
# over the next few weeks.

# %% [markdown]
# ## 7. Where the strikes stop
#
# This is the reason the case study uses the exchange's prices. A 20% drop
# means the index ends at 0.80 of today's level. Insurance against it is a
# put struck at 0.80 or below, so a probability of a crash needs prices
# there.
#
# The standard academic source is OptionMetrics' volatility surface (the
# data of guide 04). The vendor fits each day's smile at fixed deltas, and
# its deepest point is the **10-delta put**, a put with about a one-in-ten
# chance of finishing in the money. Below that strike the vendor has no
# data, and the usual convention, Martin and Shi's Appendix D among them,
# holds the volatility flat. The figure puts that assumption next to what
# the exchange actually settles, on this session's smile:

# %%
figures.fig_flat_extrapolation(surface, rate).show()
flat = figures.flat_extrapolation(surface, rate)
print(f"The 10-delta put is at k = {flat['k10']:.3f}. At k = 0.80 the exchange's settlements imply a volatility of "
      f"{100 * flat['vol_actual']:.1f}%; the flat assumption, {100 * flat['vol_flat']:.1f}%.")
print(f"Priced at those volatilities, the 0.80 put costs {flat['ratio']:.0f} times less under the flat assumption "
      f"than as settled.")

# %% [markdown]
# **Takeaway.** Below the vendor's last strike, holding the volatility flat
# prices the deep puts far too cheaply, and so understates the probability
# of a crash. The exchange's settlements show what those puts actually cost.
#
# ```{admonition} Stop and discuss
# :class: note
# Guide 02 reads the probability of a crash off exactly these deep puts.
# Which way does the flat assumption bias it? Would you expect the bias to
# be larger in calm markets or in a crisis?
# ```
#
# And it is not one lucky session. On every session in the pull, the
# exchange settled a dozen or more usable puts below the crash threshold:

# %%
rows = []
for d, day in options.groupby("date"):
    try:
        near_d, _ = smile.bracketing_expiries(day)
        s = smile.surface_from_chain(day[day["expiration"] == near_d], pull_fred.rate_on(fred, d))
        puts = s[s["cp_flag"] == "P"]
        rows.append({"date": d.date(), "days": int(s["days_to_maturity"].iloc[0]),
                     "deepest usable put (K/S)": puts["moneyness"].min(),
                     "puts below K/S = 0.80": int((puts["moneyness"] <= 0.80).sum())})
    except ValueError:
        continue
pd.DataFrame(rows).set_index("date")

# %% [markdown]
# ```{admonition} Notes on the discussion questions
# **The futures curve.** No. A future's price is set by carry, not by a
# forecast. Buying the future instead of the index leaves your cash earning
# interest and gives up the dividends, so the future is the index grown at
# the interest rate less the dividend yield, $F = S e^{(r - q)T}$. If it were
# any higher, borrowing to buy the index and selling the future would lock in
# a riskless profit. The expected return on stocks appears nowhere in that
# formula. The curve slopes down when the dividend yield is above the
# interest rate, as it was when short rates sat near zero in 2020 and 2021.
#
# **Calls, puts, and parity.** A call is the right to buy at the strike, so a
# higher strike pays less in every state; a put is the right to sell, so a
# higher strike pays more. Parity is an arbitrage relation, so a forward 50
# points off the future's settlement means the data are wrong, not the
# market: a contract joined to the wrong future or expiration, a preliminary
# settlement kept instead of the final, or a rate or time to expiration that
# is off. That makes the parity check a cheap test of the whole pull.
#
# **Who buys and who sells deep puts.** Buyers are investors who hold the
# market and want insurance against a crash: pension funds, asset managers,
# insurers, tail-risk funds. Sellers are dealers and funds that collect the
# premium. The seller loses exactly when everyone else does, when capital is
# scarcest, and cannot hedge the loss smoothly, because prices jump in a
# crash. Demand for the insurance is steady and willing sellers are few, so
# the price is high. The smirk is also a lesson the market learned: S&P 500
# options showed little of it before the crash of October 1987, and a steep
# one ever since.
#
# **The flat assumption's bias.** Downward. Holding the volatility flat below
# the last strike prices the deep puts too cheaply, so the density puts too
# little weight on a crash, and every crash probability built from it,
# risk-neutral or real-world, comes out too small. The bias is largest in
# calm markets: with volatility low, the 10-delta put sits close to the
# money, and the whole stretch from there down to 0.80 is filled by
# assumption. In a crisis volatility is high, the 10-delta put moves far out,
# and less of the tail is extrapolated. Guide 04 sees exactly this pattern
# against Martin's published series.
# ```
