"""From one session's settlement prices to an implied-volatility smile.

The exchange publishes a settlement price for every listed strike. This
module turns the chain for one session and one expiration into the smile
that ``rnd.risk_neutral_cdf`` consumes: the forward implied by put-call
parity, the Black-76 implied volatility of each out-of-the-money contract,
and the strike expressed as moneyness.

Units. ``rnd`` normalizes everything by a spot. For options on a future
the spot is the discounted forward, S* = F exp(-r tau): feeding S* to the
spot-form Black-Scholes formulas reproduces the Black-76 price of an option
on the future exactly (the ``black_scholes`` module explains the identity).
So ``moneyness`` here is K / S*, and the forward in those units is R_f.
"""

import numpy as np
import pandas as pd

import black_scholes as bs

SECID_SPX = 108105  # OptionMetrics' id for the S&P 500, for a stitchable schema
MIN_SETTLE = 0.05  # CME's smallest premium tick; a settlement AT the tick is a floor, not a price
HORIZON_DAYS = 30
MIN_DAYS_NEAR = 7  # do not read a smile off an expiry closer than a week


def parity_forward(chain, rate, tau, n_pairs=5):
    """The underlying future's price implied by put-call parity.

    For a European option on a future, F = K + exp(r tau) (C - P) at every
    strike with both a call and a put. Take the median over the ``n_pairs``
    strikes with the smallest |C - P|, the nearest-the-money pairs, where
    pricing noise matters least.
    """
    priced = chain[chain["settle"] > 0]
    wide = priced.pivot_table(index="strike", columns="cp_flag", values="settle", aggfunc="last")
    if "C" not in wide or "P" not in wide:
        raise ValueError("need both calls and puts to imply a forward")
    wide = wide.dropna(subset=["C", "P"])
    if wide.empty:
        raise ValueError("no strike has both a call and a put settlement")
    f_k = wide.index.to_numpy() + np.exp(rate * tau) * (wide["C"] - wide["P"]).to_numpy()
    order = np.argsort(np.abs((wide["C"] - wide["P"]).to_numpy()))[:n_pairs]
    return float(np.median(f_k[order]))


def surface_from_chain(chain, rate, min_settle=MIN_SETTLE):
    """Build the smile for one session and one expiration.

    Parameters
    ----------
    chain : DataFrame
        Rows of the settlement table for ONE date and ONE expiration:
        columns date, expiration, days_to_expiry, strike, cp_flag, settle.
    rate : float
        Continuously compounded risk-free rate, as a decimal.

    Returns
    -------
    DataFrame in the layout ``rnd`` expects (date, secid, days_to_maturity,
    moneyness, implied_vol, spot_price, cp_flag) plus strike, settle, and
    forward. Only out-of-the-money contracts with a settlement above the
    minimum tick and a solvable implied volatility are kept.
    """
    chain = chain.copy()
    days = int(chain["days_to_expiry"].iloc[0])
    tau = days / 365.0
    forward = parity_forward(chain, rate, tau)
    spot = forward * np.exp(-rate * tau)  # the Black-76 identity

    otm = ((chain["cp_flag"] == "P") & (chain["strike"] <= forward)) | (
        (chain["cp_flag"] == "C") & (chain["strike"] > forward)
    )
    chain = chain[otm & (chain["settle"] > min_settle)].sort_values("strike")

    kinds = chain["cp_flag"].map({"P": "put", "C": "call"})
    chain["implied_vol"] = [
        bs.implied_volatility(price=s, S=spot, K=k, r=rate, T=tau, option_type=kind)
        for s, k, kind in zip(chain["settle"], chain["strike"], kinds)
    ]
    chain = chain.dropna(subset=["implied_vol"])

    out = pd.DataFrame(
        {
            "date": chain["date"].to_numpy(),
            "secid": SECID_SPX,
            "days_to_maturity": days,
            "moneyness": chain["strike"].to_numpy() / spot,
            "implied_vol": chain["implied_vol"].to_numpy(),
            "spot_price": spot,
            "cp_flag": chain["cp_flag"].to_numpy(),
            "strike": chain["strike"].to_numpy(),
            "settle": chain["settle"].to_numpy(),
            "forward": forward,
        }
    )
    return out.reset_index(drop=True)


def bracketing_expiries(session, horizon_days=HORIZON_DAYS, min_days=MIN_DAYS_NEAR):
    """The two expirations that bracket the horizon, as (near, far).

    Expirations closer than ``min_days`` are ignored. If the horizon falls
    before the first usable expiration, the first two are returned and the
    interpolation extrapolates, as the VIX does.
    """
    days = (
        session[["expiration", "days_to_expiry"]]
        .drop_duplicates()
        .sort_values("days_to_expiry")
    )
    days = days[days["days_to_expiry"] >= min_days]
    if len(days) < 2:
        raise ValueError("need at least two usable expirations")
    before = days[days["days_to_expiry"] <= horizon_days]
    after = days[days["days_to_expiry"] > horizon_days]
    if before.empty:
        near, far = days.iloc[0], days.iloc[1]
    elif after.empty:
        near, far = days.iloc[-2], days.iloc[-1]
    else:
        near, far = before.iloc[-1], after.iloc[0]
    return near["expiration"], far["expiration"]


def closest_expiry(session, horizon_days=HORIZON_DAYS, min_days=MIN_DAYS_NEAR):
    """The usable expiration whose days to expiry are closest to the horizon."""
    days = session[["expiration", "days_to_expiry"]].drop_duplicates()
    days = days[days["days_to_expiry"] >= min_days]
    if days.empty:
        raise ValueError("no usable expiration")
    idx = (days["days_to_expiry"] - horizon_days).abs().idxmin()
    return days.loc[idx, "expiration"]


def smile_for(options, date, expiration, rate):
    """The smile for one session and expiration from the settlement table."""
    chain = options[(options["date"] == pd.Timestamp(date)) & (options["expiration"] == pd.Timestamp(expiration))]
    if chain.empty:
        raise ValueError(f"no settlements for {date} / {expiration}")
    return surface_from_chain(chain, rate)
