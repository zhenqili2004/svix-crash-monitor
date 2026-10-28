"""Tests for the smile construction from a settlement chain."""

import numpy as np
import pandas as pd

import black_scholes as bs
from smile import bracketing_expiries, closest_expiry, parity_forward, surface_from_chain


def synthetic_chain(date="2026-10-15", days=30, forward=6700.0, vol=0.18, rate=0.04):
    tau = days / 365.0
    spot = forward * np.exp(-rate * tau)
    strikes = np.arange(5000, 8400, 100.0)
    rows = []
    for k in strikes:
        for cp, kind in [("C", "call"), ("P", "put")]:
            price = bs.bs_price(spot, k, rate, vol, tau, option_type=kind)
            rows.append(dict(date=pd.Timestamp(date), expiration=pd.Timestamp(date) + pd.Timedelta(days=days),
                             days_to_expiry=days, strike=k, cp_flag=cp, settle=round(price, 2)))
    return pd.DataFrame(rows)


def test_parity_forward_recovers_the_forward():
    chain = synthetic_chain()
    assert abs(parity_forward(chain, 0.04, 30 / 365) - 6700.0) < 1.0


def test_surface_recovers_a_flat_vol():
    surface = surface_from_chain(synthetic_chain(vol=0.18), rate=0.04)
    assert set(surface["cp_flag"]) == {"P", "C"}
    assert (surface["moneyness"] > 0).all()
    near_money = surface[(surface["moneyness"] > 0.9) & (surface["moneyness"] < 1.1)]
    assert np.allclose(near_money["implied_vol"], 0.18, atol=5e-3)


def test_bracketing_and_closest_expiries():
    base = pd.Timestamp("2026-10-15")
    session = pd.DataFrame({
        "expiration": [base + pd.Timedelta(days=d) for d in (3, 15, 46, 77)],
        "days_to_expiry": [3, 15, 46, 77],
    })
    near, far = bracketing_expiries(session)
    assert (near, far) == (base + pd.Timedelta(days=15), base + pd.Timedelta(days=46))
    assert closest_expiry(session) == base + pd.Timedelta(days=15)
