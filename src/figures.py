"""The guides' interactive figures (Plotly).

Each function takes data a guide has already loaded (one session's
settlements, a smile, a risk-neutral CDF, a results table) and returns a
Plotly figure in one house style; the guide calls it and shows the result.
The numbers drawn come from the engine (``smile``, ``rnd``, ``svix``,
``crash_prob``) or from the pipeline's outputs. A few helpers for reading
the raw data (decoding a CME symbol, a future's expiration) live here too,
because the figures and tables of guide 01 are their only users.

Importing this module also sets Plotly's notebook renderer. The default
emits only Plotly's own MIME type, which ``nbconvert`` keeps and the
textbook's pages do not display; adding ``notebook_connected`` writes each
figure as HTML too, loading Plotly itself from its CDN.
"""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import black_scholes as bs
import crash_prob
import rnd
import smile
import svix

pio.renderers.default = "plotly_mimetype+notebook_connected"

BLUE, RED, GREEN, PURPLE, GREY = "#1f4e8c", "#c23b3b", "#2e8b57", "#7b4fa0", "#7a7a7a"
ORANGE = "#d9822b"
FONT = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=13)
X_WINDOW = (0.60, 1.30)  # gross returns shown in the distribution figures
MULTIPLIER = 50  # dollars per index point, for an E-mini future and its options
MONTH_CODES = dict(F=1, G=2, H=3, J=4, K=5, M=6, N=7, Q=8, U=9, V=10, X=11, Z=12)


def style(fig, height=480, **layout):
    """The house style; any layout key passed in overrides the default."""
    defaults = dict(template="plotly_white", height=height, font=FONT, margin=dict(l=60, r=20, t=60, b=40))
    fig.update_layout(**{**defaults, **layout})
    return fig


def percent(x):
    return 100 * np.asarray(x, dtype=float)


def _title(text):
    return dict(text=text, x=0, y=0.98)


# --------------------------------------------------------------------------
# Reading the raw data: CME symbols and the futures the options exercise into
# --------------------------------------------------------------------------

def third_friday(year, month):
    """The third Friday of a month: the last trading day of a quarterly ES future."""
    fridays = [week[calendar.FRIDAY] for week in calendar.monthcalendar(year, month) if week[calendar.FRIDAY]]
    return pd.Timestamp(year, month, fridays[2])


def decode_symbol(symbol, on):
    """Split a CME symbol into its parts, as of the session ``on``.

    ``ESZ6`` is root ``ES``, month code ``Z`` (December), year digit ``6``;
    ``EWV6 P7850`` adds a right (put) and a strike (7850). The year digit
    names the next year ending in that digit, counting the current one.
    """
    contract, _, option = symbol.partition(" ")
    root, code, digit = contract[:-2], contract[-2], int(contract[-1])
    year = pd.Timestamp(on).year // 10 * 10 + digit
    if year < pd.Timestamp(on).year:
        year += 10
    parts = {"symbol": symbol, "root": root, "month code": code,
             "contract month": f"{calendar.month_abbr[MONTH_CODES[code]]} {year}"}
    if option:
        parts["right"] = {"C": "call", "P": "put"}[option[0]]
        parts["strike"] = float(option[1:])
    return parts


def futures_curve(futures, on):
    """The session's ES futures, decoded, with each one's expiration and settlement.

    Also each contract's price relative to the nearest one, in percent: the
    shape of the curve without its level. The guides show that column and
    never the settlements themselves, which come under the course's
    Databento license.
    """
    rows = []
    for row in futures[futures["date"] == pd.Timestamp(on)].itertuples():
        parts = decode_symbol(row.symbol, on)
        year, month = int(parts["contract month"][-4:]), MONTH_CODES[parts["month code"]]
        expiration = third_friday(year, month)
        rows.append({**parts, "expiration": expiration,
                     "days to expiration": (expiration - pd.Timestamp(on)).days, "settlement": row.settle})
    curve = pd.DataFrame(rows).sort_values("expiration").reset_index(drop=True)
    curve["above the nearest (%)"] = 100 * (curve["settlement"] / curve["settlement"].iloc[0] - 1)
    return curve


def fig_futures_curve(curve, on, underlyings=()):
    """Every listed quarterly ES future, priced relative to the nearest one."""
    used = curve["symbol"].isin(list(underlyings))
    y = "above the nearest (%)"
    fig = go.Figure()
    for mask, name, color, size in [(~used, "ES future", GREY, 8), (used, "ES future that this session's options exercise into", BLUE, 11)]:
        c = curve[mask]
        fig.add_trace(go.Scatter(
            x=c["expiration"], y=c[y], mode="markers", name=name, marker=dict(color=color, size=size),
            customdata=np.column_stack([c["symbol"], c["contract month"]]),
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>expires %{x|%b %-d, %Y}<br>"
                          "%{y:.2f}% above the nearest contract<extra></extra>"))
    fig.add_trace(go.Scatter(x=curve["expiration"], y=curve[y], mode="lines", showlegend=False,
                             line=dict(color=GREY, width=1), hoverinfo="skip"))
    fig.update_layout(xaxis_title="Expiration (third Friday of the contract month)",
                      yaxis_title="Price above the nearest contract (%)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return style(fig, height=440, title=_title(f"The E-mini S&P 500 futures curve, settlements of {pd.Timestamp(on):%B %-d, %Y}"),
                 margin=dict(l=70, r=20, t=90, b=50))


def fig_expiration_calendar(session, curve, on, horizon_days=smile.HORIZON_DAYS):
    """Each option expiration on the row of the future it exercises into."""
    on = pd.Timestamp(on)
    expiries = (session.groupby("expiration")
                .agg(days=("days_to_expiry", "first"), contracts=("symbol", "size"), underlying=("underlying", "first"))
                .reset_index())
    near, far = smile.bracketing_expiries(session)
    futures = curve[curve["symbol"].isin(expiries["underlying"])]
    order = list(futures.sort_values("expiration")["symbol"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=futures["expiration"], y=futures["symbol"], mode="markers", name="The future's own expiration",
        marker=dict(symbol="diamond", size=13, color=GREY),
        hovertemplate="%{y} expires %{x|%b %-d, %Y}<extra></extra>"))
    bracket = expiries["expiration"].isin([near, far])
    for mask, name, color in [(~bracket, "An option expiration (end of month)", BLUE),
                              (bracket, "The two that bracket 30 days", RED)]:
        e = expiries[mask]
        fig.add_trace(go.Scatter(
            x=e["expiration"], y=e["underlying"], mode="markers", name=name,
            marker=dict(size=11, color=color, line=dict(color="white", width=1)),
            customdata=np.column_stack([e["days"], e["contracts"]]),
            hovertemplate="options expiring %{x|%b %-d, %Y}<br>%{customdata[0]} days away<br>"
                          "%{customdata[1]} contracts<br>exercise into %{y}<extra></extra>"))
    horizon = on + pd.Timedelta(days=horizon_days)
    fig.add_vline(x=horizon, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=horizon, y=1, yref="paper", text="30 days out", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(color=GREY, size=12))
    fig.update_layout(yaxis=dict(title="Underlying future", categoryorder="array", categoryarray=order),
                      xaxis_title="Expiration date",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return style(fig, height=380, title=_title(f"What expires when, as of {on:%B %-d, %Y}"),
                 margin=dict(l=90, r=20, t=90, b=50))


def fig_parity(chain, forward, rate):
    """Call minus put at each strike of one expiration, against put-call parity's line.

    Only the difference between the two settlements at a strike is drawn,
    never either settlement on its own.
    """
    calls = chain[chain["cp_flag"] == "C"].set_index("strike")["settle"]
    puts = chain[chain["cp_flag"] == "P"].set_index("strike")["settle"]
    diff = (calls - puts).dropna().sort_index()
    diff = diff[(diff.index >= 0.75 * forward) & (diff.index <= 1.15 * forward)]
    days = int(chain["days_to_expiry"].iloc[0])
    k = np.array([diff.index.min(), diff.index.max()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k, y=np.exp(-rate * days / 365) * (forward - k), mode="lines", name="Put-call parity: e<sup>−rτ</sup>(F − K)",
        line=dict(color=GREY, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=diff.index, y=diff.to_numpy(), mode="markers", name="Call minus put, at each listed strike",
        marker=dict(color=BLUE, size=5),
        hovertemplate="strike %{x:,.0f}<br>call minus put: %{y:,.2f} points<extra></extra>"))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.add_vline(x=forward, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=forward, y=1, yref="paper", showarrow=False, xanchor="left", yanchor="top",
                       text="the parity forward:<br>call and put cost the same", font=dict(color=GREY, size=12))
    fig.update_layout(xaxis=dict(title="Strike (index points)", range=[0.75 * forward, 1.15 * forward]),
                      yaxis_title="Call minus put (index points)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    expiry = pd.Timestamp(chain["expiration"].iloc[0])
    return style(fig, height=460, title=_title(f"Put-call parity, options expiring {expiry:%B %-d, %Y} ({days} days)"),
                 margin=dict(l=70, r=20, t=90, b=50))


def ten_delta_put(surface, rate):
    """Moneyness of the 10-delta put on this smile: the deepest point of a vendor's standard surface."""
    puts = rnd.otm_smile(surface, rate)
    puts = puts[puts["cp_flag"] == "P"].sort_values("moneyness")
    tau = int(surface["days_to_maturity"].iloc[0]) / 365.0
    k = puts["moneyness"].to_numpy(float)
    delta = np.abs([bs.bs_delta(1.0, strike, rate, vol, tau, option_type="put")
                    for strike, vol in zip(k, puts["implied_vol"].to_numpy(float))])
    return float(np.interp(0.10, np.maximum.accumulate(delta), k))


def fig_smile(chain, surface, rate):
    """The smile from settlement prices: what is priced, what sits at the floor, where a vendor stops."""
    spot = float(surface["spot_price"].iloc[0])
    forward = float(surface["forward"].iloc[0])
    k10 = ten_delta_put(surface, rate)
    otm = ((chain["cp_flag"] == "P") & (chain["strike"] <= forward)) | ((chain["cp_flag"] == "C") & (chain["strike"] > forward))
    floor = chain[otm & (chain["settle"] <= smile.MIN_SETTLE)]
    floor = floor[(floor["strike"] / spot > 0.4) & (floor["strike"] / spot < 1.4)]
    x_range = (0.45, 1.25)
    shown = surface[(surface["moneyness"] >= x_range[0]) & (surface["moneyness"] <= x_range[1])]
    fig = go.Figure()
    fig.add_vrect(x0=0.4, x1=k10, fillcolor=GREY, opacity=0.10, line_width=0)
    fig.add_annotation(x=x_range[0], y=1, yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text=f"Shaded: below the 10-delta put (k = {k10:.2f}),<br>where a vendor's standard surface has no strikes",
                       font=dict(color=GREY, size=12), align="left")
    for flag, name, color in [("P", "Out-of-the-money put", RED), ("C", "Out-of-the-money call", BLUE)]:
        s = surface[surface["cp_flag"] == flag]
        fig.add_trace(go.Scatter(
            x=s["moneyness"], y=percent(s["implied_vol"]), mode="markers", name=name, marker=dict(color=color, size=6),
            customdata=s["strike"],
            hovertemplate="strike %{customdata:,.0f} (k = %{x:.3f})<br>implied vol %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=floor["strike"] / spot, y=np.full(len(floor), 2.0), mode="markers", name="Settled at the 0.05 floor: not a price, dropped",
        marker=dict(symbol="x", color=GREY, size=6), customdata=floor["strike"],
        hovertemplate="strike %{customdata:,.0f}: settled at the minimum tick<extra></extra>"))
    fig.add_vline(x=0.80, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=0.80, y=0.06, yref="paper", text="a 20% drop", showarrow=False, xanchor="left",
                       font=dict(color=GREY, size=12))
    fig.update_layout(xaxis=dict(title="Moneyness k = strike / index level", range=list(x_range)),
                      yaxis=dict(title="Implied volatility (%)", range=[0, 1.15 * 100 * shown["implied_vol"].max()]),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    days = int(surface["days_to_maturity"].iloc[0])
    return style(fig, height=480, title=_title(f"The smile, from settlement prices ({days} days)"),
                 margin=dict(l=60, r=20, t=100, b=50))


def flat_extrapolation(surface, rate, alpha=0.80):
    """What a surface that stops at the 10-delta put assumes below it, against what the exchange settles.

    Returns the 10-delta put's moneyness, both volatilities at ``alpha``, the
    put struck at ``alpha`` priced both ways, in index points, and the ratio
    of the two prices. The guides show the volatilities and the ratio.
    """
    puts = rnd.otm_smile(surface, rate)
    puts = puts[puts["cp_flag"] == "P"]
    k10 = ten_delta_put(surface, rate)
    k, iv = puts["moneyness"].to_numpy(float), puts["implied_vol"].to_numpy(float)
    tau = int(surface["days_to_maturity"].iloc[0]) / 365.0
    spot = float(surface["spot_price"].iloc[0])
    vol_flat, vol_actual = float(np.interp(k10, k, iv)), float(np.interp(alpha, k, iv))
    price = lambda vol: spot * float(bs.bs_price(1.0, alpha, rate, vol, tau, option_type="put"))  # noqa: E731
    put_flat, put_actual = price(vol_flat), price(vol_actual)
    return {"k10": k10, "vol_flat": vol_flat, "vol_actual": vol_actual,
            "put_flat": put_flat, "put_actual": put_actual, "ratio": put_actual / put_flat}


def fig_flat_extrapolation(surface, rate):
    """The put wing as settled, against the volatility held flat below the vendor's last strike."""
    puts = rnd.otm_smile(surface, rate)
    puts = puts[(puts["cp_flag"] == "P") & (puts["moneyness"] >= 0.6)]
    k, iv = puts["moneyness"].to_numpy(float), percent(puts["implied_vol"])
    out = flat_extrapolation(surface, rate)
    k10, flat = out["k10"], 100 * out["vol_flat"]
    tail = k <= k10
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k[tail], y=np.full(tail.sum(), flat), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k[tail], y=iv[tail], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(194, 59, 59, 0.15)", name="What the flat assumption leaves out",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=iv, mode="markers", name="Exchange settlements (implied vol)",
                             marker=dict(color=RED, size=6), customdata=puts["strike"],
                             hovertemplate="strike %{customdata:,.0f} (k = %{x:.3f})<br>%{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0.6, k10], y=[flat, flat], mode="lines", name="A surface that stops at the 10-delta put, held flat",
                             line=dict(color=GREY, width=2.5, dash="dash"), hoverinfo="skip"))
    fig.add_vline(x=k10, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=k10, y=0.98, yref="paper", text=f"10-delta put, k = {k10:.2f}", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=GREY, size=12))
    fig.add_vline(x=0.80, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=0.80, y=0.05, yref="paper", text="a 20% drop", showarrow=False, xanchor="left",
                       font=dict(color=GREY, size=12))
    fig.update_layout(xaxis=dict(title="Moneyness k = strike / index level", range=[0.6, 1.0]),
                      yaxis_title="Implied volatility (%)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return style(fig, height=460, title=_title(
        f"Below the last strike: a 0.80 put costs {out['ratio']:.0f} times more as settled than under the flat assumption"),
        margin=dict(l=60, r=20, t=100, b=50))


def usable_expiries(session, min_days=smile.MIN_DAYS_NEAR, max_days=400):
    days = session[["expiration", "days_to_expiry"]].drop_duplicates().sort_values("days_to_expiry")
    return days[(days["days_to_expiry"] >= min_days) & (days["days_to_expiry"] <= max_days)]


def fig_surface(options, session, date, rate):
    """The implied-volatility surface, every usable expiration, on a common moneyness grid."""
    grid = np.linspace(0.70, 1.15, 91)
    rows, labels = [], []
    for row in usable_expiries(session).itertuples():
        try:
            otm = rnd.otm_smile(smile.smile_for(options, date, row.expiration, rate), rate)
        except ValueError:
            continue
        m, iv = otm["moneyness"].to_numpy(float), otm["implied_vol"].to_numpy(float)
        inside = (grid >= m.min()) & (grid <= m.max())  # no flat extrapolation
        rows.append(np.where(inside, np.interp(grid, m, iv), np.nan))
        labels.append(int(row.days_to_expiry))
    z = percent(np.vstack(rows))
    fig = go.Figure(go.Surface(
        x=grid, y=labels, z=z, colorscale="Viridis", colorbar=dict(title="Vol (%)", thickness=14),
        hovertemplate="Strike / index level %{x:.2f}<br>%{y} days<br>Implied vol %{z:.1f}%<extra></extra>",
    ))
    fig.update_scenes(
        xaxis_title="Strike / index level", yaxis_title="Days to expiration", zaxis_title="Implied vol (%)",
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)), aspectratio=dict(x=1.2, y=1.2, z=0.7),
    )
    return style(fig, height=600, title=f"The E-mini S&P 500 volatility surface, settlements of {pd.Timestamp(date):%B %-d, %Y}")


# --------------------------------------------------------------------------
# Guide 02: from option prices to a distribution, SVIX, and crash beliefs
# --------------------------------------------------------------------------

def smooth(values, grid, width=0.01):
    """A centered moving average over `width` of gross return, for display only.

    The density is the derivative of the CDF, and differentiating amplifies
    the kinks left by interpolating the smile between listed strikes.
    """
    step = float(np.median(np.diff(grid)))
    window = max(1, int(round(width / step)))
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def fig_distribution(options, session, date, rate, n_expiries=5):
    """Call prices, the risk-neutral CDF, and its density, one expiration at a time.

    The top panel is the call curve the engine prices on its fine grid, from
    the interpolated smile, not the exchange's settlements.
    """
    expiries = usable_expiries(session, max_days=200).head(n_expiries)
    first = int((expiries["expiration"] == smile.closest_expiry(session)).to_numpy().argmax())
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Call prices (% of the index level), priced from the interpolated smile",
                                        "Risk-neutral CDF: the probability of ending below each level",
                                        "Risk-neutral density (smoothed over 1% bands)"))
    traces_per = 3
    for i, row in enumerate(expiries.itertuples()):
        surface = smile.smile_for(options, date, row.expiration, rate)
        cdf = rnd.risk_neutral_cdf(surface, rate)
        keep = (cdf.grid >= X_WINDOW[0]) & (cdf.grid <= X_WINDOW[1])
        x = cdf.grid[keep]
        density = smooth(np.gradient(cdf.values, cdf.grid), cdf.grid)[keep]
        visible = i == first
        hover = "%{x:.2f}<br>%{y:.3g}<extra></extra>"
        fig.add_trace(go.Scatter(x=x, y=percent(cdf.call_grid[keep]), line=dict(color=BLUE), visible=visible,
                                 name="Call price", hovertemplate=hover), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=cdf.values[keep], line=dict(color=PURPLE), visible=visible,
                                 name="CDF", hovertemplate=hover), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=density, line=dict(color=RED), fill="tozeroy",
                                 fillcolor="rgba(194, 59, 59, 0.15)", visible=visible,
                                 name="Density", hovertemplate=hover), row=3, col=1)
    n = len(expiries) * traces_per
    buttons = []
    for i, row in enumerate(expiries.itertuples()):
        mask = [False] * n
        mask[i * traces_per:(i + 1) * traces_per] = [True] * traces_per
        buttons.append(dict(label=f"{int(row.days_to_expiry)} days ({row.expiration:%b %-d})",
                            method="update", args=[{"visible": mask}]))
    for r in (1, 2, 3):
        fig.add_vline(x=0.8, line_dash="dot", line_color=GREY, row=r, col=1)
    fig.add_annotation(x=0.8, y=1.0, xref="x3", yref="y3 domain", text="a 20% drop", showarrow=False,
                       xanchor="left", font=dict(color=GREY, size=12))
    fig.update_xaxes(title_text="Gross return: index level at expiration / today", row=3, col=1)
    fig.update_layout(showlegend=False, updatemenus=[dict(buttons=buttons, active=first, x=0, xanchor="left",
                                                          y=1.12, yanchor="bottom", showactive=True)])
    return style(fig, height=720, title=dict(text="From option prices to a distribution", x=0.5),
                 margin=dict(l=60, r=20, t=110, b=50))


def fig_svix_weights(cdf, date):
    """Out-of-the-money option prices by strike, weighted the SVIX way and the VIX way."""
    otm = np.where(cdf.grid <= cdf.forward, cdf.put_grid, cdf.call_grid)
    keep = (cdf.grid >= X_WINDOW[0]) & (cdf.grid <= X_WINDOW[1])
    x, svix_w, vix_w = cdf.grid[keep], otm[keep], (otm / cdf.grid**2)[keep]
    tau = cdf.maturity_years
    svix_level = svix.svix_index(svix.svix_squared(cdf))
    vix_style = 100 * np.sqrt(2 * cdf.forward * np.trapezoid(otm / cdf.grid**2, cdf.grid) / tau)
    share = lambda w: w / np.trapezoid(w, x)  # noqa: E731  percent of the total per 1% band of strikes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=share(svix_w), name=f"SVIX: every strike weighted equally ({svix_level:.1f})",
                             line=dict(color=BLUE, width=2.5), fill="tozeroy", fillcolor="rgba(31, 78, 140, 0.15)",
                             hovertemplate="%{x:.3f}<br>%{y:.1f}<extra>SVIX</extra>"))
    fig.add_trace(go.Scatter(x=x, y=share(vix_w), name=f"VIX-style: each strike divided by K² ({vix_style:.1f})",
                             line=dict(color=RED, width=2, dash="dash"),
                             hovertemplate="%{x:.3f}<br>%{y:.1f}<extra>VIX-style</extra>"))
    fig.add_vline(x=cdf.forward, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=cdf.forward, y=1, yref="paper", text="forward: puts to the left, calls to the right",
                       showarrow=False, xanchor="left", yanchor="top", font=dict(color=GREY, size=12))
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      xaxis_title="Strike / index level",
                      yaxis_title="Share of the total, per 1% band of strikes (%)")
    days = int(round(tau * 365))
    return style(fig, height=500, title=_title(f"Where SVIX comes from: {days}-day E-mini options, {pd.Timestamp(date):%B %-d, %Y}"),
                 margin=dict(l=60, r=20, t=90, b=50))


def fig_crash_beliefs(cdf, date, gammas=(1, 2, 3, 4, 5)):
    """Three probabilities of a crash, for every drop size, with risk aversion on a slider."""
    alphas = np.linspace(0.70, 0.98, 57)
    drops = percent(1 - alphas)
    rn = [crash_prob.risk_neutral_crash_prob(cdf, a) for a in alphas]
    log = [crash_prob.crash_probability(cdf, a) for a in alphas]
    fig = go.Figure()
    hover = "a %{x:.0f}%% drop<br>%{y:.3f}%%<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=drops, y=percent(rn), name="Risk-neutral (what the market charges)",
                             line=dict(color=RED, width=2.5), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=drops, y=percent(log), name="Log investor (Martin 2017)",
                             line=dict(color=GREEN, width=2.5), hovertemplate=hover))
    for g in gammas:
        power = [crash_prob.crash_probability_power(cdf, a, gamma=g) for a in alphas]
        fig.add_trace(go.Scatter(x=drops, y=percent(power), name=f"Power utility, risk aversion {g}",
                                 line=dict(color=PURPLE, width=2.5, dash="dot"), visible=(g == 2),
                                 hovertemplate=hover))
    steps = []
    for i, g in enumerate(gammas):
        visible = [True, True] + [j == i for j in range(len(gammas))]
        steps.append(dict(label=str(g), method="update", args=[{"visible": visible}]))
    fig.add_vline(x=20, line_dash="dot", line_color=GREY)
    fig.update_layout(
        sliders=[dict(active=list(gammas).index(2), steps=steps, x=0.05, len=0.9, y=-0.12,
                      currentvalue=dict(prefix="Power utility's risk aversion: ", font=dict(size=13)))],
        xaxis_title="Size of the drop within the month (%)", yaxis_title="Probability (%), log scale",
        yaxis=dict(type="log", tickvals=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
                   ticktext=["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30"]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
    )
    days = int(round(cdf.maturity_years * 365))
    return style(fig, height=560, title=_title(f"Three beliefs about a crash, {days}-day options, {pd.Timestamp(date):%B %-d, %Y}"),
                 margin=dict(l=70, r=20, t=90, b=110))


# --------------------------------------------------------------------------
# Guide 04: thirty years from the OptionMetrics surface (derived series only)
# --------------------------------------------------------------------------

EVENTS = [("1998-09-15", "LTCM, 1998"), ("2001-09-25", "9/11"), ("2008-11-20", "Lehman, 2008"),
          ("2011-08-15", "U.S. downgrade, 2011"), ("2015-08-25", "China, 2015"), ("2020-03-20", "COVID, 2020")]


def fig_single_names(bounds, paper_end="2022-12-31"):
    """Martin and Shi's bounds for each name, with the risk-neutral probability; the paper's Figure 1."""
    bounds = bounds[(bounds["days_to_maturity"] == 30) & (bounds["threshold"] == 0.8)].sort_values("date")
    tickers = sorted(bounds["ticker"].unique())
    fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{t}: the probability of a 20% drop within a month" for t in tickers])
    for r, ticker in enumerate(tickers, start=1):
        b = bounds[bounds["ticker"] == ticker]
        show = r == 1
        fig.add_trace(go.Scatter(x=b["date"], y=percent(b["bound_upper"]), line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), row=r, col=1)
        fig.add_trace(go.Scatter(x=b["date"], y=percent(b["bound_lower"]), fill="tonexty",
                                 fillcolor="rgba(31, 78, 140, 0.25)", line=dict(width=0),
                                 name="Martin-Shi bounds on the real-world probability", showlegend=show,
                                 customdata=np.column_stack([percent(b["bound_lower"]), percent(b["bound_upper"])]),
                                 hovertemplate="%{x|%b %Y}<br>between %{customdata[0]:.1f}% and "
                                               "%{customdata[1]:.1f}%<extra></extra>"), row=r, col=1)
        fig.add_trace(go.Scatter(x=b["date"], y=percent(b["prob_riskneutral"]), name="Risk-neutral",
                                 line=dict(color=RED, width=1.2), showlegend=show,
                                 hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>risk-neutral</extra>"), row=r, col=1)
        fig.update_yaxes(title_text="%", row=r, col=1)
    axes = [f"xaxis{'' if r == 1 else r}" for r in range(1, len(tickers) + 1)]
    paper = {f"{a}.range": ["1996-01-01", paper_end] for a in axes}
    everything = {f"{a}.autorange": True for a in axes}
    fig.update_layout(
        legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0),
        updatemenus=[dict(type="buttons", direction="right", x=1, xanchor="right", y=1.10, yanchor="bottom",
                          buttons=[dict(label="All", method="relayout", args=[everything]),
                                   dict(label=f"The paper's sample, 1996-{paper_end[:4]}", method="relayout",
                                        args=[paper])])])
    return style(fig, height=640, title=_title("Single names: bounds on the real-world probability (Martin and Shi 2025)"),
                 margin=dict(l=60, r=20, t=120, b=30))


def fig_crash_history(index, martin):
    """The S&P 500's probability of a 20% drop within a month, daily, three ways; Martin's monthly points."""
    idx = index[(index["days_to_maturity"] == 30) & (index["threshold"] == 0.8)].sort_values("date")
    martin = martin.copy()
    martin["date"] = pd.to_datetime(dict(year=martin["year"].astype(int), month=martin["month"].astype(int), day=15))
    fig = go.Figure()
    for col, name, color, width in [("prob_riskneutral", "Risk-neutral", RED, 1),
                                    ("prob_log_utility", "Log investor (Martin 2017)", GREEN, 1.2),
                                    ("prob_power_utility", "Power utility, risk aversion 2 (Martin and Shi 2025)", PURPLE, 1)]:
        fig.add_trace(go.Scatter(x=idx["date"], y=percent(idx[col]), name=name, line=dict(color=color, width=width),
                                 opacity=0.85, hovertemplate="%{x|%b %-d, %Y}<br>%{y:.2f}%<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=martin["date"], y=percent(martin["m1"]), name="Martin's published log-investor series (monthly)",
                             mode="markers", marker=dict(color="black", size=4),
                             hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>Martin</extra>"))
    series = idx.set_index("date")["prob_log_utility"]
    for when, label in EVENTS:
        window = series[(series.index > pd.Timestamp(when) - pd.DateOffset(days=45)) &
                        (series.index < pd.Timestamp(when) + pd.DateOffset(days=45))]
        if window.empty:
            continue
        peak = window.idxmax()
        fig.add_annotation(x=peak, y=100 * window.max(), text=label, showarrow=True, arrowhead=0, ax=0, ay=-28,
                           font=dict(size=11, color="#333"))
    fig.update_layout(
        yaxis_title="Probability of a 20% drop within a month (%)",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        xaxis=dict(rangeselector=dict(buttons=[dict(count=5, label="5Y", step="year", stepmode="backward"),
                                               dict(count=10, label="10Y", step="year", stepmode="backward"),
                                               dict(label="All", step="all")], x=1, xanchor="right", y=1.10,
                                      yanchor="bottom"),
                   rangeslider=dict(visible=True, thickness=0.06)),
    )
    first, last = idx["date"].min().year, idx["date"].max().year
    return style(fig, height=580, title=_title(f"Thirty years of crash fear: the S&P 500, {first}-{last}"),
                 margin=dict(l=60, r=20, t=120, b=30))


def fig_svix_check(df):
    """Our SVIX from the OptionMetrics surface against Martin's published series, with the gap."""
    corr = df["svix"].corr(df["svix_martin"])
    gap = (df["svix"] - df["svix_martin"]).abs().median()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.75, 0.25])
    fig.add_trace(go.Scatter(x=df["date"], y=df["svix_martin"], name="Martin's published SVIX",
                             line=dict(color=GREY, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["svix"], name="Ours, from the same vendor's surface",
                             line=dict(color=BLUE, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["svix"] - df["svix_martin"], name="Ours minus Martin's",
                             line=dict(color=GREY, width=0.8), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=0.6, row=2, col=1)
    fig.update_yaxes(title_text="SVIX, 30 days (%)", row=1, col=1)
    fig.update_yaxes(title_text="Gap (vol pts)", row=2, col=1)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return style(fig, height=560, title=_title(f"The replication check: correlation {corr:.4f}, median gap {gap:.2f} vol points"),
                 margin=dict(l=60, r=20, t=90, b=30))
