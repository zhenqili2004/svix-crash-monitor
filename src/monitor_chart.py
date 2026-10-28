"""Render the monitor's charts from the history.

Two formats, as in the FedWatch case study: a static PNG (matplotlib) and an
interactive HTML (plotly) for the ChartBook site. The Plotly figure is also
the chart on the monitor's page (``build_page``). All computation lives in
``monitor``; this module only draws.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from monitor import load_history
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
PNG = OUTPUT_DIR / "svix_monitor.png"
HTML = OUTPUT_DIR / "svix_monitor.html"


def plot_matplotlib(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(history["date"], history["svix_30"], label="SVIX (30-day, from E-mini settlements)", color="#1f77b4")
    if "vix" in history:
        ax1.plot(history["date"], history["vix"], label="VIX (Cboe, from FRED)", color="#7f7f7f", lw=1)
    ax1.set_ylabel("Annualized volatility, %")
    ax1.legend(frameon=False)
    ax1.set_title("SVIX and the probability of a 20% drop in the S&P 500 within a month")
    ax2.plot(history["date"], 100 * history["p_star_20"], label="Risk-neutral, P*", color="#d62728")
    ax2.plot(history["date"], 100 * history["p_log_20"], label="Log investor (Martin 2017)", color="#2ca02c")
    ax2.plot(history["date"], 100 * history["p_power_20"], label="Power utility, gamma = 2 (Martin and Shi 2025)", color="#9467bd")
    ax2.set_ylabel("Probability, %")
    ax2.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_plotly(history):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Volatility indexes", "Probability of a 20% drop within a month"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["svix_30"], name="SVIX (30-day)", mode="lines"), row=1, col=1)
    if "vix" in history:
        fig.add_trace(go.Scatter(x=history["date"], y=history["vix"], name="VIX", mode="lines", line=dict(color="gray", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=100 * history["p_star_20"], name="Risk-neutral P*", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=100 * history["p_log_20"], name="Log investor (Martin 2017)", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=100 * history["p_power_20"], name="Power utility, gamma=2", mode="lines"), row=2, col=1)
    fig.update_yaxes(title_text="Annualized vol, %", row=1, col=1)
    fig.update_yaxes(title_text="Probability, %", row=2, col=1)
    last = history.iloc[-1]
    fig.update_layout(
        title=f"SVIX and the probability of a crash, as of {last['date']:%Y-%m-%d}",
        height=650, template="plotly_white", legend=dict(orientation="h", y=-0.12),
    )
    return fig


if __name__ == "__main__":
    history = load_history()
    if history.empty:
        raise SystemExit("no history yet: run `doit` or `doit latest` first")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_matplotlib(history).savefig(PNG, dpi=150, bbox_inches="tight")
    plot_plotly(history).write_html(HTML, include_plotlyjs="cdn")
    print(f"Wrote {PNG.name} and {HTML.name} ({len(history)} sessions)")
