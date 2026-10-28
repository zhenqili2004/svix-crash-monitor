"""Build the monitor's page, `_output/monitor_page/index.html`, from the history.

One HTML file: the latest reading, the chart from ``monitor_chart``, the
last ten sessions, and how the page is made. It reads only
`_output/svix_monitor_history.csv`, never Databento, so it can be rebuilt
from the history alone. Like the history, it shows only what the pipeline
computes from the settlements, SVIX and the probabilities of a crash, and
never the settlement prices themselves, which come under the course's
Databento license.

The deploy workflow publishes this folder on the `gh-pages` branch.
"""

import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from monitor import load_history
from monitor_chart import plot_plotly
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
PAGE_DIR = OUTPUT_DIR / "monitor_page"
PAGE = PAGE_DIR / "index.html"


def make_figure(history):
    """The monitor's chart, without its title: the page has a heading."""
    fig = plot_plotly(history)
    fig.update_layout(title_text="", height=620, margin=dict(l=60, r=20, t=40, b=10),
                      font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=13))
    return fig


def nearest_before(history, when):
    earlier = history[history["date"] <= when]
    return earlier.iloc[-1] if not earlier.empty else None


def pct(p):
    return f"{100 * p:.2f}%"


def vol(x):
    return "&ndash;" if pd.isna(x) else f"{x:.1f}"


def reading_text(history):
    last = history.iloc[-1]
    lines = [
        f"The options market prices a 20% drop in the S&P 500 over the next {int(last['days_used'])} days "
        f"at {pct(last['p_star_20'])} (risk-neutral). A log investor who holds the market reads that as "
        f"{pct(last['p_log_20'])}; a more risk-averse investor (power utility, &gamma; = 2), as {pct(last['p_power_20'])}."
    ]
    if not pd.isna(last["vix"]):
        lines.append(f"SVIX is {last['svix_30']:.1f}, against a VIX of {last['vix']:.1f}.")
    changes = []
    for label, offset in [("a week", pd.DateOffset(weeks=1)), ("a month", pd.DateOffset(months=1))]:
        then = nearest_before(history, last["date"] - offset)
        if then is not None:
            change = round(last["svix_30"] - then["svix_30"], 1) + 0.0  # + 0.0 turns -0.0 into 0.0
            changes.append(f"{change:+.1f} over {label}")
    if changes:
        lines.append("Change in SVIX: " + "; ".join(changes) + ".")
    return last, lines


def build_html(history):
    last, lines = reading_text(history)
    repo = os.environ.get("GITHUB_REPOSITORY")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID")
    now = datetime.now(timezone.utc)
    updated = f"{now:%B %-d, %Y at %H:%M} UTC"
    if repo and run_id:
        updated = f'<a href="{server}/{repo}/actions/runs/{run_id}">{updated}</a>, by a GitHub Actions run'
    source = f' Code and history: <a href="{server}/{repo}">{escape(repo)}</a>.' if repo else ""
    recent = history.tail(10).iloc[::-1]
    rows = "\n".join(
        f"<tr><td>{r.date:%a, %b %-d, %Y}</td><td class='num'>{r.svix_30:.1f}</td><td class='num'>{vol(r.vix)}</td>"
        f"<td class='num'>{pct(r.p_star_20)}</td><td class='num'>{pct(r.p_log_20)}</td>"
        f"<td class='num'>{pct(r.p_power_20)}</td></tr>"
        for r in recent.itertuples()
    )
    chart = make_figure(history).to_html(full_html=False, include_plotlyjs="cdn",
                                         config={"displaylogo": False, "responsive": True})
    return TEMPLATE.format(
        p_log=pct(last["p_log_20"]), svix=f"{last['svix_30']:.1f}", as_of=f"{last['date']:%B %-d, %Y}",
        reading="".join(f"<p>{line}</p>" for line in lines),
        chart=chart, first=f"{history['date'].iloc[0]:%B %-d, %Y}", sessions=f"{len(history):,}", rows=rows,
        updated=updated, source=source,
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVIX and the Probability of a Crash</title>
<style>
  :root {{ --ink: #1b1f24; --muted: #5b6470; --rule: #e3e6ea; --accent: #1d4f91; --bg: #ffffff; --panel: #f6f8fa; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }}
  .eyebrow {{ color: var(--accent); font-weight: 600; font-size: 13px; letter-spacing: .04em; text-transform: uppercase; margin: 0 0 6px; }}
  h1 {{ font-size: 32px; line-height: 1.2; margin: 0 0 8px; }}
  h2 {{ font-size: 19px; margin: 36px 0 10px; }}
  .lede {{ color: var(--muted); font-size: 18px; margin: 0 0 24px; }}
  .reading {{ display: flex; gap: 24px; align-items: center; background: var(--panel);
             border: 1px solid var(--rule); border-radius: 10px; padding: 18px 22px; margin-bottom: 22px; }}
  .stats {{ display: flex; gap: 18px; }}
  .stat {{ text-align: center; min-width: 170px; }}
  .stat .value {{ display: block; font-size: 40px; font-weight: 700; color: var(--accent); line-height: 1; }}
  .stat .unit {{ display: block; color: var(--muted); font-size: 13px; margin-top: 6px; }}
  .date {{ display: block; font-weight: 600; font-size: 14px; margin-top: 4px; text-align: center; }}
  .reading p {{ margin: 4px 0; }}
  figure {{ margin: 0; }}
  figcaption {{ color: var(--muted); font-size: 14px; border-top: 1px solid var(--rule); padding-top: 10px; margin-top: 4px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; min-width: 560px; font-size: 15px; }}
  td, th {{ padding: 5px 14px 5px 0; border-bottom: 1px solid var(--rule); text-align: left; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  ol {{ padding-left: 22px; }} li {{ margin: 6px 0; }}
  code {{ background: var(--panel); padding: 1px 5px; border-radius: 4px; font-size: 14px; }}
  a {{ color: var(--accent); }}
  footer {{ margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 13px; }}
  footer p {{ margin: 6px 0; }}
  @media (max-width: 640px) {{ .reading {{ flex-direction: column; text-align: center; }} h1 {{ font-size: 26px; }} }}
</style>
</head>
<body>
<main>
  <p class="eyebrow">Daily monitor &middot; updated by GitHub Actions</p>
  <h1>SVIX and the probability of a crash</h1>
  <p class="lede">Ian Martin's SVIX index and the option-implied probability of a 20% drop in the S&amp;P 500
  within a month, computed every morning from the settlement prices of E-mini S&amp;P 500 options.</p>

  <section class="reading" aria-label="Latest reading">
    <div>
      <div class="stats">
        <div class="stat"><span class="value">{p_log}</span><span class="unit">probability of a 20% drop,<br>log investor</span></div>
        <div class="stat"><span class="value">{svix}</span><span class="unit">SVIX, 30-day<br>(annualized %)</span></div>
      </div>
      <span class="date">{as_of}</span>
    </div>
    <div>{reading}</div>
  </section>

  <figure>
    {chart}
    <figcaption><strong>Figure 1.</strong> Top: the 30-day SVIX next to the Cboe VIX, both annualized volatilities
    in percent. Bottom: the probability of a 20% drop in the S&amp;P 500 by the option expiration closest to one
    month, under three sets of beliefs: risk-neutral (what the market charges), a log investor who holds the
    market (Martin 2017), and a power-utility investor with risk aversion 2 (Martin and Shi 2025). {sessions}
    sessions since {first}. Hover for the values on any day.</figcaption>
  </figure>

  <h2>Recent sessions</h2>
  <div class="table-wrap">
  <table>
    <thead><tr><th>Session</th><th class="num">SVIX</th><th class="num">VIX</th><th class="num">Risk-neutral</th>
    <th class="num">Log investor</th><th class="num">&gamma; = 2</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  </div>

  <h2>How this page is made</h2>
  <ol>
    <li>Every morning at 14:30 UTC, once CME's settlement prices from the evening before have reached Databento,
        a GitHub Actions workflow pulls the latest session of E-mini S&amp;P 500 option settlements, with a
        Databento API key stored as a repository secret.</li>
    <li>It computes SVIX and the three probabilities for that session and adds the row to the history: one CSV,
        kept on the repository's <code>monitor-history</code> branch, one small commit a day. The
        <code>main</code> branch holds only code.</li>
    <li>It rebuilds this page from the history and publishes it on the <code>gh-pages</code> branch, which
        GitHub Pages serves.</li>
  </ol>
  <p>Last updated {updated}.{source}</p>

  <footer>
    <p><strong>Sources:</strong> end-of-month E-mini S&amp;P 500 options (CME Globex, root <code>EW</code>), daily
    settlement prices, from Databento; this page shows only quantities computed from them, never the prices.
    The Cboe VIX (VIXCLS) and the one-month Treasury yield (DGS1MO), from FRED, Federal Reserve Bank of St. Louis.</p>
    <p><strong>Methods:</strong> Ian Martin (2017), &ldquo;What is the Expected Return on the Market?&rdquo;
    <em>Quarterly Journal of Economics</em> 132(1); Ian Martin and Ran Shi (2025), &ldquo;Forecasting Crashes with
    a Smile.&rdquo; The engine that recovers the risk-neutral distribution is the work of Riley Haas and Marija
    Jovicic.</p>
    <p>Built for FINM 32800, Data Pipelines for Quantitative Research, University of Chicago.</p>
  </footer>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    history = load_history()
    if history.empty:
        raise SystemExit("No history yet: run `doit` (the backfill) or `doit latest` first.")
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(build_html(history))
    print(f"Wrote {PAGE.relative_to(OUTPUT_DIR.parent)} ({len(history)} sessions, latest {history['date'].iloc[-1]:%Y-%m-%d})")
