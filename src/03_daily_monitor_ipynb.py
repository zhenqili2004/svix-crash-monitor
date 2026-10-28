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
# # 03. The Daily Monitor
#
# ```{admonition} In this guide
# :class: tip
# **A monitor is a replication that runs itself.** It adds five things to
# the pipeline: a schedule, a secret, a locked environment, a ledger, and a
# deploy.
#
# **The history is the one output kept between runs.** Every other output
# is rebuilt from source; the history would cost a pull per day to rebuild,
# so it lives on a branch of its own, one small commit a day.
#
# **An outage is not a bug.** A data source that is down ends the run with a
# warning and leaves yesterday's page up; any other failure fails the run.
# ```
#
# Guide 02 computed SVIX and the probability of a crash for one session.
# This page is what happens when a replication is run every day without
# anyone at the keyboard. A GitHub Actions workflow, `.github/workflows/deploy_pages.yml`,
# wakes up at 14:30 UTC, pulls the previous session's settlements from
# Databento, computes the row, appends it to the history, records the
# history on a branch of its own, rebuilds the monitor's page, and publishes
# it. The chart below, the one on that page, is one point longer every
# morning.
#
# **Sections**
#
# - The history, and the latest row.
# - What the workflow does, step by step, and the places it can go
#   wrong.
# - Why the history lives on its own branch.

# %%
import pandas as pd
from IPython.display import HTML

import figures  # noqa: F401  sets the notebook renderer for Plotly
import monitor
import monitor_chart

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

history = monitor.load_history()
print(f"{len(history)} sessions, {history['date'].min().date()} to {history['date'].max().date()}")

# %% [markdown]
# ## The history
#
# One row per session. The columns are documented in `src/monitor.py`; the
# ones the chart uses are the 30-day SVIX (`svix_30`), the VIX close for
# the same day from FRED, and the three probabilities of a 20% drop over
# the expiration closest to one month: risk-neutral (`p_star_20`), the log
# investor's (`p_log_20`), and the power-utility investor's with
# $\gamma = 2$ (`p_power_20`).

# %%
cols = ["date", "days_used", "svix_30", "vix", "p_star_20", "p_log_20", "p_power_20", "p_star_10", "p_log_10"]
history[cols].tail(10).set_index("date").style.format({
    "svix_30": "{:.2f}", "vix": "{:.2f}",
    "p_star_20": "{:.2%}", "p_log_20": "{:.2%}", "p_power_20": "{:.2%}", "p_star_10": "{:.2%}", "p_log_10": "{:.2%}",
})

# %%
monitor_chart.plot_plotly(history).show()

# %% [markdown]
# This is the chart the monitor publishes, drawn by `src/monitor_chart.py`
# from the history. `src/build_page.py` puts it on a single HTML page, with
# the latest reading and the last ten sessions, and that page is the whole
# published site. Hover for any day's values.
#
# **Takeaway.** Both panels move with the smile of guide 01, and the
# risk-neutral probability is always the largest of the three. SVIX sits a
# little below the VIX. Both are annualized volatilities read
# off the same options, but the VIX weights each strike by $1/K^2$ and
# SVIX weights them equally, so the VIX leans harder on the put wing; the
# gap is a measure of how steep the smirk is. The probabilities move with
# the smile too, and the risk-neutral one is always the largest: it charges
# for crash states at their marginal utility.

# %% [markdown]
# ## What the workflow does
#
# `uv run doit latest` is the whole update. The task pulls the latest
# session (cheap: 19 MB, $0.00, a minute or two), computes its row if the
# history does not have it, appends it, and rebuilds the page from the
# history, `_output/monitor_page/index.html`. Run it on your laptop and it
# does the same thing; open the page in a browser to see it. Plain `uv run doit` is the same
# computation over the window in the settings, the trailing week unless you
# set START_DATE earlier, so it is also the backfill that gives the chart a
# history behind it; on Actions a second workflow does that, started from
# the Actions tab with a start date. The daily workflow adds the parts that
# make the update unattended:
#
# 1. **A schedule.** A `cron` line under `on: schedule:` starts the run
#    every morning, after CME's final settlements of the previous evening
#    have reached Databento. GitHub's cron runs in UTC; the five fields are
#    minute, hour, day of month, month, and day of week.
# 2. **A secret.** The pull needs your Databento key. It lives in the
#    repository's secret store (Settings → Secrets and variables → Actions),
#    and the step that needs it reads it from there through the `secrets`
#    context. It is never in the file, never in a commit, never in the log.
# 3. **An environment.** `astral-sh/setup-uv` installs `uv`; `uv run doit`
#    then installs Python 3.12 and the locked dependencies from `uv.lock`,
#    the same versions as on your laptop.
# 4. **A ledger.** The updated history is committed to the `monitor-history`
#    branch from a second working directory (`git worktree`): one small
#    commit a day, the day's row. A push to that branch does not start the
#    workflow, which listens to pushes on `main` only.
# 5. **A deploy.** The page is force-pushed as a single orphan commit on the
#    `gh-pages` branch, which GitHub Pages serves. Only the page: the
#    notebooks and the ChartBook site stay on your laptop.
#
# Three places it goes wrong, in our experience. A workflow that commits
# its output to `main` retriggers itself on every run, forever, unless each
# push says `[skip ci]`; keeping output off `main` removes the problem. A
# deploy that does not `git clean -fdx` first carries `_data/` and `.venv/`
# into the published branch. And a data source can be down: Databento and FRED
# both time out now and then. The pulls retry server and network errors,
# waiting longer each time, for about eight minutes. If a source is still
# down after that, the step after the pull sees the marker the pull left
# and ends the run with a warning, not a failure: yesterday's page stays
# up, and the next morning's pull, which covers the last four calendar days,
# fills the gap.
# Any other failure is a bug and fails the run.
#
# ```{admonition} Stop and discuss
# :class: note
# Databento is down for three days. What does the run look like on each of
# those mornings, what does the published chart show, and what happens on
# the fourth morning?
# ```

# %% [markdown]
# ## Why the history lives on its own branch
#
# Everything else the pipeline produces is regenerated from source: pull
# again, compute again, same answer. The history is different. Each row
# costs a session's pull, and a year of them would cost a year of pulls, so
# the monitor keeps what it has computed. Not on `main`, though: `main`
# holds source code, and nothing a run produces belongs there. The workflow
# keeps the one file, `svix_monitor_history.csv`, on a branch of its own,
# `monitor-history`. Each morning it restores the file from there into
# `_output/`, appends the new sessions, and commits the result back as one
# small commit, so the branch is the ledger of the series, one line a day,
# and `git log -p` on it shows the chart being built point by point. The
# page itself sits on `gh-pages` as a single commit that each run replaces.
# The daily run adds one session. The history behind it comes from a
# backfill, the same computation over a range of dates, run once by hand;
# if the engine ever changes, the same backfill rebuilds the history.
#
# ```{admonition} Stop and discuss
# :class: note
# The runner starts empty every morning. Where does yesterday's row come
# from? Why not keep the history on `main`, on `gh-pages`, or in the Actions
# cache?
# ```

# %%
latest = history.iloc[-1]
HTML(
    f"<p>As of <b>{latest['date']:%A, %B %d, %Y}</b>: SVIX {latest['svix_30']:.1f}, VIX {latest['vix']:.1f}; "
    f"the options market prices a 20% drop over the next {int(latest['days_used'])} days at "
    f"<b>{100 * latest['p_star_20']:.2f}%</b> risk-neutral, which a log investor who holds the market reads as "
    f"<b>{100 * latest['p_log_20']:.2f}%</b>.</p>"
)

# %% [markdown]
# ```{admonition} Notes on the discussion questions
# **Three days without Databento.** Each of those mornings the pull retries
# for about eight minutes, leaves the `SOURCE_UNAVAILABLE` marker, and the
# run ends with a warning: no row is recorded and the page is not
# redeployed, so the published chart stays as it was, short of the latest
# sessions. On the fourth morning the daily pull covers the last four
# calendar days, so on weekdays it picks up all three missed sessions and the
# history is whole again. A longer outage, or one that spans a weekend,
# leaves sessions outside that window; the backfill workflow, run once with a
# start date, fills them.
#
# **Where yesterday's row comes from.** From the `monitor-history` branch:
# the restore step fetches it and copies the CSV into `_output/` before the
# update runs. Not `main`: it holds source code, and a workflow that pushed
# output to `main` would trigger itself on every run. Not `gh-pages`: each
# run replaces it with one fresh commit, so anything kept there is wiped
# daily. Not the Actions cache: GitHub evicts a cache that goes unused for a
# week, and a cache has no history, so losing it means paying for every pull
# again. A branch is durable, free, and versioned: `git log -p` on it is the
# series, one row a day.
# ```
