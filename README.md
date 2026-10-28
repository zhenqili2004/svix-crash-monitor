# Homework 3: SVIX and the Probability of a Crash

**Due: Tuesday, November 10, at the start of class.** Also posted on Canvas.

This README is the primary source of truth for the assignment. The
[course textbook](https://finm-32800.github.io/HW3.html) has the supporting
notes and the HW guides.

You will build three things:

- **A replication of Ian Martin's SVIX index** and of his option-implied
  probability of a 20% drop in the S&P 500 (Martin 2017, *What is the
  Expected Return on the Market?*), from the settlement prices of E-mini
  S&P 500 options pulled from Databento.
- **A daily monitor**: a GitHub Actions workflow that reruns the
  replication every morning and publishes the result, one page with a
  chart, to GitHub Pages.
- **Crash bounds for Apple and AIG** (Martin and Shi 2025, *Forecasting
  Crashes with a Smile*), from the OptionMetrics volatility surface on WRDS.

## How to work on this

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once.
Then, from the root of this repository:

```bash
cp .env.example .env     # then add your Databento key and WRDS username
uv run doit
```

`uv run` installs Python 3.12 and the locked dependencies into `.venv/` the
first time and runs the command there. There is no environment to create or
activate. In VS Code, select the interpreter in `.venv/`. (To keep using the
`finm` conda environment instead, `pip install -r requirements.txt` installs
the same versions.)

Work in this order:

1. **Run `uv run pytest`.** Most tests run offline and need no data. Read
   the first failing test and write the code that passes it. The tests are
   the specification: when you are unsure what a function or a task should
   produce, read its test in `src/test_*.py`.
2. **Run `uv run doit`.** It pulls a week of data from Databento and FRED
   (free, about ten minutes) and stops at the first step that depends on a
   blank you have not filled. That is expected. The index part is done when
   `uv run doit` finishes without an error and a second run reports that
   every task is up to date.
3. **Then deploy** (Part C). Wait until step 2 is green: every run of the
   workflow takes five minutes, and the first ones fail anyway.

**Search for `TODO` to find every blank.** They are in `src/svix.py`,
`src/rnd.py`, `src/crash_prob.py`, `src/pull_optionmetrics.py`, `dodo.py`,
`chartbook.toml`, and `.github/workflows/deploy_pages.yml`. You also fill in
`src/github_skills.py` and `src/published_site.py`.

**Do not edit the test files.** If a `test_*.py` file is modified, you will
be asked to rewrite your commit history to remove any trace of the edit.

The assignment has four parts, and each test is worth one point.

## Part A: GitHub Skills (2 points)

Complete these two tutorials from the [GitHub Skills](https://skills.github.com/)
page, using **public repositories** in your own GitHub account:

- [Hello GitHub Actions](https://github.com/skills/hello-github-actions)
- [Introduction to secret scanning](https://github.com/skills/introduction-to-secret-scanning)

Then open [`src/github_skills.py`](src/github_skills.py), set each
`I_HAVE_COMPLETED_...` flag to `True`, and paste the URL of your completed
public repository. Do the first one before Part C, which asks you to write
a real workflow.

## Part B: SVIX and the probability of a crash (5 points)

Guides:
[E-mini S&P 500 Options Data from Databento](https://finm-32800.github.io/notebooks/_01_es_options_data_ipynb.html)
and
[Replicating SVIX and the Probability of a Crash](https://finm-32800.github.io/notebooks/_02_svix_and_crash_probability_ipynb.html)

Everything is computed from one session's settlement prices, for the two
end-of-month expirations that bracket 30 days. The pull
(`src/pull_es_options.py`), the implied volatilities (`src/smile.py`), and
the smile on a fine grid (`src/rnd.py`) are written for you. Three formulas
are not; each one's docstring states it, and the second guide derives it:

- **SVIX²** (`src/svix.py`, `svix_squared`): Martin's equation (12), the
  integral of out-of-the-money option prices across strikes.
- **The risk-neutral distribution** (`src/rnd.py`, inside
  `risk_neutral_cdf`): the Breeden-Litzenberger step, the CDF from the slope
  of the call prices. The isotonic fix that follows it is given.
- **The log investor's crash probability** (`src/crash_prob.py`,
  `crash_probability`): Martin's formula, one more integral of the same
  distribution.

`uv run pytest src/test_svix.py src/test_rnd.py src/test_crash_prob.py`
checks all three offline, against the closed forms for a flat smile.

**Does the replication match?** Martin publishes his one-month SVIX for
1996 to 2012, in `data_manual/martin_2017/svix2.csv`. `test_svix_matches_martin`
runs your `svix_squared` on the OptionMetrics S&P 500 surface at every
month-end in that period and requires it to stay within a fraction of a vol
point of his series. (CME computes the same construction as its CVOL
indexes, but only for its interest-rate, currency, energy, metals, and
agricultural options, not for the S&P 500.) This test needs the
OptionMetrics pull from Part D, `uv run doit single_names`; the course
runner has its own copy.

## Part C: the daily monitor (8 points)

Guide: [The Daily Monitor](https://finm-32800.github.io/notebooks/_03_daily_monitor_ipynb.html)

- **The daily update** (`dodo.py`, `task_latest`, 1 point): pull the
  latest session, add its row to `_output/svix_monitor_history.csv` if the
  history does not have it, and rebuild the monitor's page,
  `_output/monitor_page/index.html` (`src/build_page.py`, written for you;
  open the page in a browser to see it). Unlike the other tasks,
  it runs every time it is invoked. Plain `uv run doit` is the backfill:
  the same computation over the window in the settings. See
  `test_task_latest` in `src/test_dodo.py`.
- **The ChartBook manifest** (`chartbook.toml`, 1 point): register the
  monitor notebook and the history dataframe, following the entries above
  them.
- **The workflow** (`.github/workflows/deploy_pages.yml`, 5 points): fill
  in the five blanks: the schedule, the secret, the permission to push, the
  step that runs the daily update, and the branch the page is deployed to.
  `src/test_workflow.py` checks them. The other steps are written for you;
  read their comments. One of them tells a data-source outage from a bug:
  if Databento or FRED is still down after the pull's retries, the run ends
  with a warning instead of a failure and yesterday's page stays up.
- **The live page** (`src/published_site.py`, 1 point): the steps below.

### Deploying the monitor

Private repositories in the course organization have no free Actions
minutes and no GitHub Pages, so the monitor runs from a **public repository
in your own account** with the same code. Keep pushing here for grading;
the deploy workflow skips this copy (see the `if:` on its job).

1. Finish Part B and run `uv run doit` locally until it is green.
2. Create a new **public** repository in your own GitHub account, for
   example `svix-crash-monitor`, with no files in it.
3. Add it as a second remote and push:

   ```bash
   git remote add public https://github.com/<your-username>/svix-crash-monitor.git
   git push public main
   ```

   From now on, `git push` sends your work here for grading and
   `git push public main` sends it to the monitor.
4. In the public repository: Settings → Secrets and variables → Actions →
   New repository secret, named `DATABENTO_API_KEY`, with your key. The
   key goes nowhere else: not in the workflow, not in a commit.
5. Actions tab → *Build and Deploy the SVIX Monitor* → Run workflow. A run
   takes a few minutes; the first one creates two branches,
   `monitor-history` for the series (one row so far) and `gh-pages` for the
   page. If it fails, open the run: the log of the failed step says why.
6. Actions tab → *Backfill the SVIX Monitor's History* → Run workflow, with
   a start date (four weeks back is plenty). A session is a minute or two
   to pull, so this takes about half an hour; start it at any time other
   than 14:30 UTC, when the daily run begins. The rows reach the page at
   the next daily run, or run *Build and Deploy* again.
7. Settings → Pages → Deploy from a branch → `gh-pages` / `/ (root)`.
8. When the page is live at
   `https://<your-username>.github.io/svix-crash-monitor/`, open
   `src/published_site.py`, set the flag to `True`, and paste the address
   of the page and of the public repository. Push here.

The workflow then runs on its own every morning at 14:30 UTC. Check back
the next day: the `monitor-history` branch has one more commit, the history
one more row, and the chart one more point. Your `main` branch is never
written to: the only things the workflow writes are that one file on its
own branch and the page on `gh-pages`. The page shows SVIX and the
probabilities, never the settlement prices, which come under the course's
Databento license; the notebooks and the ChartBook site stay on your
laptop.

## Part D: single names, from the OptionMetrics surface (4 points)

Guide: [Crash Bounds for Single Names (Martin and Shi 2025)](https://finm-32800.github.io/notebooks/_04_single_name_crash_bounds_ipynb.html)

The code that computes the Martin-Shi bounds for Apple and AIG is written
for you (`src/single_name_bounds.py`, `src/bounds.py`,
`src/utility_correction.py`), and so are the pulls of prices and rates. One
blank is left:

- **The surface query** (`src/pull_optionmetrics.py`,
  `build_surface_query`): the `WHERE` clause that selects the S&P 500,
  Apple, and AIG at the 30- and 91-day maturities over the sample window,
  from one year's `vsurfd` table. Two tests run it against a small
  in-memory database laid out like WRDS
  (`uv run pytest src/test_pull_optionmetrics.py`).

Then run `uv run doit single_names`. It pulls the surface from WRDS (about a
minute; it needs your WRDS username in `.env` and the `.pgpass` file from
HW 1) and computes the bounds, using your Breeden-Litzenberger step from
Part B. The other two tests check that the bounds were computed across the
sample, and compare the index crash probability with the monthly series
Martin publishes, over 1996 to 2012.

## Grading

Every push to this repository runs the test suite on the course runner and
posts a score to the Actions tab: 19 tests, 1 point each. The runner uses
its own copy of the pulled data and its own run of the single-name part, so
it never needs your credentials or your `.env`.

## Other commands

```bash
uv run pytest src/test_svix.py       # one test file
uv run doit list                     # the pipeline's tasks
uv run doit latest                   # the daily update: pull the latest session, add its row
START_DATE=2026-09-01 uv run doit    # backfill the history from a date
uv run doit single_names             # the OptionMetrics part (needs WRDS)
uv run doit forget pull && uv run doit   # pull the settlements again
open _output/monitor_page/index.html # the monitor's page, as it will be published
uv run chartbook browse              # open the local ChartBook site in your browser
uv run ruff format . && uv run ruff check --fix .   # format and lint
```

## Directory structure

- `src/`: the code, the tests, and the notebooks. The notebooks are jupytext
  `.py` files that `doit` converts and runs. They only present results; the
  modules do the computing.
- `data_manual/`: data entered by hand (Martin's published series). It is
  tracked in git.
- `_data/` and `_output/`: pulled data and generated charts, tables,
  notebooks, the monitor's history, and its page. Both are gitignored, and
  `doit` recreates them; on the live monitor the workflow keeps the history
  between runs on the `monitor-history` branch.
- `docs_src/`: the hand-written pages of the local ChartBook site. `docs/`,
  the built site, is gitignored.
- `.github/workflows/`: `tests.yml` is the autograder; `deploy_pages.yml` is
  the monitor you complete.

## Credits

The engine that recovers the risk-neutral distribution and computes the
Martin-Shi bounds (`rnd.py`, `utility_correction.py`, `bounds.py`) is the
work of Riley Haas and Marija Jovicic, from their Summer 2026 FINM 32800
final project, a replication of Martin and Shi (2025), used with their
permission. The SVIX integral follows Andrew Heekin and Chandler Bird's
Summer 2026 replication of Martin (2017).
