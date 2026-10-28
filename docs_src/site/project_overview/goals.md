# Goals

## Objectives

- Replicate Ian Martin's SVIX index (Martin 2017) and his option-implied
  probability of a 20% decline in the S&P 500 from exchange settlement
  prices, and check them against the numbers CME and Cboe publish.
- Run the replication as a daily monitor: a GitHub Actions workflow that
  pulls the previous session every morning, appends it to a tracked
  history, and republishes the monitor's page.
- Replicate the single-name crash bounds of Martin and Shi (2025) for Apple
  and AIG from the OptionMetrics surface, and check the index probability
  against Martin's published series.

## Success Criteria

- `uv run doit` builds the index part from scratch: the data pulls, the
  unit-tested engine, the monitor's history and charts, four executed
  notebooks, and the site.
- SVIX computed from the OptionMetrics surface tracks the one-month SVIX that
  Martin publishes for 1996 to 2012, within a fraction of a vol point.
- The log-utility crash probability from the OptionMetrics surface tracks
  Martin's published monthly series on the 1996 to 2012 overlap.
- The workflow's morning run adds one row to the history and one point to
  the chart, with no one touching it.
