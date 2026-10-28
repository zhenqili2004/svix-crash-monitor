# Data Sources

## Overview

Two market data sources, one public macro source, and two small files kept
by hand. Notebooks never touch the network; they load the cached parquets
the pulls produce.

## Datasets

| Dataset | Source | Frequency | Description |
|---------|--------|-----------|-------------|
| E-mini S&P 500 option settlements | Databento, `GLBX.MDP3`, parent symbol `EW.OPT`, schema `statistics` | Daily | The exchange's settlement price for every listed end-of-month E-mini option, joined to the `definition` records for expiration, strike, and right; the trailing week |
| ES futures settlements | Databento, `GLBX.MDP3`, parent symbol `ES.FUT`, schema `statistics` | Daily | Settlement prices of the outright E-mini futures, to check the parity-implied forward |
| VIX and DGS1MO | FRED ([VIXCLS](https://fred.stlouisfed.org/series/VIXCLS), [DGS1MO](https://fred.stlouisfed.org/series/DGS1MO); no API key) | Business days | The Cboe VIX close, and the one-month Treasury yield used as the risk-free rate |
| OptionMetrics volatility surface | WRDS, `optionm_all.vsurfd{YYYY}` | Daily | The standardized surface for the S&P 500, Apple, and AIG at 30 and 91 days, 1996 to the latest release |
| OptionMetrics prices and zero curve | WRDS, `optionm_all.secprd{YYYY}`, `optionm_all.zerocd` | Daily | Closing prices for the spot, and the zero-coupon curve for the rates |
| Martin's published series | https://personal.lse.ac.uk/martiniw/ | Daily, monthly | SVIX² and the crash probability, 1996 to 2012, in `data_manual/martin_2017/` |

## Data Pipeline

- `src/pull_es_options.py` pulls the settlements and definitions one session
  at a time. Everything it asks for is free under the course's Databento
  subscription; it prices each query with the free `metadata.get_cost`
  endpoint first and refuses to download anything whose estimate is not
  $0.00.
- `src/pull_fred.py` pulls the two FRED series from the public CSV endpoint.
- `src/pull_optionmetrics.py` pulls the surface, prices, and zero curve from
  WRDS, one year at a time. It is part of `doit single_names`, not of the
  default build, because the public monitor runs without WRDS.
- `src/monitor.py` turns each session into one row of
  `_output/svix_monitor_history.csv`. The GitHub Actions workflow restores
  that file from the `monitor-history` branch every morning, appends the
  new sessions, and records it there again, so `main` holds only source.
