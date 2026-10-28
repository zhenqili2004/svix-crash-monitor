# SVIX and the Probability of a Crash

The top panel is the 30-day SVIX computed every morning from the previous
session's E-mini S&P 500 option settlements, next to the Cboe VIX. Both are
annualized volatilities in percent; SVIX weights every strike equally where
the VIX weights by 1/K², so it sits a little below the VIX, and the gap
widens when the smile steepens.

The bottom panel is the probability that the S&P 500 falls 20% or more by
the expiration closest to one month, under three sets of beliefs: the
risk-neutral probability the options market charges for, the belief of an
investor with log utility who holds the market (Martin 2017), and the
belief of an investor with power utility and risk aversion 2 (Martin and
Shi 2025). The risk-neutral number is the largest, because it prices crash
states at their marginal utility rather than their frequency.

**Updated daily.** A GitHub Actions workflow runs at 14:30 UTC, appends the
session to the history kept on the repository's `monitor-history` branch,
and republishes this site. The notebook *Replicating SVIX and the
Probability of a Crash* derives every number from one session's settlement
prices.
