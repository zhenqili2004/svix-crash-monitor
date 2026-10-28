# Methodology

## Approach

Everything starts from one session's settlement prices for one expiration.

1. **The forward.** Put-call parity at each strike with both a call and a
   put gives the price of the underlying future, `F = K + e^{rτ}(C − P)`;
   the median over the five nearest-the-money pairs is used.
2. **The smile.** Each out-of-the-money contract's settlement is inverted to
   a Black-76 implied volatility, with the discounted forward
   `S* = F e^{−rτ}` playing the role of the spot. Strikes become moneyness
   `k = K / S*`.
3. **The priced grid.** The smile is interpolated linearly in moneyness onto
   a grid of 2,000 points on `[1/3, 3]` (flat outside the observed strikes,
   as in Martin and Shi, Appendix D), and a call is priced at every point.
4. **SVIX²** is the integral of out-of-the-money prices on that grid,
   `2 / (τ R_f) [∫ put + ∫ call]`, Martin (2017) equation (12). The two
   expirations bracketing 30 days are interpolated in total variance, as
   the VIX is, to a constant 30-day horizon.
5. **The risk-neutral distribution** of the gross return comes from the
   slope of the call curve, `Q(k) = 1 + R_f dC/dk` (Breeden and
   Litzenberger), made monotone by an isotonic regression.
6. **Crash probabilities.** `Q(0.8)` is the risk-neutral probability of a
   20% drop. Martin (2017) turns it into the belief of a log investor who
   holds the market, `P = E*[R 1{R ≤ 0.8}] / R_f`, which is
   `0.8 Q(0.8) / R_f − put(0.8)`. Martin and Shi (2025) do the same for
   power utility with risk aversion 2, their equation (7).
7. **Single names.** For Apple and AIG the risk-neutral distributions of the
   stock and of the index give the Fréchet-Hoeffding bounds of Martin and
   Shi's Result 3 on the real-world probability of a 20% drop.

The notebooks derive each step with the numbers for one session.

## Implementation Notes

- The engine (`rnd.py`, `utility_correction.py`, `bounds.py`) is Riley Haas
  and Marija Jovicic's, from their Summer 2026 replication of Martin and Shi,
  used with their permission; `svix.py` follows Andrew Heekin and Chandler
  Bird's replication of Martin (2017). Both are credited in the README.
- Settlement prices at the exchange's minimum tick are floors, not prices,
  and are dropped before the inversion.
- Expirations closer than seven days are not read: a smile a few days from
  expiry is dominated by the pin.
- Probabilities are read from the expiration closest to 30 days rather than
  interpolated, so they always come from one traded chain.

## Caveats and Limitations

- The E-mini option's underlying is the ES future, not the cash index, so
  "a 20% drop" is a 20% drop in the future's price by the option's expiry,
  which differs from the index's return by the carry over a month, a few
  tenths of a percent.
- CME computes the same simple-variance construction as its CVOL indexes,
  but only for options on interest-rate, currency, energy, metals, and
  agricultural futures; there is no CVOL for the S&P 500. The benchmark for
  the SVIX integral is therefore Martin's own published series, which the
  single-name part reproduces from the OptionMetrics surface. CVOL also
  differs in detail: it sums over listed strikes rather than an interpolated
  smile, and it interpolates variance, not total variance, to 30 days.
- Martin's published series ends in January 2012 and the course's Databento
  feed starts in June 2010; the overlap check uses the OptionMetrics
  surface, not the E-mini settlements.
