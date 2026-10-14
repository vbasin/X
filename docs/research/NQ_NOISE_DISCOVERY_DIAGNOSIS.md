# Noise-area discovery diagnosis: exit semantics and turnover

Present: diagnosis of **2011–2020 discovery only** in run
`nq-noise-area-continuation/20260924T040428Z`.
Supersedes: no implementation or result.
Next: explicitly preregister an exit-semantics experiment before any additional
validation is inspected.

No validation-directory artifact or reserved 2026 tape was read for this analysis.
Discovery ledgers and `discovery_trials.json` were read; the shared feature matrix
was loaded with a parquet filter `session < '2021-01-01'`. The report does not
infer replacement profitability from these diagnostics.

## Observed result and mechanism

All 24 configurations failed their qualification conditions, although some had
positive discovery net P&L. The selected exploratory policy 11 used multiplier
1, 30-minute long-only entries and the tighter band/VWAP stop. It had 1,637
completed trades, one unresolved trade, net **-$6,085.32**, win rate **21.56%**,
and profit factor **0.9563**.

| Completed-trade diagnostic | Value |
|---|---:|
| Median holding time | 13 minutes |
| Exits within 1 / 5 / 30 minutes | 17.0% / 35.2% / 63.7% |
| Median initial stop distance | 4.75 NQ points |
| Median distance / causal trailing 20-minute one-minute ATR | 1.94 |
| Median distance / cash range observed at decision | 14.65% |
| Median distance / prior-session same-clock expected move | 30.57% |
| Initial stop selected from VWAP instead of upper band | 4.70% |
| Stop exits | 1,430 trades, -$121,614.80 |
| Session-close exits | 207 trades, +$115,529.48 |
| Trades lasting no more than five minutes | 577 trades, -$42,288.72 |

The initial stop is almost always the same upper boundary that admitted the
long entry. Price need only retest that breakout boundary to exit; it does not
need to invalidate the broader day's directional move. Ratcheting every minute
can then tighten it further. These measurements support a **tight-exit/turnover
hypothesis**, not a causal claim that changing the stop will improve results.
The 4.75-point median spans a decade with very different NQ price levels;
normalized ratios are more informative than that raw number alone.

Fixed one-point price friction plus $3.36 commission costs $38,240.32 over the
completed ledger. Adding it back mechanically gives +$32,155 of same-trade price
movement before nominal friction. This is **not a zero-cost strategy backtest**:
costs also affect loss-count limits and admissible future trades. On the fixed
ledger, break-even price friction is only 0.814 points round trip after commission,
below the declared one-point assumption. Low-quality short holds consume the
larger gains on the minority of sustained trends.

The unresolved trade starts June 30, 2020 at 14:00 UTC and has
`missing_session_end`. It remains invalid and must not be dropped to obtain a
passing result. Missing source coverage requires investigation separately from
exit design.

## Primary-source comparison: this adaptation changed exit timing

The paper restricts trading to half-hourly checks and describes the current-band
plus VWAP exit. Its authors' published Python implementation resolves the timing:
it computes directional close/band/VWAP conditions, samples them only at the
30-minute decision cadence, carries exposure between checks, and shifts exposure
before scoring subsequent price changes. It does **not** execute a minute-by-minute
intrabar stop or retain a monotonic running maximum of past boundaries.

Our adaptation added both of those tighter behaviors. It also omits the
prior-close band anchor, uses NQ rather than SPY, fixed one-contract sizing,
three-entry/two-loss limits, and different execution costs. It is not a replication
of the paper. Sources: [paper, section 3](https://alexandria.unisg.ch/server/api/core/bitstreams/a99aba00-f967-49b3-aceb-f544dc386e0b/content),
[authors' Python implementation, Step 3](https://concretumgroup.com/backtesting-7-years-of-free-data-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/).

## Recommended bounded follow-up

First isolate the mechanism rather than add indicators: keep the same entries
and compare the current exit to a cadence-close confirmation exit using the
current band/VWAP condition. A separate fixed catastrophic stop, defined from
causally available volatility or a structural boundary at entry, can limit
intra-interval risk. Its placement must be declared before results are read;
minute monitoring of that distant safety stop is different from minute monitoring
of the near entry boundary.

VWAP-only and opposite-band exits are distinct alternatives that allow deeper
pullbacks. They also expose the trade to larger sudden reversals; improved win
rate would not itself establish superior economics. If tested, disclose their
additional trials and evaluate net return, tail loss, drawdown, turnover and
stability under the same costs. Do not select a favorable time window or lower
cost assumption from this diagnosis.

## Physical artifacts

- [Completed-trade summary](../../data/research/diagnostics/nq-noise-discovery-20260924/valid_summary.json)
- [Policy 11 joined diagnostics](../../data/research/diagnostics/nq-noise-discovery-20260924/selected_policy11_diagnostics.parquet)
- [All 24 discovery policy diagnostics](../../data/research/diagnostics/nq-noise-discovery-20260924/all_discovery_policy_diagnostics.csv)
- [Distribution table](../../data/research/diagnostics/nq-noise-discovery-20260924/summary.csv)

The distribution table and joined parquet retain the unresolved row. The JSON
and percentages reported above restrict economic summaries to completed trades
while disclosing the unresolved count explicitly.
