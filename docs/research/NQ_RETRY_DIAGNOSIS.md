# NQ retry: measured diagnosis of the failed selective model

Present: independent diagnosis of run `nq-selective-intraday/20260924T033352Z`.
Supersedes: no implementation; supplements `NQ_SELECTIVE_RESULTS.md`.
Next: test a small, preregistered event-level economic hypothesis with an
execution policy matching its labels, rather than expand the same model search.

The earlier work established a causally constructed negative result but stopped
before investigating its causes. Its main design error was treating generic
five-minute states as interchangeable trade opportunities and expecting a broad
return regressor to discover an economic mechanism. Its main process error was
reporting the rejection as the end of the research task.

This diagnosis reads only the cached April 2025–June 2026 experiment and its raw
source files. No July–September 2026 reserve data was inspected. All numerical
findings below are descriptive of already-exposed development data; they cannot
become fresh confirmation evidence for a replacement design.

## 1. Forecasts did not transfer to the actual trading policy

Reloading the two saved models and replaying their original policy gives:

| Selected model / evaluation | Trades | Mean forecast R | Mean realized R | Net USD |
|---|---:|---:|---:|---:|
| Jan–Feb model, training | 256 | 0.199 | 0.633 | 80,144.84 |
| Jan–Feb model, inner | 62 | 0.210 | 0.220 | 11,571.68 |
| Jan–Feb model, outer | 65 | 0.193 | -0.082 | -6,188.40 |
| Mar–Apr model, training | 533 | 0.094 | 0.333 | 106,569.12 |
| Mar–Apr model, inner | 115 | 0.101 | 0.132 | 11,473.60 |
| Mar–Apr model, outer | 117 | 0.094 | -0.105 | -13,443.12 |

Training replay is in-sample diagnosis, not performance evidence. Inner results
were used to select the models and thresholds and are also not independent proof.
Forecast/outcome correlation across all candidate rows fell from 0.253 in training
to 0.031 inner / 0.058 outer for the Jan model, and from 0.231 to 0.072 / -0.023
for the March model. The March model's positive forecasts did not translate into
positive returns even before the sequential portfolio constraint.

**An additional, distinct failure appeared in January–February:** the 115
above-threshold candidates, taking the highest-scored direction at each timestamp,
averaged +0.161R and summed to +$7,258.60. The 65 actually admissible trades lost
$6,188.40. The hypothetical sum is **not executable performance**: labels overlap,
positions occupy later opportunities, and loss/count limits change the sample.
It demonstrates a mismatch between the regression target and the executable
policy, not a discovered profitable replacement.

The target minimized squared prediction error on each direction/time row in R.
Selection maximized dollar P&L with one position, finite daily entries and a
loss stop. Different stop widths also imply different dollar exposure. None of
these transformations preserves an estimated edge automatically.

## 2. The apparent sample size overstated independent information

The matrix has 18,684 decision times across 300 sessions with eligible rows;
the input inventory has 303 sessions. Each time contributes two directions and
two exit templates. Same-session adjacent labels overlap for 79.5–79.9% of
15-minute candidates and 81.6–82.7% of 30-minute candidates. Lag-one same-session
outcome correlations are 0.496–0.502 and 0.589–0.600, respectively. Mean realized
label durations are about 10.6 and 17.0 minutes against a five-minute grid.

There is no evidence that these rows are independent observations. A numerical
effective sample size is deliberately not asserted: dependence across horizons,
directions, days and regimes cannot be captured reliably by one AR(1) formula.
The splits do keep label endpoints before subsequent folds; overlap here is an
information/weighting issue, not proof of a train/test causality violation.

Each outer fold selected from 18 model/template/threshold combinations on roughly
two months. The score was `daily_mean - 0.5 * iid_daily_SE`; that is a weak
selection penalty, not a multiplicity-adjusted confidence bound. Only 62 and 115
actual inner trades backed the selected models. More threshold mining would
increase this weakness.

## 3. A hard stop ceiling changed the eligible population across regimes

Recomputing the original stop on every source tape's scheduled 69-point grid:

| Month | Scheduled times | Stop >60 points | Actually eligible | Median daily median stop |
|---|---:|---:|---:|---:|
| Apr 2025 | 1,173 | 394 (33.6%) | 745 | 40.5 |
| Jul 2025 | 1,449 | 3 (0.2%) | 1,433 | 16.75 |
| Nov 2025 | 1,173 | 199 (17.0%) | 964 | 40.0 |
| Feb 2026 | 1,311 | 196 (15.0%) | 1,105 | 32.5 |
| Jun 2026 | 1,449 | 533 (36.8%) | 848 | 46.75 |

Across all months there were 20,907 scheduled times, 1,871 excluded by the stop
ceiling, and 18,684 eligible times. Another 352 failed other eligibility checks;
this diagnosis does not attribute those to one cause. No ATR values were missing
on the scheduled grid. April 7–8, 2025 and June 9, 2026 had no eligible rows.

Thus `2 * trailing one-minute ATR <= 60` is a substantial regime-selection rule,
not just order sizing. It should be an explicit part of any hypothesis. Causal
normalization alone does not make train/test populations comparable. Removing
the ceiling is not justified from this table: that would also increase risk.

## 4. Loss concentration provides diagnostics, not new fitted filters

The March–April policy was 76.1% short. Across both active folds, 117 shorts lost
$7,403.12 and 65 longs lost $12,228.40; switching direction alone is not supported.
The original long-only baseline made $6,635.08 with substantial drawdown, so the
learned directional selection added little useful evidence relative to that
simple comparator.

| Exit reason | Trades | Net USD |
|---|---:|---:|
| Stop | 86 | -63,643.96 |
| Target | 30 | 31,644.20 |
| Time | 66 | 12,368.24 |

At 08:45–08:55 CT, 14 trades lost $6,947.04 with average stop width 50.91 points.
The 10:00 hour lost $9,946.20 across 45 trades. The 09:00 and 13:00 hours were
positive, but these are small, post-hoc slices and do not authorize optimizing
time filters on the same reported test. Both first trades and later reentries
lost money, so repeated entries are not the sole explanation.

## 5. Execution evidence was overstated

Databento defines TBBO as trade events with the BBO immediately **before the
effect of the trade**. MBP-1 includes top-of-book updates between trades.
Sources: [TBBO schema](https://databento.com/docs/schemas-and-data-formats/tbbo),
[MBP-1 schema](https://databento.com/docs/schemas-and-data-formats/mbp-1).

The simulator takes the first print-associated quote after decision plus latency
as the available market fill. That is a reproducible sampled-quote assumption,
not proof that liquidity remained executable after the recorded trade consumed
it. It also observes stop crossings only at trade snapshots and cannot recover
intervening quote excursions. An independent replay against the same TBBO can
verify implementation consistency, but cannot independently verify those missing
market states. The adverse slippage stress helps, but does not reconstruct them.

No new forward-looking feature join or book-state timing bug was identified in
the reviewed modules. Stop/target exits use subsequent adverse-side quotes after
latency, rather than clamping at ideal barrier prices. The previously documented
legacy normalization leakage remains a separate issue bypassed by this model.

## 6. Concrete replacement design

The next model should estimate the economics of a named, infrequent event with
a mechanically defined invalidation point. A plausible experiment is the first
opening-range escape after the initial 15-minute cash auction, optionally with
one explicitly defined failed-breakout reversal. Freeze the event rule before
scoring it. One initial opportunity per side/session avoids manufacturing dozens
of overlapping copies of the same move. A continuation and failed-auction design
are separate hypotheses; their results must not silently merge into a retrospectively
optimized strategy.

Use only information available when the event is confirmed: distance from the
completed opening range, causal range/volatility, developing VWAP, prior closed
session levels and observed rejection/acceptance. A structural stop should
represent failure of that event. Size economic exposure separately; do not let
a normalized regression target conceal large varying one-contract dollar risk.

Begin with the fixed event policy and transparent per-session outcomes. Add a
small regularized probability/return model only if it improves the *same*
executable event policy in chronological folds over its fixed baseline. Weight
and bootstrap at the session level, calibrate entirely within past windows,
and report long/short and volatility-regime stability. Explicit cash abstention
remains necessary when the model has no supported advantage.

These are hypotheses motivated by the failure mechanism, not promises that
opening-range trading is profitable. Their testable advantage is cleaner
identification: which event, which invalidation, which achievable payoff, and
whether ML adds value. Untouched confirmation should be consumed once only after
a frozen replacement survives development; any redesign after seeing it makes
that period development data.

## Reproduction and physical artifacts

Run from the repository root:

```bash
venv/bin/python -m scripts.research.diagnose_nq_selective_retry
```

The script refuses to overwrite its output directory and asserts all inspected
sessions precede July 1, 2026. Its initial run completed in 3.5 seconds. Artifacts:

- [Policy train/inner/outer calibration](../../data/research/diagnostics/nq-retry-20260924/policy_train_inner_outer.csv)
- [Prediction population calibration](../../data/research/diagnostics/nq-retry-20260924/prediction_calibration.csv)
- [Prediction buckets](../../data/research/diagnostics/nq-retry-20260924/prediction_buckets.csv)
- [Label dependence](../../data/research/diagnostics/nq-retry-20260924/label_dependence.csv)
- [Daily coverage](../../data/research/diagnostics/nq-retry-20260924/coverage_by_session.csv)
- [Monthly coverage](../../data/research/diagnostics/nq-retry-20260924/coverage_by_month.csv)
- [Candidate economics by month, side and horizon](../../data/research/diagnostics/nq-retry-20260924/population_by_month_horizon_side.csv)
- [Ledger by time](../../data/research/diagnostics/nq-retry-20260924/ledger_by_hour_ct.csv)
- [Ledger by exit](../../data/research/diagnostics/nq-retry-20260924/ledger_by_exit_reason.csv)

No original experiment artifact or implementation was modified.
