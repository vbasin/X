# ADR-052: Path-realised account simulator (label-implied KPI is optimistic)

## Status
Accepted — implemented and measured 2026-08-21. Adds a **second evaluator** alongside
the label-implied KPI; it does not replace the leaderboard contract and does not change
the live pin (`nq-50pt-econ`).

## Date
2026-08-21

## Context

Every economics number on the leaderboard is **label-implied**. `label_implied_pnl`
(`src/evaluation/_kpi.py`) prices a trade off the confusion matrix: a correct
directional call earns `+move_threshold`, a wrong one is charged `-mae_tolerance`. It
never reads a price path.

The labels underneath it are themselves **close-only**. `generate_labels_tbm`
(`src/training/labels.py`) takes a single `prices` series — the 5s `_close`
passthrough — and `_classify_triple_barrier` compares closes to both barriers. Nothing
in the chain can observe an excursion that happened *inside* a 5-second bar.

With `move_threshold = 50` that is harmless: a 50-point target is far larger than a 5s
NQ bar. With `mae_tolerance = 6` it is not. A 5s bar during the cash open routinely
ranges more than 6 points, so a bar that traded 9 points against the entry and closed
+1 is scored as an untouched stop. The bias is one-directional and lands squarely on the
metric the objective is short of (`nq-50pt-xgb-unique` requires `hit_rate ≥ 0.55`;
ADR-047 measures 0.328).

`src/backtest/engine.py` is a step better — it walks the 5s close series with real
entries, exits and a cooldown — but it documents the same conservatism gap:
"exits are evaluated on subsequent bar closes (5s bars — no intrabar assumption)".

Separately, the account rules an operator actually trades under (contract sizing,
margin, a concurrent-position cap, a per-session loss halt) were nowhere in the
evaluation chain. Their omission matters in both directions: they can veto trades the
KPI counted, and they can end a session early.

## Decision

Add `src/backtest/portfolio_sim.py` — a path-realised portfolio simulator that resolves
first-touch on the **1-second high/low path** and runs every trade through an account.
It reads no labels; P&L comes only from the path.

**Path source.** New `parquet_path1s` dataset (`scripts/data/build_path1s.py`,
`data_paths.parquet_path1s_dir()`): raw 1s OHLC passthrough from
`databento.ohlcv_1s` (NQ.c.0, 07:00–15:15 CT), one file per session. This is a *path*
dataset, not a feature dataset — no transforms, so there is no look-ahead question.
Seconds with no exchange bar are reindexed onto a contiguous grid with
`high = low = close`: no trade means no excursion.

**Constraints are enforced, not learned.** The position cap, margin capacity, hold
limit, session window, cooldown and loss halt are environment rules evaluated in the
simulator. A model must never spend capacity rediscovering a rule that can be applied
for free.

**Trade frequency is a dial, not a training target.** `sweep_tau` reports the
`(n_trades, expectancy, drawdown, $/month)` frontier across confidence thresholds. A
minimum-trade *quota* in a reward function makes idling impossible but also makes
negative-expectancy trades mandatory, and creates an end-of-period gambling incentive.
Frequency stays where it already is in `kpi.requirement` — a **constraint on model
selection** (`n_signals ≥ 30`), never a term in the objective.

**Bias is measured, not assumed.** `tie_policy` selects what happens when one 1s bar
straddles both barriers: `stop_first` (default, conservative), `target_first`
(optimistic bound), or `close_only` (reproduces the legacy close-series blind spot).
`compare_tie_policies` re-prices identical signals under all three, so the optimism in
a published KPI is a number rather than an argument.

Ordering inside a second is conservative and fixed: mark held positions on the bar's
extremes → book scheduled barrier exits → test the loss halt on worst-case equity
(realised + adverse unrealised) → only then consider a new entry.

## Measurement — ADR-047's own signals, rescored

Input: `data/research/search/nq-50pt-xgb-unique/20260821T110058Z/oof_xgb_causal_topgroups.parquet`
under the ADR-047 inference policy (τ=0.65, 08:30–10:00 CT, 600s cooldown), 255
sessions 2025-07-01 → 2026-06-30, 1 contract, costs $2.50/side + 1 tick entry slippage
+ 2 ticks stop slippage.

The simulator independently reproduces **n=58 trades** — the same count ADR-047's
`open_cooldown` policy reports — which validates the gating before the economics are
read.

| Evaluator | Hit rate | Net expectancy (pts) | Total (pts) |
|---|---|---|---|
| Label-implied (ADR-047, `open_cooldown`) | 0.328 | **11.34** | 658 |
| 1s path, `close_only` (legacy blind spot) | 0.224 | 5.91 | 343 |
| 1s path, `stop_first` (honest floor) | **0.190** | **3.97** | 230 |

Path exits: 47 stops, 11 targets, 0 time-barrier. Average hold 48 s, average MAE 7.2
points — i.e. the median trade was already past the 6-point stop.

**Eight of ADR-047's nineteen "correct calls" never reached +50 before −6 on the path.**
Net expectancy falls 11.34 → 3.97 points, a 65% haircut, against a requirement of 20.

### The τ frontier contradicts ADR-047's operating point

Same signals, same account, τ swept (255 sessions, 1 contract, `stop_first`):

| τ | n | trades/session | hit rate | net exp (pts) | net $ | **median month $** | MDD $ |
|---|---|---|---|---|---|---|---|
| 0.50 | 370 | 1.45 | 0.184 | 3.50 | 25,930 | +640 | −6,045 |
| **0.55** | **225** | **0.88** | **0.222** | **5.64** | **25,360** | **+1,230** | **−3,375** |
| 0.60 | 123 | 0.48 | 0.211 | 5.19 | 12,775 | +658 | −1,620 |
| 0.65 *(ADR-047)* | 58 | 0.23 | 0.190 | 3.97 | 4,600 | **−110** | −2,160 |
| 0.70 | 16 | 0.06 | 0.250 | 7.38 | 2,360 | +228 | −675 |
| 0.75 | 3 | 0.01 | 0.000 | −6.75 | −405 | −203 | −405 |
| ≥0.85 | 0 | 0.00 | — | — | 0 | — | 0 |

τ = 0.55 **dominates** ADR-047's τ = 0.65 on every axis that matters — 4× the trades, a
higher hit rate (0.222 vs 0.190), higher net expectancy (5.64 vs 3.97 points), and a
median month of +$1,230 against −$110. ADR-047 chose 0.65 by *label-implied* net
expectancy, which rewards a small high-confidence subset whose apparent edge does not
survive path pricing. This is the concrete argument for selecting an operating point off
a path-realised frontier rather than by a single label-implied scalar.

No halt day occurred at any τ: with a 6-point stop one contract risks ~$130, so a $5,000
account never approached its 50%-of-day limit. Drawdown binds across sessions instead —
τ=0.65 draws down 43% of a $5,000 account at **1** contract, so 5 contracts would draw
−$10,800 against $5,000 of capital, i.e. ruin.

Two secondary findings:

* `ambiguous_exit_frac = 0.000`, and `stop_first` is identical to `target_first`. At 1s
  resolution a single bar spanning a 56-point round trip does not occur, so intrabar
  tie-breaking is **not** a live concern for the 50/6 geometry. It would be for a 5/5
  geometry, and it is exactly what the 5s close series cannot see.
* At 1 contract the strategy is a lottery, not an income: 10 months, median month
  **−$110**, mean +$460, best +$3,220, only 40% of months profitable. A monthly income
  target cannot be met by a distribution whose median is negative regardless of size.

## Consequences

* Leaderboard economics remain label-implied and comparable to history. Path-realised
  numbers are reported separately; the two are not interchangeable and the gap above is
  the reason.
* `nq-50pt-xgb-unique` is further from its `net_expectancy ≥ 20` / `hit_rate ≥ 0.55`
  requirement than ADR-046/047 indicated. No promotion follows from this ADR.
* The honest-fill problem is a **labelling** problem, not only a scoring one: training
  targets built by `generate_labels_tbm` on 5s closes teach the model that intrabar
  stop-outs are wins. Re-labelling from the 1s path is a prerequisite for any model
  aimed at the 6-point stop, and is tracked in ROADMAP.
* `purge_bars` is a raw row count (`walk_forward_calendar.DEFAULT_PURGE_BARS = 120`,
  correct at 5s = 10 min). Any 1s walk-forward must set `purge_bars = 600` or the purge
  covers 2 minutes of a 10-minute label horizon.

## Alternatives considered

**Extend `src/backtest/engine.py` to 1s.** Rejected: the engine's loop is built around
rule/schedule/prediction modes on the 5s feature matrix, and the account interactions
(sizing × margin × cap × halt) need a per-second event loop. Grafting one onto the other
would have made both harder to reason about. The two evaluators stay separate and are
cross-checked by trade count.

**Resolve fills from `databento.tbbo` ticks.** Deferred, not rejected. Tick sequence is
the only way to order two barriers touched inside the same second — but the measurement
above shows that case does not arise at 50/6, so 1s OHLC is sufficient here. Revisit if
a geometry with a target under ~10 points is ever pursued.

**A minimum-trade quota in the reward.** Rejected — see *Decision*.

## References

* `src/backtest/portfolio_sim.py`, `tests/test_portfolio_sim.py`
* `scripts/data/build_path1s.py`, `scripts/search/run_portfolio_sim.py`
* [ADR-023](ADR-023-triple-barrier-label-scheme.md) — TBM first-touch labels (close-only)
* [ADR-027](ADR-027-economics-meta-sparse-objective.md) — economics KPI
* [ADR-047](ADR-047-inference-uniqueness-open-policy.md) — the policy rescored here
* [ADR-048](ADR-048-binary-side-head-rejected.md), [ADR-049](ADR-049-tbm10-side-transfer-rejected.md) — side-model negative results
