# ADR-127: TBBO print tape as the substrate, and a three-stage cascade replacing P&L-ranked first passage

## Status
Accepted — 2026-09-11. Supersedes the single-model policy of
[ADR-123](ADR-123-first-passage-candidate-exit-surface.md) and rebuilds on the
level geometry of [ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md),
which is retained unchanged. Every first-passage label and every metric derived
from one is superseded: the row grain, the label definition and the data source
all changed, so nothing is comparable across this boundary.

## Date
2026-09-11

## Context

### 1. The model optimised the wrong quantity

The incumbent is an `XGBRegressor` fitted on `pnl_r` and ranked by predicted
expectancy ([eval_first_passage_model.py:181-186](../../scripts/features/eval_first_passage_model.py)).
Predicted P&L is the product of two things the model never separates: how likely
price is to move, and how much that move is worth given the bracket that happened
to be enumerated. A speculative 60-point target beats a high-conviction 15-point
one whenever the extra distance outweighs the lower probability, so the ranking
systematically prefers distance. "Is a move coming" and "what is the move worth"
are different questions and were being answered as one number.

The row grain made it worse. A row is `(ts, side, target, stop)`, so one
underlying move generates ~150 rows per decision bar that differ only in bracket
choice. The 2026-06-30 matrix holds 9,145 rows across 25 decision bars. The model
sees the same move over and over with different arithmetic attached.

### 2. Decision cadence was 15 minutes, and that was an accident

`FirstPassageConfig.decision_grid_seconds` is 60, but
[build_training_matrix.py:131-132](../../scripts/features/build_training_matrix.py)
subsamples with `--bar-stride 15` before writing, so the model trained on
15-minute decisions. That was a training-set-size decision that silently became a
modelling one. A setup valid at 08:30 whose reaction resolves by 08:35 was never
rescored.

Going finer under the incumbent schema is unaffordable: 4,500 decisions per
session × ~150 brackets ≈ 680,000 rows per session, ~190M total.

### 3. Every label was built on reconstructed data

`parquet_path1s` carries 1-second OHLCV bars from `databento.ohlcv_1s` under the
continuous symbol `NQ.c.0`. Three approximations followed:

- **Volume-at-price was smeared.** `session_profile` spreads each bar's volume
  uniformly across every bin its `[low, high]` covers. The POC and value-area
  edges — the inputs to every bound, target and stop — were built on that smear.
- **Aggressor side was absent.** Delta arrived through `allocate_tf_delta`, which
  redistributes a *15-second* hive aggregate onto 1-second rows by volume share.
  The sign was real; the intrabar price attribution was not.
- **First touch resolved only to the second.** When target and stop both traded
  inside one second the outcome came from `tie_policy = "stop_first"` — a policy,
  not an observation.

`databento.tbbo` holds 193M print-level rows over 2025-04-06 → 2026-09-11 with
exact price, size, aggressor side and the BBO at the print. It was being used only
through derived 15-second bar aggregates.

A fourth problem was a genuine contract mismatch:
[cache_large_prints.py:60](../../scripts/features/cache_large_prints.py) filters
`symbol LIKE 'NQ%'`, which admits back months *and* calendar spreads
(`NQM6-NQU6`), while the bars it classifies against come from `NQ.c.0`. The
large-print taxonomy was comparing prints from one instrument to candles from
another.

## Decision

### A. The print tape becomes the substrate

`scripts/data/build_tape.py` writes one file per session to
`data/sources/features/parquet_tape/`:

```
tape_YYYY-MM-DD.parquet
    ts (UTC index) | price float64 | size int32 | side int8
    bid_px float64 | ask_px float64 | bid_sz int32 | ask_sz int32
```

`side` is the aggressor (`+1` lifted the offer, `-1` hit the bid). Rows are
ordered by `(ts_event, sequence)`, which is what makes first touch resolvable.
The lead contract is resolved per session through `active_contract`, so the
symbol mismatch is gone by construction; an exact-symbol match excludes calendar
spreads. **319 sessions, 94.1M prints, 3.9 MB per session.**

`src/features/tape.py` provides the replacements: `tape_profile` (exact
volume- and delta-at-price, no spreading), `tape_levels` (POC/VA from the exact
profile), `tape_delta_1s` (retires `allocate_tf_delta`), `print_ofi` (Cont 2014
OFI from the stored quote), `tape_flow_bars` (hive-shaped frame computed from
this session's prints), and `sparse_tables`/`rmq`/`first_cross` for O(1) range
queries with vectorised binary search.

The BBO columns are not optional. Without them `of_ofi` collapses onto
`of_delta`, and `es_ofi_cum` was the incumbent model's highest-gain feature.

### B. Three stages, each answering one question

**Stage 1 — move detector.** One row per `(ts, side)`. No target, no stop, no
bracket, no P&L. Label:

$$y_\text{move} = \mathbb{1}\left[\text{MFE}_\text{side} \geq \texttt{move\_r}\cdot R \;\wedge\; \text{MAE}^{<t_\text{move}}_\text{side} < \texttt{adverse\_r}\cdot R\right]$$

Adverse excursion is measured **only up to the moment the favourable threshold is
cleared**. Measuring it over the whole horizon — the first implementation —
disqualified a run of +180 points because it retraced 16 points an hour later, and
cut the positive rate from 30% to 3%.

Features are the `es_*` and `fs_*` families. Every `tgt_*` and `stp_*` column is
excluded, which is the whole point.

**Stage 2 — stop rule, not stop search.** The stop is resolved by rule so the
label stops being a function of a searched parameter.

**Stage 3 — target selector.** Runs only on gated bars. Row grain
`(ts, side, target_bound)`; the stop comes from Stage 2, so there is no cartesian
product. Label `y_reach`: did price touch this bound before the stop. Ranking is
by probability of being reached, not by expectancy, so a tight high-conviction
target can beat a distant speculative one. The row also carries `t_reach`
(seconds until the target was touched, `-1` if never within the label horizon)
and, since 2026-09-15, `t_stop` (seconds until the fixed 0.45 R stop was pierced,
`-1` if never within the horizon). `t_stop` is not a training feature — it is in
the eval `LEAK` set — but it lets any replay engine (the WebUI backtest adapter,
future paper/live tooling) distinguish stop-outs from horizon censoring and free
the position-book slot at the honest time instead of at the horizon.

This is what makes 5-second cadence affordable:

| | rows/session |
|---|---|
| Incumbent (60s labels, 15-min matrix) | ~56,000 |
| Naive 5s, incumbent schema | ~680,000 |
| **Cascade at 5s** | **~9,000 (Stage 1) + ~500 (Stage 3)** |

### C. Uniqueness weighting is mandatory, not optional

At 5-second cadence a 30-minute move is claimed by 360 consecutive decisions.
`src/evaluation/uniqueness.py` implements concurrency and average-$1/c_t$
uniqueness (López de Prado, *AFML* ch. 4) via a difference array, so cost is
linear rather than quadratic in overlap. Weights are used twice: as training
sample weights, and as the effective sample size $n_\text{eff} = \sum_i u_i$ in
every reported t-statistic. `n_eff` is printed beside `n` throughout.

Without this, moving to 5 seconds would have improved every t-statistic while
changing no underlying edge.

## Measured result: the structural stop does not beat distance

`scripts/features/probe_stop_rule.py` places each candidate stop and asks whether
it was touched before the move completed, against a **distance-matched random
control** — a stop at the identical distance on no structure at all.

Measured 2026-06-01 → 06-10, 5s cadence, ~153k probe rows:

| policy | median dist_r | survival | survival ctrl | **edge vs ctrl** | coverage |
|---|---|---|---|---|---|
| tightest | 0.43 | 0.669 | 0.656 | **+0.013** | 1.00 |
| wick | 0.49 | 0.710 | 0.694 | **+0.015** | 0.77 |
| zone | 0.51 | 0.698 | 0.685 | **+0.012** | 0.72 |
| absorption | 0.50 | 0.699 | 0.681 | **+0.018** | 0.06 |
| safest | 0.64 | 0.773 | 0.759 | **+0.014** | 0.51 |

Every structural policy beats its own control by 1.0–1.8 percentage points, and
the ranking is explained almost entirely by distance: 0.41 R → 67%, 0.64 R → 77%.
Sweeping `large_order_size` over {10, 20, 50} and `lookback_bars` over {5, 10}
does not change the conclusion; at `large_order_size = 50` the absorption edge is
**negative** (−0.015) on 484 samples.

**The structure is not doing the work. Distance is.** `StopConfig.policy`
therefore defaults to `fixed` at `fixed_dist_r = 0.45`, which has 100% coverage
and matches the structural policies' survival at the same distance. The
structural policies are kept so the finding stays reproducible.

This is a negative result and it was worth getting before building Stage 2 on the
absorption shelf. The hypothesis — a large sell print in the lower wick of an up
candle marks a defended bid — is mechanically sound and is correctly detected by
`build_stop_bars`; it is simply not predictive of survival once distance is
controlled for. Its 6% coverage also makes it unusable as a primary rule.

## Measured result: Stage 1 separates moves from non-moves

Full build, 2025-05-01 → 2026-06-30, 5-second cadence, 301 sessions,
**2,663,138 rows**, 65 features, base rate 0.2935. Walk-forward, 5 folds, purged.

**Weighted AUC 0.6627.** The gate frontier, with τ taken from train quantiles:

| τ (train q) | n | **n_eff** | per session | precision | lift | mean mfe_r | t vs base | median t_move |
|---|---|---|---|---|---|---|---|---|
| all rows | 1,895,452 | 16,348 | 8,857 | 0.3086 | 1.00× | 2.01 | — | 71 s |
| 0.50 | 946,620 | 8,283 | 4,423 | 0.4061 | 1.32× | 2.08 | 18.1 | 59 s |
| 0.75 | 524,611 | 5,465 | 2,451 | 0.4489 | 1.46× | 2.31 | 20.9 | 39 s |
| 0.90 | 235,732 | 3,230 | 1,102 | 0.4936 | 1.60× | 2.65 | 21.0 | 23 s |
| **0.95** | 122,331 | **2,063** | 572 | **0.5297** | **1.72×** | 2.90 | 20.1 | 16 s |
| 0.99 | 25,559 | 719 | 119 | 0.6001 | 1.95× | 3.27 | 16.0 | 8 s |

Precision, lift and forward excursion all rise monotonically with τ, and the
result is stable across folds (precision 0.517–0.537 at τ=0.95, t 8.4–9.7) and
across sides (short 0.5267, long 0.5328 — the mirroring in
`build_move_matrix` works).

**The uniqueness ratio is 0.0086.** 2.66M raw rows carry the information of
~22,600 independent observations: at a 5-second cadence with a median 71-second
move, roughly 116 consecutive decisions claim the same underlying event. Every
statistic above is computed on `n_eff`, so the τ=0.95 t-statistic of 20.1 rests
on 2,063 effective samples, not 122,331. **Had the same frontier been reported on
raw `n` it would have shown t ≈ 155** — the exact inflation this ADR exists to
prevent.

**The honest caveat is `median_t_move`.** At τ=0.95 the gated move completes in
**16 seconds**, and at τ=0.99 in 8. The model is largely detecting bursts already
in progress rather than forecasting them, which puts entry latency directly in
the critical path: a fill arriving 20 seconds after the decision has missed the
move it was selected for. This is a tradability question, not a statistical one,
and it is the first thing Stage 3 and any live path must answer. Raising
`move_r` above 0.25 and `horizon_seconds` are the levers.

## Measured result: Stage 3 ranks reachability, and the cascade composes

Built on the τ_move = 0.95 gate — **6.45%** of Stage-1 rows survive it — 214
sessions, **559,582 rows** (≈2,600 per session against the incumbent's ≈56,000, at
12× finer cadence). 11 features, all geometry; `p_move` is excluded so the two
gates stay independent. Reach rate 0.2136.

**Weighted AUC 0.8569** — far above Stage 1's 0.663, which is expected: "will
price get to this level before that one" is a more tractable question than "is a
move coming".

The τ_reach frontier has an **interior optimum**, which is the whole point:

| τ_reach | n | n_eff | reach rate | mean R | median tgt_r | t |
|---|---|---|---|---|---|---|
| all rows | 329,579 | 2,065 | 0.2336 | **−0.078** | 2.70 | −4.03 |
| 0.50 | 166,371 | 1,078 | 0.3989 | +0.064 | 1.35 | 2.70 |
| **0.75** | 85,587 | 587 | 0.5707 | **+0.111** | 0.69 | **4.67** |
| 0.90 | 34,030 | 245 | 0.7830 | +0.096 | 0.28 | 4.62 |
| 0.95 | 17,190 | 129 | 0.8933 | +0.073 | 0.14 | 4.20 |

Tightening the gate raises the hit rate monotonically (0.23 → 0.89) while pulling
targets nearer (2.70 R → 0.14 R), and expectancy peaks in between. **A model
ranking on predicted `pnl_r` cannot represent this trade-off**, because distance
and probability are summed before it sees them. Ranking on reachability alone
exposes it, which is what the redesign was for.

Taking a random candidate loses money (−0.078 R, t = −4.03). Taking the nearest
bound unconditionally earns +0.060 R (t = 2.22). The model at τ_reach = 0.75 earns
+0.111 R (t = 4.67) — **it beats the nearest-bound heuristic, not just chance.**

**The cascade composes.** At fixed τ_reach = 0.75, raising τ_move lifts expectancy
monotonically — +0.111 → +0.134 → +0.166 → +0.208 R at τ_move quantiles
0.0/0.25/0.50/0.75. Stage 1 adds information Stage 3 does not have, which is the
claim the two-head split rests on.

Best joint cell (τ_move ≥ q0.50, τ_reach ≥ q0.75), after argmax-per-bar and the
one-position book: 3,953 trades over 122 sessions, 31.1/session, reach rate 0.679,
+0.103 R/trade, **n_eff 3,875 of 3,953** — the book's own output is 98% unique, as
it must be, since a single-position book cannot hold overlapping trades.

### Two book bugs found and fixed

Both would have inflated the result, and both are the same species as the int32
overflow that voided every pre-2026-09-11 book figure in
[ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md).

1. **Non-stable sort.** `sort_values` defaults to quicksort, so tied timestamps
   ordered arbitrarily and the greedy walk was non-deterministic — two runs of
   identical logic returned 3,193 and 2,129 trades. Fixed with `kind="mergesort"`.
2. **Zero-duration blocking.** A target reached inside the decision second has
   `t_reach == 0` and freed the book instantly, so several positions could open at
   the same instant: 3,193 rows spanned only 2,995 distinct timestamps. Fixed with
   a one-second floor plus an explicit argmax-per-bar step, and asserted in the
   output (`distinct ts` must equal row count).

A third error was methodological rather than a bug: applying the **pre-book**
uniqueness weights to **post-book** trades discounts overlap twice, since the book
has already removed it. Weights are now recomputed against the trades actually
held, which moves the book's t from an over-penalised 3.64 to 13.69.

### Cost sensitivity

Median target 13.50 pts against a median 16.09 pt stop — a **0.84:1 payoff won
68.8% of the time**. That is a scalping profile, and scalping profiles die on
costs, so they are stated explicitly:

| assumption | net pts | net $ (1 MNQ) | pts/trade | t |
|---|---|---|---|---|
| frictionless | +14,080 | +$28,159 | +3.56 | 13.72 |
| 1 tick/side + $1 RT | +10,127 | +$20,253 | +2.56 | 9.87 |
| 2 ticks/side + $1 RT | +8,150 | +$16,300 | +2.06 | 7.94 |

It survives two ticks of slippage per side. **This is not a promotion claim.**
Three folds only, thresholds chosen by reading the frontier, no DSR/PBO, and
31 trades/session at a 25-second median hold puts the whole result at the mercy of
fill quality that this backtest does not model.

## Consequences

- **All first-passage labels and metrics are superseded.** Row grain, label
  definition and data source changed together. The τ frontier in ADR-126 and the
  ROADMAP describes a different problem.
- `parquet_path1s` is no longer the label substrate. It remains for the bar
  backtester. It also cannot be extended past 2026-07-14, where `ohlcv_1s` for
  `NQ.c.0` ends, while the tape runs to 2026-09-11.
- `allocate_tf_delta` is retired from the cascade path. It is still correct for
  anything reading `parquet_path1s`.
- **`tie_policy` is dead on the tape path.** First touch is observed.
- `sigma_pts`, `buffer_pts` and the level lookback from ADR-126 are unchanged and
  still govern Stage 3's bounds.
- The knob count grew: `move_r`, `adverse_r`, `horizon_seconds`,
  `decision_grid_seconds`, `fixed_dist_r`, `max_targets`, plus two thresholds
  $\tau_\text{move}$ and $\tau_\text{reach}$. The Deflated Sharpe / PBO debt from
  [ADR-122](ADR-122-structural-context-and-closed-research-loop.md) is now larger
  and must be paid before promotion.
- `fs_ofi_*`, `fs_absorption_*` and `fs_impact_efficiency_*` now receive true
  Cont-2014 OFI. On the incumbent path
  [build_training_matrix.py:166](../../scripts/features/build_training_matrix.py)
  sets `ofi = delta`, so those columns never saw their documented input.

## Alternatives rejected

- **Keep the P&L regression and add a probability head as a feature.** The
  regression target still rewards distance; adding a probability column does not
  remove the incentive, it only gives the tree another way to express it.
- **Drop to 5 seconds under the incumbent schema.** ~190M rows and an ~8-hour
  label build to answer a question the cascade answers with 6× *fewer* rows than
  today.
- **Learn the stop as a third model head.** Rejected on the probe evidence: with
  structure worth ~1.4 points of survival against a distance-matched control,
  a third head would fit noise and add two more thresholds.
- **Use `mbp_10` depth for queue-position-aware fills.** The table covers
  2026-07-21 → 2026-08-20 and the label range is 2025-05-01 → 2026-06-30 — zero
  overlap. Revisit when depth history is long enough.
- **Keep `large_prints.parquet`.** Filtered to `size >= 20` at cache time and
  contaminated by back months and calendar spreads. The tape carries the full
  lead-contract print stream, so the threshold becomes a runtime parameter.

## Follow-ups

1. `cache_large_prints.py` has no lead-contract filter while
   `live_chart.load_large_prints` does. Either fix it or retire it in favour of
   the tape.
2. `features_5s_part1.parquet` is corrupt (`TProtocolException`) and will break
   any full-hive scan of the 5s store.
3. `ohlcv_1s` for `NQ.c.0` shows ~⅓ normal row counts on 2026-06-16/17/19; cause
   unverified. The tape is unaffected.
