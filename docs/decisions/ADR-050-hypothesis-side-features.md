# ADR-050: Dedicated side model on vol / VWAP slope / MA / stochastic

## Status
Accepted as a **negative result** — measured 2026-08-21.
Does **not** promote. Live pin remains `nq-50pt-econ`.

## Date
2026-08-21

## Context

ADR-049 scanned all 216 columns and trained a ±10 TBM XGB on the full
matrix. The operator’s objection: that is not the same as testing the
usual **side** stories in isolation — high vol → downturn, VWAP slope,
moving-average crossovers — as a **small dedicated model**.

Parquet has no EMA-crossover column. A causal 1-minute 9/21 EMA cross
was built from `_close` (per Chicago day, 1-min last, visible at T+1m).

## Decision

Walk-forward binary XGB on **only**:

`rv_realized_vol_60`, `rv_vol_ratio`, `vw_vwap_slope_24/120`,
`vw_vwap_dist`, `so_sync_score_9_60`, `so_fast_d_9_3`,
`so_full_d_60_10`, `ht_daily_trend_10/20`, `cs_clv_ma_10`,
`cs_body_direction`, plus `ema_cross_9_21`.

Train on 50/6 directional bars (open, 10-minute uniqueness). Combine
with the ADR-047 timer. Also publish quintile P(up|dir) / P(down|dir)
so the named rules are visible without the tree.

## Results (`20260821T121631Z`)

Open-session 50/6 **directional** bars (~58k), quintiles:

| Feature | AUC (↑ → more **up**) | Low-q P(up)/P(down) | High-q P(up)/P(down) |
|---------|----------------------:|---------------------|----------------------|
| `rv_realized_vol_60` | **0.553** | 0.385 / **0.615** | **0.521** / 0.479 |
| `vw_vwap_slope_120` | 0.495 | 0.490 / 0.510 | 0.477 / 0.523 |
| `vw_vwap_dist` | 0.492 | 0.488 / 0.512 | 0.483 / 0.517 |
| `so_sync_score_9_60` | 0.519 | 0.432 / 0.568 | 0.481 / 0.519 |
| `ema_cross_9_21` | 0.491 | — | 0.469 / 0.531 overall |
| `ht_daily_trend_10` | 0.493 | 0.487 / 0.513 | 0.494 / 0.506 |

**High realized vol is associated with more 50/6 *ups*, not downs.** Quiet
tape is where the rare directional labels skew down. VWAP slope, VWAP
distance, stochastic sync, daily trend, and a causal 9/21 EMA cross are
all coin-flips.

Small-model OOF AUC **0.499** (folds 0.46–0.53). Combined with the
timer: n=49 HR 0.265 NE **7.86** L/S 26/23 vs long-only n=57 HR 0.333
NE **11.67** L/S 57/0.

## Consequences

- The named technical-side stories are implemented and measured. They do
  not call TBM +50/−6 side on this L1 sample.
- Causal policy remains the long-only open timer (ADR-047).
- `--hypotheses` on `python -m src.search.side_head` reproduces the run.

## Supersedes

None. Answers the “you didn’t test vol / VWAP slope / MA as a side
model” objection to [ADR-049](ADR-049-tbm10-side-transfer-rejected.md).
