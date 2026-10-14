# ADR-137: Continuous lookback adjacency

## Status
Accepted — implemented 2026-09-23. Live pin unchanged. Existing Track C
parquets are not rebuilt by this change.

## Date
2026-09-23

## Context

Lookback features were restarting at the cash open or at the edge of the
visible chart window:

- Live Chart EMA / MFI / stochastic / SAR ran on the RTH-filtered frame when
  Aggregate was off, so Monday 08:30's prior bar was Friday's last cash bar
  ([ADR-132](ADR-132-live-chart-multi-ema-overlay.md) clause 4).
- The same overlays were null for the first `N−1` visible bars because no
  bars before the requested lookback were loaded.
- Track C `fs_ema*` was an `ewm` on the cash-only 1s path, so EMA6 at 08:30
  did not see 08:29 / 08:28. `es_atr_r` kept only 10 minutes of prior 1m
  candles and shards loaded `warmup_days=0`. Path-encoder ATR left the first
  ~60 seconds of each session NaN.

## Decision

1. **Adjacency.** A feature that needs the previous N bars at time T reads
   the timeframe's stored bars ending at T, in time order, including Globex
   and bars outside the visible window or the cash label slice. Visibility
   filters (ETH Display, Aggregate, cash entry window) choose which rows are
   shown or labeled. They do not choose the lookback neighbors.
2. **No invented holes.** A multi-hour gap is not filled with synthetic
   flat seconds. Warm-start uses bars that exist (chart cagg before the
   window; prior `path_*.parquet` tail; pre-cash 1m closes).
3. **Origin resets stay.** CVD, OFI_CUM, VWAP, and A/D still reset at their
   selected Session / BOD / MKT origin.
4. **Live Chart.** `series()` prepends up to the overlay's lookback (EMA
   span, MFI 14, stochastic periods, a short SAR pad) of continuous OHLC
   and computes those overlays on that frame, then aligns onto the displayed
   buckets. Aggregate no longer filters that calc.
5. **Track C.** `build_flow_state(..., warmup_close_1m=)` prepends 1m closes.
   `build_move_matrix` loads two warmup days of 1m bars and keeps 120 minutes
   before the session for `es_atr_r` / `es_realized_vol` and the EMA warm
   start. `path_ohlc_features(..., warmup=)` and `path1s_folds.load_session`
   stitch the prior path tail so open ATR is finite when a prior file exists.

## Consequences

- Operators see an 08:30 EMA that already includes 08:29, 08:28, … even when
  overnight candles are hidden.
- On-disk move matrices and path folds stay stale until rebuilt. This ADR
  does not run that rebuild.
- Supersedes ADR-132 clause 4 (Aggregate filters the EMA source).

## Related

- [ADR-132](ADR-132-live-chart-multi-ema-overlay.md)
- [ADR-109](ADR-109-live-chart-display-fit-and-eth.md)
- Modules: `src/webui/live_chart.py`, `src/features/flow_state.py`,
  `src/training/path_features.py`, `src/search/path1s_folds.py`,
  `scripts/features/build_move_matrix.py`
