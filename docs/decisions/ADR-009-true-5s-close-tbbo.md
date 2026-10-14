# ADR-009: True 5-Second Close Price from TBBO

## Status
Accepted

## Date
2026-05-05 (Phase 17H)

## Context

The Parquet column `_close` is the price series used exclusively for label
generation at training time.  Labels (BIG_UP, BIG_DOWN, NO_SIGNAL) are derived
by `src/training/labels.py` via a forward-rolling max/min excursion over the
`_close` series.  The column is not passed as a model input.

### Prior design (v5.0 / Phase 17E)

`_close` was constructed in `engine.py` as:

```python
close_resampled = candles_1m["close"].resample(self.timestep).last().ffill()
combined["_close"] = close_resampled.reindex(combined.index)
```

`candles_1m` comes from `databento.tbbo_1min`, a TimescaleDB continuous
aggregate that materialises 1-minute OHLCV bars.  Its `close` column is the
last trade price of each 1-minute bar.  Forward-filling to the 5-second grid
means the same close value is repeated for all 12 bars within the minute —
only ~180 distinct values across the 2,160-bar session.

### Problem

A 1-minute close lags intrabar price moves by up to 59 seconds.  Consider a
50-point upswing that begins at 13:01:02 and peaks at 13:01:48 UTC:

- At bar 13:01:00 the label's forward excursion window starts from the
  13:00:xx close (the value repeated across all 12 bars of 13:00:00–13:00:55).
- The first time `_close` reflects the move is 13:02:00 (the 13:01 candle
  close), which is already 58 seconds after the move began.
- Label assignment shifts up to 12 bars (60 seconds) from the true event time.

This introduces systematic timing noise into the label series.  The 50-point
MAE threshold is tight enough that these 12-bar shifts can flip a label from
BIG_UP to NO_SIGNAL at the true signal bar, and assign BIG_UP to a bar that
already saw the price action.

## Decision

Replace the 1-min-candle source with the last NQ trade price per 5-second bar
from `databento.tbbo`:

```python
close_5s = (
    tbbo.set_index("ts_event").sort_index()["price"]
    .resample(self.timestep)
    .last()
    .ffill()
)
combined["_close"] = close_5s.reindex(combined.index)
```

`tbbo` (trade-and-best-BBO) is already loaded per date and filtered to the
lead-month NQ contract by `_load_tbbo`.  The `price` column is the actual
trade price; `resample(...).last()` selects the last trade within each 5-second
bar; `.ffill()` fills bars with no trades (uncommon during the 08:00–11:00 CST
session).

A 1-min candle fallback is retained for dates where tbbo is unavailable.

## Consequences

### Positive

- Label timing aligns with actual bar prices: the bar at timestamp T uses the
  price at T, not the price at the end of the minute containing T.
- Distinct `_close` values per session: ~180 (before) → ~2,160 (after), one
  per 5-second bar.
- Removes a systematic up-to-12-bar label lag that likely contributed to the
  BIG_DOWN structural collapse observed in model_v009_3class.

### Negative / Constraints

- All existing checkpoints (model_v007 through model_v009) were trained on
  1-min-close labels.  Changing `_close` invalidates those checkpoints for
  direct comparison; retraining from scratch is required.  This was already
  planned as part of Phase 17H (model_v010_iv_fixed).
- Labels may shift on a small fraction of bars near the 50-point threshold
  boundary.  The magnitude is unknown until training; however, since the
  threshold is 50 points and typical intrabar noise is ~2–5 points, the
  fraction of affected bars is expected to be low (~1–3%).

## Supersedes

Nothing superseded.  This is an additive change to the label data pipeline;
ADR-007 (detection-2 MAE label scheme) and ADR-008 (3-class softmax) remain in
force — only the `_close` price source changes.

## Cross-References

- `docs/reference/FEATURE_CATALOG.md` — `_close` non-feature column entry
- `src/features/engine.py` — implementation of the `_close` construction
- `scripts/patch_close_5s.py` — one-time retroactive patch for Oct–Dec 2025
  parquets
- [ADR-007](ADR-007-detection-2-mae-filter.md) — label scheme (unchanged)
- [ADR-008](ADR-008-3class-softmax-class-weight.md) — 3-class architecture (unchanged)
