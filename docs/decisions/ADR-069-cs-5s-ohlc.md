# ADR-069: 5s candle OHLCV on the `cs_` parquet (`cs_open` … `cs_return`)

## Status
Accepted — implemented 2026-08-27.

## Date
2026-08-27

## Context

The 5s feature matrix had no model-facing open / high / low / volume / range
/ direction / return. `_close` (ADR-009) is the last print in the bar, but it
is excluded from training (`_` prefix). Shape features under `cs_` were
1-minute (or slower) only.

## Decision

1. Eight 5s-native `cs_` columns from TBBO prints on the parquet grid
   (left-labeled `[T, T+5s)`):
   - `cs_open` / `cs_high` / `cs_low` / `cs_close` / `cs_volume`
   - `cs_range` = high − low
   - `cs_direction` = `sign(close − open)` (`0` when equal)
   - `cs_return` = `(close − open) / open`

2. **`cs_close` is `_close`.** Labels and backtest still use `_close`. The
   feature catalog twin is `cs_close` (same series, included in training).

3. **Empty bars** (no prints): volume `0`; open = high = low = close = last
   `_close`; range / direction / return = `0`.

4. **Do not stale-forward** these eight columns with 1-min `cs_*`
   (`_SLOW_FEATURE_PREFIXES`). High/low are expanded to include `cs_close`
   when tick `last()` and `_close` disagree on duplicate `ts_event`.

5. **Patch** is server-side `time_bucket('5 seconds', ts_event)` + `first` /
   `max` / `min` / `sum(size)` — no raw tick pull.
   `python scripts/features/patch_parquets.py --modules cs5s`

6. Named views that select `cs_` (`sequence_baseline`, `tbm_topgroups`) pick
   these up automatically.

## Consequences

- Catalog v5.14 248 → **v5.15 256**. `cs_` 25 → 33.
- Frozen `feature_schema.json` bundles are unchanged until retrained.
- Live pin unchanged.

## Related

- [ADR-009](ADR-009-true-5s-close-tbbo.md) — `_close` definition
- [FEATURE_CATALOG.md](../reference/FEATURE_CATALOG.md) §15
- `src/features/candle_structure.py` (`compute_5s_ohlc`)
