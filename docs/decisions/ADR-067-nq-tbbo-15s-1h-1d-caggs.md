# ADR-067: NQ TBBO 15s / 5m / 1h / 1d continuous aggregates

## Status
Accepted — implemented 2026-08-27.

## Date
2026-08-27

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs native 15s / 1m / 5m / 1h / 1d
features. 1-minute bars already exist as `databento.tbbo_1min`. Databento
OHLCV schemas are only 1s / 1m / 1h / 1d (no 15s, no 5m), and vendor 1d
is UTC midnight — rejected for Chicago session work in ADR-065.

## Decision

1. **`databento.tbbo_15s`** — cagg on `databento.tbbo`,
   `time_bucket('15 seconds', ts_event)`, same OHLCV + last BBO +
   buy/sell volume columns as `tbbo_1min`. Left-labeled.
2. **`databento.tbbo_5m`** — nested cagg on `tbbo_1min`,
   `time_bucket('5 minutes', bucket)`. Clock 5-minute bars (unix epoch
   `% 300 == 0`). Added 2026-08-27 after 15s/1h/1d.
3. **`databento.tbbo_1h`** — nested cagg on `tbbo_1min`,
   `time_bucket('1 hour', bucket)`. Clock hours (UTC hour = CT hour).
4. **`databento.tbbo_1d`** — cagg on `tbbo` ticks with
   `time_bucket('1 day', ts_event, 'America/Chicago', origin 17:00 CT)`.
   Session **T** is **[T-1 17:00, T 17:00) America/Chicago**, left-labeled
   at session open. CME is dark 16:00–17:00 CT, so the last trade is at
   or before T 16:00 CT.
5. NQ outrights only: `symbol ~ '^NQ[HMUZ][0-9]+$'`. No ES, no calendar
   spreads.
6. Backfill 2025-04-06 00:00 UTC through session 2026-08-25 (exclusive
   end 2026-08-25 17:00 CDT). Auto-refresh policies keep them current
   after that.

Nested 1d on `tbbo_1min` is not allowed (Timescale rejects a different
bucket origin than the inner 1-minute cagg). Nested 5m and 1h are allowed.

## Consequences

- Does not change EXT `1day` (ADR-065: Chicago calendar day labeled
  08:30 CT). This warehouse daily bar is the Globex session.
- Does not change `market_session.py` (4:00 PM / 3:00 PM CT used by
  VWAP / CVD).
- Live pin unchanged. Feature parquets are a later step.

## Related

- [ADR-041](ADR-041-closed-bar-slower-features.md) — left-labeled buckets
- [ADR-065](ADR-065-ext-hour-day-timeframes.md) — EXT 1hour / 1day
- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — native TF objective
- Script: `scripts/data/create_tbbo_tf_caggs.py`
