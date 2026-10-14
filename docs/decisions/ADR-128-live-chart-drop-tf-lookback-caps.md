# ADR-128: Live Chart drop per-TF lookback caps (MAX_BARS only)

## Status
Accepted — implemented 2026-09-15. Live pin unchanged.

Supersedes the per-timeframe lookback caps in
[ADR-105](ADR-105-live-chart-custom-lookback.md) decision 4,
[ADR-116](ADR-116-live-chart-one-minute-range-bar-ceiling.md) decision 3,
and [ADR-124](ADR-124-live-chart-lookback-date-range.md) decision 4
(cap clause only). Retains the global five-year parse ceiling and the
50,000-candle response ceiling from ADR-116 decision 2. The 36-hour
raw-tick cap for OFI/VWAP prints is unchanged.

## Date
2026-09-15

## Context

Custom lookbacks (`20d`, `2026-05-28..2026-06-01`) silently shrank on
`15s` (1d) and `5s` (4h) via `_TF_MAX_LOOKBACK`. Operators saw only the
right-hand tail of a date range (e.g. June 1 alone on 15s) or a blank
pane when that tail was overnight-only under ETH Display off. `1min`
was already uncapped (ADR-116); short TFs were not.

## Decision

1. **Remove `_TF_MAX_LOOKBACK`.** `window_for` no longer shrinks the
   requested duration or date range by timeframe. Meta
   `tf_max_lookback` is an empty map (compat).
2. **Keep `MAX_BARS = 50_000`.** Series responses still retain the
   newest 50,000 candles when the window is denser than that.
   Payload adds `bars_capped` / `max_bars`; `lookback_capped` mirrors
   `bars_capped` for older clients. Status shows
   `showing newest 50,000 bars` when truncated.
3. **Keep the 5-year parse ceiling** (`MAX_LOOKBACK_DELTA`) and the
   36-hour print/tick overlay cap.

## Consequences

- `15s` / `5s` honor the full Custom From/To or duration (until
  `MAX_BARS` clips). Multi-day 5s still scans `databento.tbbo` and may
  be slow or time out on very wide windows.
- UI Custom… help no longer cites per-TF duration caps.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-105](ADR-105-live-chart-custom-lookback.md) — custom duration
- [ADR-116](ADR-116-live-chart-one-minute-range-bar-ceiling.md) — MAX_BARS
- [ADR-124](ADR-124-live-chart-lookback-date-range.md) — date range
- Module: `src/webui/live_chart.py`
