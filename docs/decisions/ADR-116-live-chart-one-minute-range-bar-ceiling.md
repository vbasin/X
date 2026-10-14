# ADR-116: Live Chart one-minute range and bar ceiling

## Status
Accepted — implemented 2026-09-07. Live pin unchanged.

**Superseded in part by [ADR-128](ADR-128-live-chart-drop-tf-lookback-caps.md):**
decision 3 (other TFs keep per-TF caps) is removed; decision 2
(`MAX_BARS = 50_000`) remains the sole response ceiling for all TFs.

Amends [ADR-105](ADR-105-live-chart-custom-lookback.md) decision 4 for
the one-minute timeframe.

## Date
2026-09-07

## Context

The one-minute Live Chart accepted a Custom `24d` duration but silently
clipped it to 14 days, then clipped its returned candles to 12,000 bars.
On a multi-week chart that bar limit excluded the earlier requested candles.

## Decision

1. **One-minute range.** `1min` no longer has a per-timeframe lookback cap.
   The existing global five-year duration ceiling still applies.
2. **Criticality stop.** Live Chart responses retain at most **50,000**
   candles (`MAX_BARS`). The newest candles are retained when the limit is
   exceeded.
3. **Other timeframes.** Existing 5s, 15s, 5min, 15min, 1hour, and 1day
   lookback caps remain unchanged. The 36-hour raw-tick cap for tick-derived
   overlays also remains unchanged.

## Consequences

- `Custom: 24d` at 1 minute returns the complete requested window when it is
  within 50,000 bars.
- A larger one-minute request remains bounded at 50,000 returned candles.
- The API's `tf_max_lookback` map omits `1min`, allowing the UI to distinguish
  an uncapped duration from timeframes with an explicit cap.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-105](ADR-105-live-chart-custom-lookback.md) — custom lookback parsing
- Module: `src/webui/live_chart.py`