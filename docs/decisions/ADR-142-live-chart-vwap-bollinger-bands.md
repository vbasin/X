# ADR-142: Live Chart VWAP Bollinger bands (±2σ)

## Status
Accepted — implemented 2026-09-24. Live pin unchanged.

## Date
2026-09-24

## Context

Operators wanted Bollinger-style bands around the Live Chart VWAP line on
the price pane. The native 5m hive already defines volume-weighted σ around
session VWAP (`σ = sqrt(E[price²] − VWAP²)`, upper/lower = `VWAP ± 2σ`,
[ADR-090](ADR-090-native-tf-5m-features.md)). The band window should follow
the existing VWAP origin reset (Session / BOD / MKT), not a separate lookback
length.

## Decision

1. **Control.** Under VWAP’s ▸ params on `#/chart`: checkbox
   **Bollinger bands (±2σ)**. Persists as `localStorage.lc_vwap_bb`.
2. **Request.** `/api/chart/series` accepts `vwap_bands=1`. When set, the
   response includes `overlays.vw_bb_upper` and `overlays.vw_bb_lower` as
   price levels (not catalog overlay keys — avoids the 8-overlay cap).
3. **Math.** Same volume-weighted σ as ADR-090 / `native_tf_5m._vwap_5m`,
   over the selected VWAP origin window. `load_vwap_bars` now aggregates
   `SUM(price²·size)` alongside `pv` / `size`.
4. **Draw.** Dashed gold lines on the NQ price pane, sharing VWAP’s origin
   break gaps.

## Consequences

- VWAP + bands share one origin control; no separate length knob.
- Chart-Model / Notes snapshots may carry `vwap_bands`.

## Related

- [ADR-090](ADR-090-native-tf-5m-features.md) — VWAP ± 2σ hive definition
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — VWAP on Live Chart
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
