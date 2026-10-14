# ADR-132: Live Chart multi-EMA price overlay

## Status
Accepted — implemented 2026-09-17. Live pin unchanged.

## Date
2026-09-17

## Context

Operators wanted one or more EMA lines drawn on Live Chart candles, each
with its own lookback in bars, using the same multi-instance UX as Volume
Profile (category checkbox + Add dropdown + cards + Apply). Fixed catalog
overlay keys (`ema_9`, `ema_21`, …) cannot express an arbitrary period or
duplicate periods cleanly.

## Decision

1. **Multi-instance EMA category** (not a fixed `OVERLAYS` row). UI mirrors
   Volume Profile: `#/chart` overlays panel gains an **EMA** group with
   enable checkbox, preset/Custom Add dropdown, per-EMA cards, and Apply
   (refresh). Specs persist in `localStorage` (`lc_ema`, `lc_ema_specs`).
2. **Request/response contract.** `/api/chart/series` accepts
   `emas` JSON: `[{id, n, lookback}, …]` (max 8). Response includes
   `emas` metadata and parallel arrays under `overlays["ema_{n}"]`.
3. **Lookback includes the current bar.** EMA is standard span-N
   `ewm(span=N, adjust=False, min_periods=N)` on chart closes. The bar
   whose value is plotted is one of the N closes in the window
   (`lookback=1` equals close). Early bars before a full window are
   null.
4. **Source bars are the continuous series** ([ADR-137](ADR-137-continuous-lookback-adjacency.md)
   supersedes the earlier Aggregate-filter rule). EMA, MFI, stochastic, and
   SAR are computed on calendar-adjacent bars, including overnight and bars
   before the visible window, then aligned onto the displayed candles.
   Aggregate still filters origin-reset flow (CVD / OFI / VWAP) and VP tape.
   CLV stays per-bar.

## Consequences

- Operators can stack EMA9 / EMA21 / custom lengths on the price pane
  without catalog churn.
- Chart-Model Design chart requests may carry `emas` so simulations see
  the same lines (variable catalog still OHLC/VP-centric).
- Not a hive feature column — display overlay only.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — multi
  Volume Profile pattern this UX follows
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
