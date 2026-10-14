# ADR-112: Live Chart stacked panes (volume under price, oscillator panes, zero line)

## Status
Accepted — implemented 2026-09-04. Live pin unchanged.

Amends [ADR-087](ADR-087-live-chart.md) (single-pane layout) and
[ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) (CVD on the
left axis of a shared lower pane with volume; custom `#lc-split`
divider).

## Date
2026-09-04

## Context

Live Chart stacked NQ candles on the top of one Lightweight Charts
widget and jammed **volume + CVD** into the same lower band via
`scaleMargins`. Volume had no y-axis ticks. CVD sat on the left scale
of that shared band, so:

1. Volume was below CVD, not directly under price.
2. Oscillators (CVD, stoch, MFI, …) shared one fake lower pane.
3. Dragging a y-axis could not scale one subgraph without moving
   others that shared `scaleMargins`.

Operators wanted volume as part of the price chart (directly under the
candles) with its own volume labels, CVD as a **separate** chart item
below that, and the same treatment for any future overlay that is not
drawn on price (stoch, etc.): divider, own y-axis, independently
draggable scale. Signed graphs need a visible **zero** floor line.

Lightweight Charts **v4.2.3** (the vendored standalone) has no pane
API. v5 adds `addSeries(..., paneIndex)`, native separators, and
per-pane price scales.

## Decision

1. **Library.** Vendor `lightweight-charts.standalone.production.js`
   is **v5.2.1**. Series are created with `chart.addSeries(Type, opts,
   paneIndex)`. Backtest Chart uses the same helpers (`lwAddSeries`,
   `lwSetMarkers`) on pane 0.
2. **Pane 0 — price.** NQ candles on the right y-axis. Price overlays
   stay here: VWAP, SAR stars, Volume Profile, Large prints, ETH
   shade. Canvas drawings clip to this pane. Dragging the right axis
   scales price only.
3. **Pane 1 — volume.** Histogram always on, directly under price,
   own right y-axis (`priceFormat: volume`), dashed **0** floor,
   autoscale min locked at 0. Native pane separator between price and
   volume. Stretch factor persisted in `localStorage` `lc_pane_stretch`.
4. **Pane 2+ — oscillator overlays.** Every catalog entry with
   `pane: osc` (CVD, delta, OFI, OFI_CUM, VWAP distance, MFI, A/D,
   CLV, stoch, …) gets its **own pane** below volume when enabled,
   with a divider and its own right y-axis. Dragging that axis does
   not move price, volume, or sibling oscillators. Future non-price
   overlays inherit this by setting `pane: osc` in `OVERLAYS`.
5. **CVD.** No longer `pane: left` on a shared band with volume.
   `pane: osc` + `zero_line: 1`. Line / candles / both still apply,
   on CVD’s pane only.
6. **Zero line.** Volume always. Any overlay with `zero_line: 1`
   (signed series: CVD, delta, OFI, OFI_CUM, A/D, CLV, VWAP
   distance) draws a dashed y=0 line with an axis label and keeps 0
   inside the autoscale range. 0–100 oscillators (MFI, stoch) keep
   their own range (MFI still has 20/80).
7. **Resize.** Native pane stretchers replace `#lc-split`.
   `handleScale.axisPressedMouseMove.price` stays on so each pane’s
   y-axis is independently compressible. Initial sizes use
   `pane.setHeight`; drag the separator to change the share.
8. **Vendor.** The standalone script is loaded as
   `/vendor/lightweight-charts.standalone.production.js?v=5.2.1`.
   Panes are created with `chart.addPane` and series are moved with
   `moveToPane`. Each pane’s right scale is
   `chart.priceScale("right", paneIndex)` so volume / CVD / price
   never share one axis. A cached v4 library silently stacked every
   series on pane 0 (one y-axis, CVD on volume).

## Consequences

- Live pin unchanged.
- Overlay catalog `pane: left` is gone; treat `osc` as “own pane
  below volume”.
- Operators must **restart the WebUI process** and hard-refresh
  `#/chart` so the v5 vendor file and new `OVERLAYS` load.
- Custom `lc_pane_split` is unused; pane sizes are `lc_pane_stretch`.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — CVD
  origin; superseded lower-pane layout
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
- Vendor: `src/webui/static/vendor/lightweight-charts.standalone.production.js`
