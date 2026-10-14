# ADR-144: Live Chart compact scale tags, hover resize, bound arrows

## Status
Accepted — implemented 2026-09-26. Amended the same day: scale tags must tolerate an empty series, and the cursor readout pins in place. Live pin unchanged.

## Date
2026-09-26

## Context

Live Chart drew each series' last value with Lightweight Charts' native scale
label. Those boxes are taller than a price tick and opaque, so a feature
value coasting along the right axis hid the prices around it. The cursor
readout could be dragged off the pointer, and `preventDefault` on that drag
swallowed the double-click that was supposed to reattach it. Stochastic and
MFI already drew 80/20 lines, and VWAP can draw ±2σ bands, with no cross marks.

## Decision

1. **Compact last-value tags.** Series `lastValueVisible` is off. `#/chart`
   draws one-line tags on the right scale, the same font size as the price
   ticks, right-aligned, with a 2px color bar. The price tag keeps its level.
   Other tags in the same pane step aside when they would overlap, and a 1px
   tick marks the true level when a tag is nudged. Helpers live in
   `src/webui/static/lc_tags.js`.
2. **Cursor readout.** **Pin**, or a right-click on the chart, freezes the
   card where it is (`lc_hover_corner=pin`, position in `lc_hover_pos`).
   Dragging the grip drops it there too; it does not jump to a corner.
   **Follow cursor**, or a double-click on the grip, tracks the pointer
   again. Edges and corners resize the card; the size is `lc_hover_size`.
   Older corner pins (`tl` / `tr` / `bl` / `br`) still dock.
3. **Bounds.** Each Stoch card has Upper / Lower (default 80 / 20) and a
   Bound arrows checkbox, stored on the card in `lc_stoch_specs` (display
   only; not sent in the `stochs` request). MFI has the same controls
   (`lc_mfi_upper`, `lc_mfi_lower`, `lc_mfi_arrows`). VWAP bands gain
   **Band arrows** (`lc_vwap_bb_arrows`).
4. **Cross marks.** When arrows are on: a red arrow on a cross down through
   the upper bound, a green arrow on a cross up through the lower bound.
   A 1px vertical line in the arrow color runs through the arrow from the
   top to the bottom of that feature's pane. VWAP uses the close against the
   ±2σ bands on the price pane. Arrows stay for every cross still in view.

## Consequences

- A series with no points yet is not a bar. `lcFiniteSeriesValue(null)` is
  null. Reading `.value` off that null used to throw while the candle series
  was loading, which skipped the volume `setData` and left the volume pane
  empty under the error text.
- Price ticks stay readable next to feature values.
- A pinned readout can be resized and put back on the cursor without a reload.
- Bound arrows are off until the operator turns them on. Apply (↻) commits
  Stoch card edits. MFI bound edits and the VWAP arrow toggle redraw immediately.

## Related

- [ADR-142](ADR-142-live-chart-vwap-bollinger-bands.md) — VWAP ±2σ bands
- [ADR-143](ADR-143-live-chart-multi-stoch.md) — multi StochasticFast
- Module: `src/webui/static/app.js`, `src/webui/static/lc_tags.js`, `src/webui/static/style.css`
