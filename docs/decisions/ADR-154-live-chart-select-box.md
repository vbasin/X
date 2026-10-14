# ADR-154: Live Chart Select draws a box

## Status
Accepted — 2026-09-30.

**Extends:** [ADR-141](ADR-141-live-chart-sar-params-and-notes.md) Select cursor. The note store and Create Note flow are unchanged aside from the optional price box on `range`.

## Date
2026-09-30

## Decision

1. **Box.** Select drag records both axes of the pane under the pointer. Horizontal is the plot-time span (same as before). Vertical is that pane's value scale: candle price on pane 0, volume on pane 1, the oscillator scale on any later pane. The highlight is that rectangle. It stays inside the pane where the drag started, including when the pointer crosses into another pane.
2. **Anchor.** The box is stored as `priceLo` / `priceHi` and converted back through the pane series on each paint, so a time or price zoom moves it with the data. A time-only band (top of the chart to the bottom) is no longer the selection shape.
3. **Zoom.** Zoom in / zoom out still fit or scale the time span. When the box has a non-zero value span, the same click sets that pane's price scale to the box (fit, then halve or double), which turns autoscale off until the axis is double-clicked.
4. **Notes.** `range` still requires `start` and `end`. A box also stores `price_lo`, `price_hi`, and `pane`. Older notes without those fields stay valid.

## Consequences

- Static cache-bust: `?v=20260930-select-box` on `style.css` and `app.js`. No service restart.
- Create Note is still disabled until the box covers more than one plot time.
