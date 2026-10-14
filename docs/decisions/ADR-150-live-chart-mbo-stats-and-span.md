# ADR-150: Live Chart MBO stats row, 5-day window, selection zoom

## Status
Accepted — 2026-09-30.

**Supersedes:** [ADR-145](ADR-145-live-chart-mbo-order-map.md) §3 key-bar placement and §4's newest-48-hour cap.

## Date
2026-09-30

## Decision

1. **Window.** `order_map` keeps the newest 5 days (`MAX_SPAN_H = 120`) of a request, still clamped to `databento.mbo` coverage. The payload includes `max_span_h`. A shortened window is `truncated`.
2. **Stats row.** The order-map summary moves into `#lc-mbo-head`, directly under the cursor readout (`#lc-readout`), at the same font size (`--lc-font`). The label is **MBO stats** (the contract symbol is not shown). Outcome order is filled, resting, pulled in, pulled away. The coverage phrase is the Chicago range `MM/DD/YY hh:mm:ss - MM/DD/YY hh:mm:ss CT` of the served window. A truncated window says `Limit: N day max`.
3. **Order Map button.** It is the last item on the stats row. Pressing it shows or hides the legend (BID/ASK, size scale, markers, fit and jump buttons) on the row underneath.
4. **Zoom.** Icon buttons sit immediately right of Select. With a selection, zoom in fits that range (and halves it if it is already the view); zoom out doubles around it. With no selection, zoom halves or doubles the current view. Choosing Crosshair clears the selection highlight.

## Consequences

- A 5-day order-map request scans more of `databento.mbo` than the 48-hour cap. The in-memory cache key is unchanged. Restart the web UI so the new cap is served.
- Static cache-bust: `?v=20260930-mbo-stats` on `style.css`, `lc_orders.js`, and `app.js`.
