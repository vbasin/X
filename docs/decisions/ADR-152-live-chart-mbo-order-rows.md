# ADR-152: Live Chart MBO order rows on double-click

## Status
Accepted — 2026-09-30.

**Extends:** [ADR-145](ADR-145-live-chart-mbo-order-map.md) (order map). The window cap is the same 5 days as [ADR-150](ADR-150-live-chart-mbo-stats-and-span.md).

## Date
2026-09-30

## Decision

1. **Open.** Double-clicking a drawn order (the line, or the filled circle, cancel cross, exchange-clear mark, or iceberg mark at its end) opens a dialog of that order's `databento.mbo` rows. Reload-chain bands are not an order and do not open it. Choosing Crosshair is unchanged. Esc, Close, or the backdrop dismisses the dialog; leaving the chart does too.
2. **Query.** `GET /api/chart/mbo-order` filters `order_id` and `symbol`. `databento.mbo` is partitioned on `ts_recv` and has no `order_id` index, so the scan is also bounded to the order's displayed life (every segment of that id), padded by 2 seconds and clamped to coverage. A life longer than 5 days keeps the newest 5 days (`truncated`). Rows are ordered by `ts_event`, then `sequence`, with the same action tie-break the order map already uses when those two are equal. `order_id` is returned as a string.
3. **Dialog.** The table lists every column of the row. Times are shown in America/Chicago with microseconds; the UTC timestamp is the cell tooltip. The user scrolls inside the dialog. A hit on the 20,000-row cap says so.

## Consequences

- A double-click still decompresses the `ts_recv` chunks that cover the order's life. A short-lived order is a short range. An order whose drawn life is several days costs more, on the same order as a wide order-map scan, and the statement limit is 90 seconds.
- Restart the web UI so the new route is served. Static cache-bust: `?v=20260930-mbo-detail`.
