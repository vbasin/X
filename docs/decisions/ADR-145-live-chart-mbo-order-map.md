# ADR-145: Live Chart Order map (MBO) — large resting orders and reload chains

## Status
Accepted — implemented 2026-09-26. The 48-hour compute cap and the always-on key bar are superseded
by [ADR-150](ADR-150-live-chart-mbo-stats-and-span.md) (2026-09-30): the window is the newest 5 days,
and the stats line sits with the cursor readout. Double-clicking a drawn order opens its MBO rows
([ADR-152](ADR-152-live-chart-mbo-order-rows.md)). Iceberg limit orders can be hidden from the
Order map parameter-card **Icebergs** checkbox (default on; [ADR-149](ADR-149-mbo-exchange-purge-reconciliation.md)).
Live pin unchanged.

## Date
2026-09-26

## Context

`databento.mbo` (order-level events, NQ lead contract, ADR-less table: `DATABASE_SCHEMA.md` §2.9) was
ingested 2026-09-25. TBBO large prints only show *executed* trades (a 50-lot resting order that is hit
in six clips appears as six small prints, and one that never trades is invisible). MBO shows every order
from placement to removal, so large **resting** size, how it interacted with price, and re-added size at
one price after it was filled become visible.

## Decision

1. **Overlay `lt_mbo_orders` ("Order map (MBO)")**, group Order flow, `pane: canvas`, `needs: mbo`,
   `params: mbo` — a canvas overlay like `lt_prints`, with its own parameter card. It is excluded from the
   series builder (`live_chart.py`) and from the crosshair / oscillator paths in `app.js`.
2. **Drawing** (`static/lc_orders.js`, one classic script; `app.js` only calls `paramsHtml` / `load` /
   `draw` / `clear` with a helper `bridge`): one horizontal line per order from placement to removal,
   bid green / ask red, thickness ∝ log2(size), size printed on large long-lived orders, `●` filled,
   `✕` cancelled, `▶` resting (dotted extension to the right edge), `◌` pre-existing at a UTC-midnight
   snapshot, dotted connectors for orders that changed price, shaded bands for reload chains, 10/50-point
   price grid with labels, key bar above the chart, hover tooltip, `▲/▼` edge markers for orders outside
   the visible price range.
3. **Vertical scrolling.** Wheel over the price pane moves the price axis (`priceScale.setVisibleRange`,
   which disables autoscale), Ctrl+wheel zooms it, Shift+wheel is left to the chart; key-bar buttons
   *Fit candles*, *Fit all orders* and `▲ n` / `▼ n` (jump to the nearest order beyond the view).
4. **Data.** `GET /api/chart/mbo-orders` (`src/webui/mbo_orders_api.py`) clamps the window to the MBO
   coverage. The original cap was the newest 48 h; [ADR-150](ADR-150-live-chart-mbo-stats-and-span.md)
   raises it to the newest 5 days. The route picks the contract by trade count and reuses `src/webui/mbo_orders.py`
   (lifecycle, outcome and chain logic, shared with `scripts/research/mbo_large_orders.py`, DB-free and
   tested) and returns unix-second JSON. The heavy step (two window scans, ~15-40 s for a full session) is
   cached in memory per (symbol, window, clip size); every parameter change after that is 0.1-0.2 s.
   The window start is snapped to 30 min (server and client) because the chart window is "now"-relative
   and refreshes constantly — without this every refresh was a cache miss and results were discarded.
5. **Reload chain** (definition, not the noisy level-turnover count): an order ≥ *clip* lots is fully
   filled and a new order of similar size (×0.5–2) appears at the same side and price within *dt* s,
   ≥ *n* times, lasting ≥ *secs* and absorbing ≥ *abs* contracts (defaults 5 lots / 2 s / 3 / 10 s / 50).
   Ask chain = passive selling; bid chain = passive buying. `pinned` / `HELD` / `BROKE` come from 1-min
   trade candles. This is behaviour, not proof of intent.
6. **Determinism.** CME records can share `(ts_recv, sequence)`; "filled" is decided from the order's own
   set of fills, never record order, and the SQL has a total ordering. Regression-tested.

## Consequences

- MBO is **not live**: it is not in the live streamer and Databento serves it ~8 h behind, so the newest
  data lags real time (the key bar shows "MBO through …"). Real-time needs a live MBO subscription (licence
  and storage decision, ~7 GB/month) and is out of scope here.
- `vtech-mbo-daily.timer` (2026-09-26) keeps `databento.mbo` current inside Databento's free ~1-month
  window; older history cannot be re-pulled at $0.
- Only displayed size is visible (icebergs appear as many orders). Approach distance uses 1-min trade
  closes as the price path.
- Tests: `tests/test_mbo_orders.py`, `tests/test_mbo_orders_api.py` (no database).

## Related

- [DATABASE_SCHEMA.md §2.9](../architecture/DATABASE_SCHEMA.md) — `databento.mbo`
- [ADR-111](ADR-111-live-chart-large-prints.md) — TBBO large prints (the overlay this follows)
- `research_cases/mbo_liquidity_heatmap/results/REPORT.md` — the offline renderings this reproduces
