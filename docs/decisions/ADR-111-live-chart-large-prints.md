# ADR-111: Live Chart large-print TBBO overlay

## Status
Accepted — implemented 2026-09-03; draw/hover contract completed
2026-09-04 (folded in; no separate ADR). Live pin unchanged.

Amends [ADR-087](ADR-087-live-chart.md) (overlay catalog) and
[ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
(nested overlay params).

## Date
2026-09-03 (overlay + query). 2026-09-04 (bar-center + frac×barSpacing
lock, outline, price rails, hover, `side=N`).

## Context

Live Chart candles, CVD, and Volume Profile sum **all** TBBO fill
sizes. Operators wanted a **large size filter**: every lead-month
fill with `size ≥ N` in the **current chart lookback**, drawn as a
bubble at that fill’s price with the contract count inside.

Candles come from caggs (`tbbo_1min` / …), not tick rows. OFI/VWAP
tick loads are capped at 36 hours and pull every size. A size filter
must query `databento.tbbo` with `size >= N` over the chart window.

Lightweight Charts v5 draws each candle at logical index `i`
**centered** (`indexToCoordinate` uses `i + 0.5`).
`timeToCoordinate` is exact-match only, so a tick timestamp either
returns null or the **bar center**. Interpolating in pixels between
those centers without subtracting half a bar puts the print on the
**next** candle’s half; zoom changes `barSpacing / 2` so it slides.
A first lock attempt called `logicalToCoordinate(i + frac − 0.5)`.
The vendored v5.2.1 `indexToCoordinate` rejects non-integers
(`t % 1 == 0`) and returns **0**, so every bubble stacked on the
left of the plot. Green/red fill can vanish on a same-color candle.
Hover showed the snapped bar, not the print. `side='N'` packed as
empty and drew as green.

## Decision

1. **Overlay.** Catalog key `lt_prints`, label **Large prints**,
   Order flow group, canvas pane (not a Lightweight Charts series).
   Nested param **Min size (contracts)** (default **10**, same as
   hive `lt_*`). Include if `size ≥ min`.
2. **Window.** `GET /api/chart/prints` uses the same
   `window_for(lookback, timeframe)` as `/api/chart/series` — the
   visible chart range, not the 36h tick cap. `eth_aggregate=0`
   clips to cash 08:30–15:00 CT like other overlays.
3. **Query.** `WHERE ts_event ∈ [start, end) AND symbol = ANY(leads)
   AND size >= min_size`. Lead-month filter matches `load_ticks`.
   Cap **8 000** most-recent matching fills (`prints_capped`).
4. **Cache.** The chart holds rows in memory while the overlay stays
   on and timeframe / lookback / min size / aggregate are unchanged.
   Live polls send `since=<last end>` and append. Changing those
   knobs or leaving `#/chart` drops the cache.
5. **Lock.** Place each fill at
   `timeToCoordinate(barTime) + (frac − 0.5) × barSpacing` where
   `frac` is `(ts_event − bar_open) / timeframe_seconds` clamped to
   `[0, 1]` and `barTime` is that candle’s plot time (same key SAR
   uses). `barSpacing` is the pixel gap between adjacent
   `timeToCoordinate` results (falls back to
   `timeScale.options().barSpacing`). Do **not** call
   `logicalToCoordinate` with a fractional index: the vendored v5.2.1
   helper only accepts integers (`t % 1 == 0`) and otherwise returns
   **0**, which stacked every bubble on the left of the plot. Do
   **not** pass the tick unix to `timeToCoordinate` (exact-match only
   → null or bar-center). Y is `priceToCoordinate(price)` on the
   price pane.
6. **Draw.** Label is the integer size. Black (`#000`) outline so a
   green print on a green candle (or red on red) stays visible.
   Short horizontal **price rails** on the left and right of the
   circle at the fill price. Overnight prints hide when Night
   session Display is off.
7. **Hover.** Crosshair hit-test on the drawn bubbles. The floating
   tooltip (not the top readout strip) shows fill **price**,
   **quantity**, Databento **`ts_event`** (America/Chicago,
   microseconds), and aggressor side. Print rows stay off the
   readout so wrapping chips cannot change `.lc-chart-host` height
   and fire the pane `ResizeObserver` (`applyPaneHeights`), which
   was stealing height from price into CVD/osc panes on every hover.
8. **Side.** `B` = buy aggressor (lifted the ask) green; `A` = sell
   aggressor (hit the bid) red; `N` = unspecified (CME auction /
   Globex 17:00 open / no aggressor) light gray. Payload keeps `ts`
   ISO and `side` in `{B,A,N}`.
9. **`price`.** The **trade/print price**, not `bid_px` / `ask_px`
   (those are the BBO immediately before the trade).

Not hive `lt_*` columns. Does not size-filter CVD or Volume Profile.

## Consequences

- Hard-refresh the console (`?v=` bust).
- A 10-day lookback at min size 1 can hit the 8 000 cap; raise min
  size rather than scanning unbounded tape.
- Fill size, not parent order (CME sweeps split).
- ETH shade edges that go through `timeToX` also sit on bar-open /
  bar-end instants instead of bar centers (same lock).

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) —
  nested overlay params
- [ADR-109](ADR-109-live-chart-display-fit-and-eth.md) — time-axis
  instants vs `barSpacing` half-bar drift
- [ADR-112](ADR-112-live-chart-stacked-panes.md) — price pane canvas
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
