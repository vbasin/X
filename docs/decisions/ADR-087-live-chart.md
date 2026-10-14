# ADR-087: Live Chart tab (candles + overlays; MBP-10 optional)

## Status
Accepted — implemented 2026-08-28. Decision §4 (MBP-10 ladder) superseded by [ADR-148](ADR-148-live-chart-drop-mbp10-ladder.md) (2026-09-29). Live pin unchanged.

## Date
2026-08-28

## Context

The WebUI **Border Visualizer** (`#/border`) was an MBP-10 heatmap + ladder
over `databento.mbp_10`. Operators also need a standard NQ candlestick at
an arbitrary bar width, with TBBO-native feature overlays (e.g. session CVD
on 1-minute bars). L2 is 13–19M events/day and is the wrong default.

## Decision

1. **Tab.** Sidebar **Live Chart** (`#/chart`). `#/border` still opens the
   same view. Instrument is always **/NQ** (lead-month).
2. **Candles.** `GET /api/chart/series` returns OHLCV from Databento TBBO
   caggs (`tbbo_15s` / `tbbo_1min` / `tbbo_5m` / `tbbo_1h` / `tbbo_1d`).
   `5s` is `time_bucket` on `databento.tbbo`. `15min` is resampled from 5m.
   Lookback is a duration string (presets + custom; [ADR-105](ADR-105-live-chart-custom-lookback.md))
   or a CT date range (`YYYY-MM-DD..YYYY-MM-DD`; [ADR-124](ADR-124-live-chart-lookback-date-range.md)).
   No per-TF lookback caps ([ADR-128](ADR-128-live-chart-drop-tf-lookback-caps.md));
   responses clip at **50,000** newest bars. Default is
   **1 minute / Globex session / live poll**.
3. **Overlays.** Curated TBBO-computable series from the feature modules
   (`of_delta` / `of_cvd` / `of_ofi` / `vw_vwap` / `vw_vwap_dist` /
   `cs_clv` / `mf_mfi` / `ad_line` / `so_fast_d_9_3` /
   `so_sync_score_all` / `ps_sar` as stars). Tape
   overlays load prints for the same window (≤ 36h). Options IV is not
   loaded. **ADR-097 / ADR-108:** `of_cvd` / `of_delta` use cagg buy−sell
   for the full lookback. CVD, OFI_CUM, and A/D each have their own
   origin clock. Volume Profile is a nestable overlay (Single VP /
   recurring; session timeframe including Custom From/To clocks +
   placement; [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)).
   Not hive columns.
4. **MBP-10.** Optional ladder + depth imbalance via existing
   `/api/book/snapshot`. Heatmap playback is not the default surface.
   Superseded by [ADR-148](ADR-148-live-chart-drop-mbp10-ladder.md): the
   ladder, the depth-imbalance panel, and the `/api/book/*` routes are gone.
5. **Catalog.** Overlay list is `src.webui.live_chart.OVERLAYS`.
6. **Crosshair readout.** A color-coded strip above the candles plus a
   hover box at the cursor show OHLC, volume, and each enabled overlay
   value at that bar (CVD, VWAP, …), plus horizontal price and SVP/VP
   volume-at-price when those overlays are on. VP rows are only the
   profile whose tape window contains that bar, not earlier recurring
   sessions ([ADR-125](ADR-125-live-chart-hover-session-vp.md)). The hover box can be
   dragged onto any chart corner (snaps tl/tr/bl/br); double-click the
   handle to follow the cursor again.

## Consequences

- Border Visualizer heatmap is no longer the tab body. Book heatmap APIs
  remain for a later optional panel. ADR-148 removes those routes and the
  ladder panel; `src/webui/book.py` stays unmounted.
- Live pin unchanged.

## Related

- [ADR-057](ADR-057-mbp10-5s-mean-depth-imbalance.md) — MBP-10 stats
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — candle caggs
- [ADR-068](ADR-068-tbbo-event-ofi.md) — OFI
- Module: `src/webui/live_chart.py`
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — CVD origin
  + original SVP / range VP
- [ADR-105](ADR-105-live-chart-custom-lookback.md) — wider presets +
  custom lookback prompt; raises per-TF caps
- [ADR-124](ADR-124-live-chart-lookback-date-range.md) — Custom… Date
  range (From/To CT calendar dates)
- [ADR-128](ADR-128-live-chart-drop-tf-lookback-caps.md) — drop per-TF
  lookback caps; `MAX_BARS` only
- [ADR-125](ADR-125-live-chart-hover-session-vp.md) — hover VP is the
  covering session only
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) —
  nested overlay params; CVD vs OFI origins; Volume Profile SSOT
  (folds 110 / 113 / 115)
- [ADR-109](ADR-109-live-chart-display-fit-and-eth.md) — viewport fit,
  compact x-axis, value-area wash extend, night-session display/shade/aggregate,
  chart text size
- [ADR-114](ADR-114-live-chart-replay.md) — Replay playhead clips the
  chart as-of a draggable time cursor
- [ADR-111](ADR-111-live-chart-large-prints.md) — Large prints overlay
  (TBBO fills ≥ min size as price bubbles; bar-center + frac×barSpacing
  lock, outline, rails, hover)
- [ADR-112](ADR-112-live-chart-stacked-panes.md) — volume pane under
  price; oscillator overlays each get their own pane + y-axis; zero
  floor line on signed series. Supersedes the single-widget
  scaleMargins lower band.
