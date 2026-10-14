# ADR-114: Live Chart replay playhead

## Status
Accepted — implemented 2026-09-04. Live pin unchanged.

Amends [ADR-087](ADR-087-live-chart.md) (Live Chart display).

## Date
2026-09-04

## Context

Operators want to scrub the Live Chart as if it were printing live:
hide every candle, overlay, print, and volume-profile calc **after** a
chosen time, then slide that time to the right to watch the tape unfold.

## Decision

1. **Control.** Display **Replay**. Live polling pauses while it is on.
2. **Playhead.** A gold vertical line on the plot (drag the line or
   the sidebar slider). Default is the last bar. The **time axis**
   still pans and zooms — Replay does not capture axis pointer
   events. Scrubbing freezes the visible time range so `setData`
   cannot scroll the chart.
3. **Clip.** Bars after the playhead are Lightweight Charts whitespace
   so the x-axis does not jump. Overlay series, SAR, large prints, and
   night shade stop at the playhead. Price scale autoscales on visible
   data only.
4. **Volume profile.** Causal series clip immediately. VP / SVP tape is
   recomputed as-of the playhead via `GET /api/chart/series?asof=`
   (debounced). Candle lookback is unchanged; `asof` only cuts profile
   windows and the current-SVP reference bar.
5. **Off.** Clearing Replay restores the full loaded window.

Not a hive column. Live pin unchanged.

## Consequences

- Hard-refresh `#/chart`. Replay is off by default.
- Single VP under Replay with a blank date is the session of the
  playhead bar, tape through that instant — not the session of “now”.
  A pinned `session_date` stays on that date; `asof` still clips the
  window.

## Related

- [ADR-087](ADR-087-live-chart.md)
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md)
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) —
  Single VP date + Custom session clocks
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
