# ADR-141: Live Chart SAR AF params + scenario Notes

## Status
Accepted — implemented 2026-09-24. Live pin unchanged.
The Select gesture's full-height time band is extended by [ADR-154](ADR-154-live-chart-select-box.md): the drag is a box in time and the pane's value scale.

## Date
2026-09-24

## Context

1. Parabolic SAR on `#/chart` used fixed Wilder AF defaults (`0.02 / 0.02 / 0.2`)
   while the Backtest EXT Parabolic SAR card already exposes `start`,
   `increment`, and `maximum`. Operators wanted the same knobs on Live Chart,
   with the EMA / Volume Profile pattern (checkbox + refresh + parameter card).

2. Operators also needed a way to brush a chart range, capture every visible
   overlay and its parameters, attach a short text note, and revisit those
   snapshots later from a dedicated Notes page.

## Decision

1. **SAR parameter card.** The SAR overlay group on Live Chart gains a
   refresh button and ▸ panel with one card: SAR start (AF), increment, and
   maximum AF — bounds and defaults match `PSAR_PARAM_SPEC` /
   `sar_on_closed`. Specs persist in `localStorage` (`lc_sar`, `lc_sar_params`).
   `/api/chart/series` accepts `sar` JSON `{start, increment, maximum}` and
   echoes applied values in the response when `ps_sar` is selected.

2. **Select cursor + Notes.** Live Chart toolbar adds Crosshair / Select and
   **Create Note**. Select disables chart pan, click-drags a time range on the
   candles, then Create Note opens a text dialog. Saving POSTs to
   `/api/chart/notes` with name = `{contract} {range CT}`, user text, and a
   full chart snapshot (timeframe, lookback, overlays, origins, EMA specs,
   VP specs, SAR params, display prefs). Notes live under
   `ops/webui/chart_notes/` and are listed at `#/notes` (sidebar **Notes**).

## Consequences

- Live Chart SAR stars can match EXT Parabolic SAR AF settings.
- Scenario research notes survive browser clears (server-side store).
- Chart-Model Design chart requests may carry `sar` alongside `emas`.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — nested params pattern
- [ADR-132](ADR-132-live-chart-multi-ema-overlay.md) — multi-EMA UX this SAR card mirrors
- Module: `src/webui/live_chart.py`, `src/webui/chart_notes.py`, `src/webui/static/app.js`
