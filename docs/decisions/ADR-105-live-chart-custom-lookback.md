# ADR-105: Live Chart wider lookbacks + custom duration prompt

## Status
Accepted — implemented 2026-08-29. Live pin unchanged.

**Superseded in part by [ADR-128](ADR-128-live-chart-drop-tf-lookback-caps.md):**
decision 4 (per-TF lookback caps) is removed; duration parsing and the
Custom… prompt remain. `MAX_BARS` is the response ceiling.

Amends [ADR-087](ADR-087-live-chart.md) decision 2 (closed lookback enum
and per-TF caps). Overlay chrome after [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
nests CVD/OFI params and multi Volume Profile; lookback is unchanged.

## Date
2026-08-29

## Context

Live Chart lookback was a closed set (`30m` / `1h` / `2h` / `4h` /
`session` / `1d` / `5d` / `20d`) and the API rejected anything else.
Per-timeframe caps were tight (5s ≤ 2h, 1min ≤ 5d, 1h ≤ 120d), so a
20-day pick on 1-minute bars silently clipped to five days. Operators
need multi-week / multi-month candles on 5m–1d and a way to type an
arbitrary duration.

## Decision

1. **Presets.** Dropdown adds `8h` / `12h` / `2d` / `3d` / `7d` /
   `10d` / `30d` / `60d` / `90d` / `1y`. `session` is still the
   Globex open (17:00 CT). Default remains **1 minute / session /
   live poll**.
2. **Free duration.** `GET /api/chart/series?lookback=` parses
   `Nm` / `Nh` / `Nd` / `Nw` / `Ny` and the word forms (`45 min`,
   `12 hours`, `2 weeks`). Aliases `globex` / `eth` → `session`.
   Canonical token is echoed on the payload (`45 min` → `45m`).
   Hard ceiling is **5 years**.
3. **Custom prompt.** Toolbar Lookback includes **Custom…**. That
   opens an in-page prompt *Look back how far?* (examples `6h`,
   `3d`, `2w`, `90d`, `1y`). The value is stored in `lc_lb` and
   shown as `Custom: 45m` when it is not a preset.
   Date-range From/To is [ADR-124](ADR-124-live-chart-lookback-date-range.md).
4. **Per-TF caps** (still applied after parse; `MAX_BARS` 12,000
   still clips the response):

   | TF | Cap |
   |----|-----|
   | 5s | 4h (`databento.tbbo` scan) |
   | 15s | 1d |
   | 1min | 14d |
   | 5min | 60d |
   | 15min | 180d |
   | 1hour | 730d |
   | 1day | 1825d |

   Payload `lookback_capped` is true when the window was clipped.
   Tape overlays (OFI / VWAP) stay on the 36h print cap. CVD /
   OFI_CUM still cover the visible window (ADR-097).

## Consequences

- 5s of a full Globex session is still too expensive; pick 15s+ or
  accept the 4h clip.
- Closed-enum clients that only send the old keys keep working.

## Related

- Module: `src/webui/live_chart.py` (`parse_lookback`, `window_for`)
- UI: `#/chart` Lookback select + `#lc-lb-prompt`
- [ADR-124](ADR-124-live-chart-lookback-date-range.md) — Custom… Date range (From/To CT dates)
