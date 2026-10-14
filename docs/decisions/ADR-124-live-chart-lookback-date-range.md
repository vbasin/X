# ADR-124: Live Chart custom lookback date range

## Status
Accepted — implemented 2026-09-11. Live pin unchanged.

**Superseded in part by [ADR-128](ADR-128-live-chart-drop-tf-lookback-caps.md):**
decision 4’s per-TF lookback caps are removed; date-range parsing and
the 5-year ceiling remain. Overlay chrome is unchanged.

Amends [ADR-105](ADR-105-live-chart-custom-lookback.md) decision 3
(custom duration prompt only). Overlay chrome and per-TF caps are
unchanged.

## Date
2026-09-11

## Context

Custom lookback was a duration lower bound: `7d` means *now − 7 days
through now*. Operators also need a closed historical window (From and
To dates) that does not slide with wall time — e.g. one event week,
or a month that is not “the last 30 days”.

The series API already had an `end=` as-of clip for tests and replay,
but the toolbar never sent it, so a range could not be chosen in the
UI without losing the duration path.

## Decision

1. **Two custom modes.** Toolbar **Custom…** offers **Duration** (ADR-105
   lower bound, still the default) and **Date range** (From / To).
2. **Range token.** `GET /api/chart/series?lookback=` and
   `/api/chart/prints` accept `YYYY-MM-DD..YYYY-MM-DD` (alias
   `YYYY-MM-DD to YYYY-MM-DD`). Canonical token uses `..`. Dates are
   **America/Chicago calendar dates, inclusive**: From 00:00 CT through
   24:00 CT on To (exclusive next midnight), matching VP date-only
   bounds. Same-day is valid. To before From is 400.
3. **As-of clip.** If To is today relative to the request as-of (`end=`
   or now), the right edge is that as-of instant, not midnight tonight,
   so a range ending today still shows the live tape. A range entirely
   after as-of is an empty window.
4. **Caps.** Per-TF lookback caps and the 5-year ceiling still apply
   after parse; `lookback_capped` is true when the From edge was
   pulled forward. Tape overlays stay on the 36h print cap.
5. **Live poll.** Applying a range whose To is before today unchecks
   **Live** (historical window does not grow). Duration custom and
   ranges that include today leave the Live checkbox as the operator
   set it.
6. **Persistence.** The range token is stored in `lc_lb` and shown as
   `Custom: 2026-08-01 → 2026-08-15`. Closed-enum duration clients are
   unchanged.

## Consequences

- A range is a snapshot of those CT dates, not a sliding lookback.
  Re-open Custom… on a range token to edit From/To.
- 5s of a multi-day range still clips to 4h from the right edge.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-105](ADR-105-live-chart-custom-lookback.md) — duration custom
- Module: `src/webui/live_chart.py` (`parse_lookback_range`, `window_for`)
- UI: `#/chart` Lookback **Custom…** + `#lc-lb-prompt` Duration / Date range
