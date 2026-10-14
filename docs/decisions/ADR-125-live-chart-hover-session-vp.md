# ADR-125: Live Chart hover VP is the covering session only

## Status
Accepted — implemented 2026-09-11. Live pin unchanged.

Amends [ADR-087](ADR-087-live-chart.md) decision 6 (hover / readout VP
rows) and [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
decision 3 (recurring expansions in the readout). Histogram drawing and
Extend lines are unchanged.

## Date
2026-09-11

## Context

Recurring Volume Profile expands to one histogram per session (or week)
in the lookback. The hover box and the top readout listed
`profiles.slice(0, 4)` — the first four payloads, which are the
**oldest** sessions. Hovering Thursday still showed Monday–Wednesday
POC / volume-at-price / LVN.

## Decision

1. **Covering window.** Hover and readout VP rows are the profiles
   whose half-open tape window `[start_unix, end_unix)` contains the
   crosshair bar. Several specs can match the same bar (e.g. VP1 cash
   and VP2 full session). Recurring expansions of one spec do not.
   The hover box lists the five levels **VHi** (`high`), **VLo**
   (`low`), **VAH**, **VAL**, **POC** (then volume-at-price / LVN).
   The top readout strip keeps a compact **POC** chip so the OHLC
   line does not wrap.
2. **Not Extend.** Forward Extend (1 / 2 / All sess) still draws lines
   into later sessions; those later bars do **not** inherit the earlier
   session’s levels in the popup.
3. **Fallback.** If `start_unix` / `end_unix` are missing, match
   Globex session-end date (`vpSessionDay`).
4. **Cap.** At most eight covering profiles (the spec cap).

## Consequences

- Cash / premarket / custom clocks only appear while the cursor is
  inside that clock window, not for the rest of the Globex day.
- A Single VP pinned to another date is silent when hovering a
  different session.

## Related

- [ADR-087](ADR-087-live-chart.md) — hover box
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — VP
- UI: `#lc-hover` / `#lc-readout` (`profileCoversUnix`)
