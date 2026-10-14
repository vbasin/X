# ADR-121: Live Chart minimum grouped bars for LVN

## Status
Accepted — implemented 2026-09-07. Live pin unchanged.

## Context

Grouped-tick Volume Profiles can produce isolated one-bar LVN fields from normal
profile noise. Operators need a way to require an LVN to persist across multiple
adjacent grouped histogram bars before it is displayed.

## Decision

1. Add per-profile `lvn_min_groups`, default `1`, with an allowed range of
   `1..20`.
2. Apply the threshold to grouped-tick LVN nodes only. A grouped LVN run is
   retained only when it contains at least `lvn_min_groups` consecutive grouped
   bars; shorter runs are treated as ordinary profile body.
3. Leave raw tick-histogram LVN behavior unchanged, because raw ticks are not
   grouped bars and should not inherit a grouped-bar threshold.
4. Expose the setting as **LVN min grouped bars** in each Volume Profile card.

## Consequences

- `1` preserves current behavior.
- `2` removes isolated one-group LVNs while retaining two-group-or-longer LVN
  fields.
- The setting works with the existing `group_ticks` choice; it counts rendered
  grouped histogram bars, not raw NQ ticks or dollars.
- POC, Value Area, KDE HVN zones, and raw histogram volumes are unaffected.

## Related

- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — Volume Profile parameters
- [ADR-117](ADR-117-live-chart-grouped-tick-volume-profile.md) — grouped-tick histogram
- [ADR-120](ADR-120-live-chart-adaptive-kde-hvn-zones.md) — KDE HVN zones
- Module: `src/webui/live_chart.py`
