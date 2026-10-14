# ADR-118: Live Chart Total Volume delta split display

## Status
Accepted — implemented 2026-09-07. Live pin unchanged.

Amends [ADR-117](ADR-117-live-chart-grouped-tick-volume-profile.md).

## Date
2026-09-07

## Context

Volume Profile side displays could show total volume, total volume split by
aggressor, or delta volume. Operators also need the total-volume footprint and
its signed buy/sell imbalance in one aligned display.

## Decision

Add **Total Volume delta split** to each profile's Left display and Right
display menus. The total-volume bar determines width. A second segment overlays
it from the display's outer edge, with width equal to `abs(buy - sell) / total`
of that bar. The overlay is buy-colored when buy volume is at least sell volume,
and sell-colored otherwise. The side's existing opacity applies to both layers.

## Consequences

- Total participation and directional imbalance can be inspected without
  allocating both display sides to the same profile.
- Existing display modes and stored profile settings are unchanged.

## Related

- [ADR-117](ADR-117-live-chart-grouped-tick-volume-profile.md) — grouped ticks
  and bilateral displays
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — Volume
  Profile parameters and layout
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
