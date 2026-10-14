# ADR-117: Live Chart grouped-tick Volume Profile displays

## Status
Accepted — implemented 2026-09-07. Live pin unchanged.

Amends [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
(Volume Profile histogram / LVN behavior).

## Date
2026-09-07

## Context

The former `Binned histogram` split each profile's observed price range into
8–20 equal-width floating price bands. Its boundaries could fall between NQ's
tradable $0.25 levels and shifted whenever the profile high or low changed.
One profile could render only one buy/sell-split histogram.

## Decision

1. Rename the control to **Histogram**. Its choices are **Tick histogram** and
   **Grouped ticks**; the retired `Bins` input is removed.
2. Grouped ticks aggregates native tick bins into a selected fixed sequence:
   `1`, `2`, `4`, `8`, `16`, `32`, or `64` ticks. Boundaries are anchored to
   the global NQ tick grid. Default is `4` ticks ($1.00).
3. POC and value area remain calculated from raw tick-level volume. LVN uses
   the active display resolution: native ticks for Tick histogram and grouped
   bins for Grouped ticks.
4. Every profile has independent **Left display** and **Right display** modes:
   Total volume, Total volume split, Buy volume, Sell volume, Delta volume, or
   None. A single selected display retains the current full-width inward
   rendering. When both are selected, each starts at the profile midpoint and
   grows outward, allowing two complementary volume views for one VP.
5. Existing stored profiles migrate to an inward Total volume split display,
   determined from their placement/side, preserving their previous appearance.

## Consequences

- Grouped rows always represent tradable NQ price increments.
- Operator-selected group resolution controls visual aggregation without a new
  TBBO query; raw VAP remains the source of truth.
- `hist_bins` is no longer accepted or emitted by the Volume Profile contract.

## Related

- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — Volume
  Profile parameters and layout
- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
