# ADR-119: Live Chart VWAP origin parameter

## Status
Accepted — implemented 2026-09-07. Live pin unchanged.

## Context

The Live Chart `vw_vwap` overlay was computed from a tick window with a hard-coded
session grouping, while CVD exposed an operator-selectable reset origin. VWAP
therefore could not be aligned with the chart's selected session clock, and its
reset behavior was not visible in the UI or request payload.

## Decision

1. Add `vwap_origin` to the Live Chart series API and chart-model requests. It
   uses the existing origin choices: `session` (17:00 CT), `bod` (midnight CT),
   and `mkt` (08:30 CT); the default remains `session`.
2. Calculate VWAP as cumulative price×size divided by cumulative size within
   each origin period, using the same origin-period helper as CVD. VWAP distance
   uses that same reset series.
3. Add a nested VWAP control in the overlay parameters, persist it in browser
   storage, send it on normal, replay, and model-design loads, and include the
   active origin in the rendered price-series title.
4. Do not use the bounded raw-tick loader for VWAP. Aggregate `price * size`
   and size by chart bucket in SQL across the complete requested lookback, then
   perform origin-period cumulative arithmetic on the compact result.

## Consequences

- VWAP can be compared directly with CVD under the same reset clock.
- Existing callers remain compatible because `session` is the default.
- Other raw-tick overlays remain capped at the existing 36-hour limit; VWAP is
   bucket-aggregated across the complete requested lookback.
- The change fixes the reset contract. If a selected VWAP still has no values,
  the remaining diagnostic is the underlying TBBO tick query/coverage rather
  than a missing frontend series registration.

## Related

- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — CVD origin clocks
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — nested overlay parameters
- Module: `src/webui/live_chart.py`
