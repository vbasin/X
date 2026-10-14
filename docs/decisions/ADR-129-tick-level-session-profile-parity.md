# ADR-129: Tick-level session profile parity with the WebUI

## Status
Accepted — 2026-09-15. Applies to future cascade level calculations only; existing
move/target parquet artifacts are not rebuilt by this change.

## Context

The cascade's `tape_levels()` used one-point floor bins and reported bin midpoints,
while the WebUI queried the same lead-contract TBBO tape at the native NQ tick
resolution. On 2026-05-29 this produced different levels: the research path
reported POC 30407.50, VAL 30350.50, and VAH 30441.50, while the WebUI reported
POC 30400.00, VAL 30348.50, and VAH 30439.50. The high and low matched because
they are raw extremes.

## Decision

Use the WebUI calculation as the canonical session-profile strategy for the model:

- cash session: 08:30–15:00 America/Chicago;
- exact lead-contract TBBO prints;
- round prices to the native NQ tick, 0.25 points;
- aggregate print size at each tick;
- POC is the highest-volume tick;
- expand from POC until 70% of volume is included;
- when adjacent volumes tie, include both sides, matching the WebUI.

The shared `src.features.tape.tick_value_area()` helper is used by both the model
`tape_levels()` path and the WebUI `value_area()` path. The generic one-point
`ProfileConfig.bin_size` remains available for legacy profile consumers and does
not control cascade session levels.

## Consequences

- Future move-matrix, target-matrix, stop-probe, and related level calculations
  will use actual traded tick prices rather than one-point bin midpoints.
- Existing artifacts remain untouched and are not comparable to future artifacts
  where session levels affect labels or target geometry. They must be rebuilt
  before new model evaluation.
- The 2026-05-29 parity check now returns `lo=30270.00`, `VAL=30348.50`,
  `POC=30400.00`, `VAH=30439.50`, and `hi=30536.00` in both paths.

## Validation

`tests/test_tape_cascade.py` includes the native-tick value-area behavior. The
focused suite passes 41 tests, and the 2026-05-29 tape calculation matches the
WebUI backend exactly.
