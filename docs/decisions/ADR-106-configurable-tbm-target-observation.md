# ADR-106: Configurable TBM target observation with close-based stops

## Status
Accepted — implemented 2026-09-02. Live pin unchanged.

## Date
2026-09-02

## Context

Legacy `triple_barrier` labels observe both the profit target and the adverse
stop from 5-second closes. That can miss a profit target printed by a TBBO trade
inside a 5-second bucket. The `nq-50pt-econ-lg-st` contract retains its
intentional 5-second close observation of the 6-point MAE stop, but needs an
objective-controlled profit-target observation source.

## Decision

1. Add `labels.target_observation` to the objective schema:
   `candle_close` (default), `candle_hilo`, or `tbbo`.
2. Stops always remain 5-second-close observations. Target checks use the
   selected source. With `candle_hilo` or `tbbo`, a target in a bucket wins over
   the same bucket's close-based stop because the close is observed last.
3. Canonical 5-second `cs_high` / `cs_low` are lossless extrema of TBBO trade
   prints, so `tbbo` consumes those columns without a second raw-tick database
   scan. It is intentionally equivalent to `candle_hilo` at this label grid.
4. The setting is recorded in persisted label contracts and fold-cache keys.
   Native-TF fold construction rejects a mismatched 15-second sidecar rather
   than silently training against stale close-only labels.
5. `nq-50pt-econ-lg-st` selects `target_observation: tbbo`. Rebuild its 15s
   labels and all dependent fold caches before a new search. Existing runs,
   including `20260829T182514Z`, remain close-only and are not comparable.

## Consequences

- A target hit at any TBBO trade price inside a 5-second bucket is retained;
  the target-time auxiliary label uses the same observation source.
- The target/stop contract remains asymmetric by design and is not an intrabar
  OCO simulation. Path-realised 1-second evaluation remains the execution
  truth source.
- Changing `target_observation` is a label-semantic change and requires label
  and model regeneration.

## Related

- [ADR-023](ADR-023-triple-barrier-label-scheme.md) — triple-barrier labels
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — persisted label stores
- [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md) — target-time auxiliary
- `src/training/labels.py`, `src/training/aux_labels.py`