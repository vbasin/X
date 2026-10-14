# ADR-061: Pine SAR+EMA optional exit at the tagged SAR trail

## Status
Accepted — implemented 2026-08-26.

## Date
2026-08-26

## Context

The Stage-0 Pine replica ([ADR-058](ADR-058-external-models-pine-sar-ema.md))
closes a position on the bar where price intersects the in-effect Parabolic
SAR trail by filling at that bar's **close** (and, with the EMA slope gate
on, may wait even later until L/S slope confirms). The operator asked for a
knob on the External Models **Parameters** view: exit at the **SAR value of
that bar** instead.

Wilder's trail that was tagged is the stop in force at the **start** of the
bar (`nextBarSAR` of the previous closed bar). After the same-bar flip,
`sar[i]` is the **new** opposite-side trail (near the extreme). Filling a
stopped long at the post-flip SAR would be a near-high fill, not a stop.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Change the replica to always stop at SAR | Matches a SAR-as-stop reading | Breaks Pine close fills; Stage 0 is a replica |
| B. Opt-in `exit_at_sar` (chosen) | Factory still Pine close; operator can score the trail stop | Extra knob; same-bar PT vs SAR uses the existing stop-wins rule |
| C. 1s path-realised SAR touch (ADR-052) | Honest first-touch | Out of Stage 0; optional later |

## Decision

1. New bool `exit_at_sar` on the SAR engine (factory **false**). Shown on
   `#/external` Parameters for card `psar` (ADR-062 moved it off the
   combined `psar_ema20` form). The form is still the catalog `param_spec`.

2. When **on**, a position aligned with the previous bar's SAR trend treats
   that bar's incoming trail as a stop: high/low first-touch, gap-through
   at the open, reason `sar_stop`. This fires on the intersection bar even
   if the EMA slope gate would have held until close. Same-bar OCO
   profit-target / stop-loss still apply; when two stops tag, the **tighter**
   (first-touch) level fills.

3. When **off**, behaviour is unchanged from ADR-058 (close of the
   intersection bar, or later under `ema_gate`).

4. The fill is the tagged trail, not the post-flip SAR dot. Signal-lock
   treats `sar_stop` like the other stop-loss exits.

## Consequences

- Operators can compare Pine close-of-bar vs SAR-as-stop on the same
  candles without changing the factory replica.
- Live pin, `ps_*` features, and ML catalog are unchanged.

## Related

- [ADR-058](ADR-058-external-models-pine-sar-ema.md) — External Models plane
- [ADR-055](ADR-055-parabolic-sar-closed-bars.md) — SAR state machine
- Package: `src/external_models/pine_sar_ema.py`, catalog `exit_at_sar` on `psar`
- [ADR-062](ADR-062-ext-model-split-and-stack.md) — SAR is its own engine; `exit_at_sar` stays on that card
