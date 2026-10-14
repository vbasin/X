# ADR-048: Binary side head on directional labels (rejected)

## Status
Accepted as a **negative result** — implemented and measured 2026-08-21.
Does **not** promote. Live pin remains `nq-50pt-econ`.

## Date
2026-08-21

## Context

The causal 3-class tree is a **long-only timer** (ADR-047): open + τ=0.65 +
10-minute cooldown → 57 longs / 0–1 shorts, NE 11.7, HR 0.33. Always-long
at the same timestamps is identical. The operator asked to **add** something
that tells **direction** (up vs down) under TBM +50/−6 / 10m, L1 only,
causal parquet.

3-class softmax fails at side because ~87% of bars are flat. The correct
add-on is a **second head**: binary XGB trained only on TBM-directional
rows (`y∈{up,down}`), walk-forward, then combined with the existing
activity/timing filter.

## Decision

1. `src/search/side_head.py` — expanding calendar WF, train on open-session
   directional bars with 10-minute uniqueness, purge 10 minutes at the OOF
   boundary. Features are causal location/flow (CLV, VWAP, VA, OF, ES lag),
   not vol-as-side.
2. Combine: fire only when the 3-class timer would fire (`p_up≥0.65`,
   08:30–10:00 CT) **and** `|P(up|dir)−0.5| ≥` a train-chosen margin;
   side = sign of the binary head; 10-minute cooldown.
3. Do not change `config.yaml`. Do not promote unless the combined book
   beats long-only on NE **and** emits both sides with side accuracy > 0.5
   on directional outcomes.

## Results (`20260821T115346Z`)

| Head | OOF |
|------|-----|
| Binary side AUC (open, directional) | **0.494** (folds 0.48–0.51) |
| Long-only timer (baseline) | n=57 HR 0.333 NE **11.67** MDD −49 L/S **57/0** |
| Timer + side head | n=57 HR 0.175 NE **2.82** MDD −182 L/S **28/29** side-acc **0.40** |
| Side head alone (open, 10m cd) | n=2241 HR 0.110 NE −0.85 L/S 1021/1220 side-acc 0.53 |

The add-on **does emit shorts**. They are the wrong shorts. Combining it
with the timer cuts NE by ~9 points.

Single-feature AUCs on directional bars **at timer-fire times** are fade
(not continuation): `vw_vwap_slope_120` 0.25, `vw_vwap_dist` 0.31. Using
them to **flip** longs to shorts also loses (NE 5.8). **Skipping** longs
above VWAP raises NE to 14.0 but drops n to 24 (below the 30-trade floor)
and is still long-only.

## Consequences

- There is no L1 causal feature (or tree on this set) that calls TBM
  +50/−6 side at the timer’s timestamps. A second head cannot fulfill a
  two-sided 55% HR request on this label and data.
- The live-capable causal policy remains ADR-047: long-only open impulse
  filter, 10-minute cooldown, NE ~11, HR ~33%.
- `side_head` stays as a research module + test; it is not a serving path.

## Supersedes

None. Closes the “add a direction head” request against
[ADR-047](ADR-047-inference-uniqueness-open-policy.md). Complements
ADR-046 (activity, not side, on the 3-class softmax).
