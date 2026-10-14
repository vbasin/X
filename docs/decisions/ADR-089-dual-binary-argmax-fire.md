# ADR-089: Dual-binary fire is argmax + event τ=0.1 (not 0.5)

## Status
Accepted — implemented 2026-08-28. **Amended same day:** event cut
`max(p_long, p_short) ≥ 0.1` (operator). **Amended 2026-08-28
([ADR-095](ADR-095-regime-context-fire-aux.md)):** `nq-50pt-econ-lg-st`
sets `labels.fire: 0.40`. Library default remains 0.1. Live pin unchanged.

## Date
2026-08-28

## Context

ADR-076 derived canonical `{direction, confidence}` as
`up` if `p_long ≥ p_short` **and** `p_long ≥ 0.5` (short symmetric).
Custom Test `nq-50pt-econ-lg-st/20260828T101050Z` 15s LSTM never emitted
an up: `p_long` max 0.474, and on true-up bars `p_short` was usually
larger. Independent BCE + a 0.5 cut is an always-flat (or short-only)
book whenever both sigmoids sit below 0.5, even if they rank.

Each head is already `P(TBM 50/6 on that side)`. Argmax is **direction**.
The **level** of the winner is the 50/6 event. τ=0.5 blocked longs that
ranked. τ=0 (first draft of this ADR) treated 0.03 vs 0.02 as a long —
direction without an event.

The 0.5 gate is not in the YAML (`nq-50pt-econ-lg-st` has no
`precision_at_tau`). It was hardcoded in
`dual_binary_to_canonical` and re-applied at serve via `dir_thr`
(catalog default 0.5).

3-class `dir_threshold` is a different knob and stays.

## Decision

1. Dual-binary canonical book is **event ∧ direction**:
   long if `p_long > p_short` **and** `p_long ≥ 0.1`;
   short if `p_short > p_long` **and** `p_short ≥ 0.1`;
   else flat. That is `max(p) ≥ 0.1` and argmax. Default
   `DUAL_BINARY_FIRE = 0.1`. Pass `fire=0.5` for the old cut;
   `fire=0` for argmax-only.
2. Serving `_frame_from_dual` no longer passes bundle `dir_thr` as
   `fire`. 3-class `direction_from_proba(..., dir_thr)` is unchanged.
3. Per-head `confusion_at_fire` in diagnostics remains **0.5** as a
   calibration readout (`n_pred_pos` = bars with `p ≥ 0.5` on that
   sigmoid). It is not the trade book.

## Consequences

- Live pin unchanged (`nq-50pt-econ` is 3-class).
- A 0.47 vs 0.10 bar is a **long** (event 0.47 ≥ 0.1). A 0.03 vs 0.02
  bar stays **flat**. Existing run artifacts scored at 0.5 — re-score
  or re-run.
- `side_rank` (ADR-088) still trains relative mass; this rule is serve/OOF.
- **ADR-095:** τ=0.1 on `20260828T215046Z` fired 70% of 15s bars
  (NE −1.20). That objective now declares `labels.fire: 0.40`. Other
  dual-binary code paths still default to 0.1 unless YAML sets fire.

## Related

- [ADR-076](ADR-076-custom-test-store-head-labels.md) — original 0.5 cut
- [ADR-080](ADR-080-dual-binary-sequence-heads.md)
- [ADR-088](ADR-088-aux-side-rank-decision.md) — relative `p_long` vs `p_short`
- [ADR-095](ADR-095-regime-context-fire-aux.md) — lg-st fire 0.40
