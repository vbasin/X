# ADR-088: Aux path-side ranking regularizer on dual-binary decisions

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

Dual-binary `p_long` / `p_short` are independent sigmoids (ADR-080). Aux
Huber heads (ADR-082 / ADR-084 / ADR-086) reconstruct path statistics on
**every** bar, but those outputs are dropped at serve and do not enter
the fire rule. Custom Test `nq-50pt-econ-lg-st/20260828T101050Z` showed
the trunks collapsing (`corr(p_long, p_short) = 0.93`; predicted
`MFE_long` vs `MFE_short` **+0.73** while true labels were **−0.21**).
Per-head ROC was event-vs-quiet; `p_long − p_short` among true
directional bars was ~0.50.

The dense aux labels already have a side on almost every bar:
`MFE_long_10m` vs `MFE_short_10m` (peak favorable excursion each way).
That is a different question from sparse TBM 50/6, and is the signal
that should **order** the two decision heads without replacing them.

Rejected: 3-class softmax (changes the lg-st contract); using Huber
predictions at serve; train uniqueness / dropping quiet rows; ranking
on z-scored MFE (fold-scale dependent).

## Decision

1. When `labels.aux` is on, `compile_dual_binary_model` adds a
   `side_rank` output: `DualSideFraction` =
   `p_long / (p_long + p_short)` (clipped). Loss is binary CE.
   Weight **0.3**. Primary BCE on `p_long` / `p_short` stays 1.0.
2. Target `y = 1` iff `MFE_long_10m > MFE_short_10m`. Sample weight
   `clip(|Δ| / 50, 0, 1)` so a 50-pt path winner counts fully and a
   1-pt chop barely moves the gradient. Ties / NaN have weight 0.
   All train rows remain in the TBM BCE (no uniqueness subsample).
3. Serving / OOF / KPI still read only `p_long` / `p_short`.
   Canonical book is argmax **and** `max(p) ≥ 0.1`
   ([ADR-089](ADR-089-dual-binary-argmax-fire.md)).
   `stack_head_predict` ignores `side_rank`.
4. Trees are unchanged (no aux graph). TCN / CNN / Transformer grow
   `side_rank` only if they compile with `aux=True` (LSTM is the only
   caller today).

## Consequences

- Live pin unchanged. Independent BCE remains the event detector;
  `side_rank` only regularizes **relative** long vs short mass.
- Saved `20260828T101050Z` graphs still load (no `side_rank` output).
  New LSTM Custom Test graphs have the extra output. Re-run lg-st
  LSTM; do not reuse `20260828T101050Z`.
- `DualSideFraction` is a registered Keras layer (`vtech>DualSideFraction`).

## Related

- [ADR-080](ADR-080-dual-binary-sequence-heads.md) — independent sigmoids
- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) — aux columns
- [ADR-084](ADR-084-objective-opt-in-aux-heads.md) — YAML opt-in
- [ADR-086](ADR-086-side-specific-aux-targets.md) — short `−z(return)`
