# ADR-044: Attention-LSTM `class_balance=sqrt` sample weights

## Status
Accepted — implemented 2026-08-21.
Follows [ADR-043](ADR-043-cudnn-lstm-gpu-train-path.md) (cuDNN LSTM GPU path).
Uses the same loss-weighting contract as TCN (`class_balance=sqrt`).
**Amended 2026-08-27 (ADR-080):** dual-binary sequence models apply the
same `class_balance` independently on `y_long` and `y_short`. Skipping
weights for that head collapsed P(event) below the 0.5 fire.

## Date
2026-08-21

---

## Context

Custom Test LSTM `20260821T085855Z` trained in 700 s on the A10 (cuDNN + 100k
last-N) but the OOF 3×3 was **all-flat**: 0 up / 0 down predictions. Confidence
was not constant (`P(flat)` 0.60–0.998, 1.0M distinct values). `1−P(flat)` vs
true directional had AUC **0.70** (move/no-move skill) and **0.55** side AUC.
Unweighted sparse CE on ~87% TBM flats is an always-flat fixed point — the
same TCN failure before `class_balance=sqrt`.

The LSTM plugin did not pass `sample_weight` into `model.fit`. Catalog
`params_default` omitted `class_balance`, so Custom Test could not show or
override the knob.

## Decision

1. **Default `class_balance: "sqrt"`** on `attention_lstm` catalog
   `params_default`. Weight for a class with frequency \(f\) is
   \(\sqrt{1/f}\) (gentler than sklearn `balanced`).
2. **`AttentionLSTMPlugin.fit`** calls `_make_sample_weight` (same helper as
   TCN / trees) and feeds the Keras `Sequence` the third tuple element.
3. **Custom Test** lists the knob from catalog defaults +
   `search_space.class_balance` categorical `{None, balanced, sqrt}` so the
   model-card params table is a dropdown. `None` = unweighted (the failed
   085855Z path). Opt-in HPO may vary that one key (ADR-040).

Rejected: leaving the UI-only param without plugin wiring (no-op); defaulting
to `balanced` (over-fires on TCN); using `dir_threshold` before the
directional heads have mass.

## Consequences

- A new LSTM Custom Test uses sqrt weights unless the operator sets
  `(none)` / `balanced`.
- Meta-stack on an all-flat base is still a waste; re-check the 3×3 after
  this change before stacking.
