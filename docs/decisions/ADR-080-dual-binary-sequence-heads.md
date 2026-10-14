# ADR-080: Dual-binary named heads on sequence DL families

## Status
Accepted — implemented 2026-08-27. Live pin unchanged.

## Date
2026-08-27

## Context

Custom Test `nq-50pt-econ-lg-st/20260827T224538Z` trained a 15s Attention-LSTM
with `params.head: dual_binary`. OOF already had `p_long` / `p_short` (shared
encoder + BCE). Diagnostics and the GPU heartbeat still opened with 3-class
`"labels": ["down", "flat", "up"]` and `operating_point: argmax`, so the run
looked like a softmax model.

Two further defects:

1. LSTM `fit` skipped `class_balance` whenever the head was dual-binary.
   Unweighted BCE on ~6% positives per head is the same always-negative
   fixed point ADR-044 fixed for 3-class CE. This run never fired at 0.5
   (`p_long` max 0.153, `p_short` max 0.138) even though the heads ranked
   (OOF ROC-AUC long 0.74 / short 0.71).
2. TCN / CNN / Transformer still compiled a 3-class softmax and would
   silently train that graph if selected on a dual-binary board. They are
   capable of the same shared-encoder + two-sigmoid layout as LSTM; the
   gap was wiring, not architecture.

Rejected: a fail-closed raise on TCN / CNN / Transformer. Dual-binary is
a head adapter (ADR-076), not an LSTM-only trick.

## Decision

1. **Named heads on every sequence family.** `head=dual_binary` compiles
   two sigmoid outputs `p_long` and `p_short` on the family's encoder
   (LSTM, TCN, CNN, Transformer), independent BCE
   (`src.models.dual_binary.compile_dual_binary_model`). Not a 3-class
   softmax and not a single `Dense(2)` named `output`. Trees stay two
   estimators (`DualBinaryPair`).
2. **Per-head sample weights.** Dual-binary sequence `fit` uses catalog
   `class_balance` (default `sqrt`) independently on `y_long` and
   `y_short`. Cost-matrix 3×3 loss stays off for this head.
3. **Diagnostics.** When OOF has `p_long` / `p_short`,
   `compute_diagnostics` leads with `head: dual_binary`,
   `labels: ["long", "short"]`, and per-head `roc_auc` /
   `average_precision` / `confusion_at_fire`. The derived 3-class book
   stays under `canonical_labels` (`operating_point: dual_binary`, not
   `argmax`). Heartbeat `last_fold.heads` reports `p_mean` / `p_max`.
4. **Serving.** Sequence and tabular predict accept dict heads, `(N, 2)`,
   or `predict_pair`. Canonical `{direction, confidence}` is argmax of
   `p_long` vs `p_short` **and** `max(p) ≥ 0.1`
   ([ADR-089](ADR-089-dual-binary-argmax-fire.md)).
5. **CNN / Transformer Custom Test path.** 2-D fold arrays are windowed
   and Sequence-fed like LSTM / TCN (rebuild from actual `F`).

## Consequences

- Live pin unchanged. 3-class path unchanged when `head` is omitted /
  `multiclass`.
- `20260827T224538Z` OOF is valid dual-binary; its diagnostics JSON was
  regenerated. Do not reuse that run for a weighted-head comparison —
  re-run Custom Test so the sequence model trains with per-head `sqrt`
  weights.
- Saved `model.keras` from that LSTM run is `Dense(2)` named `output`.
  New dual-binary runs write `p_long` / `p_short`. Serving loads both.

## Related

- [ADR-044](ADR-044-lstm-class-balance-sqrt.md) — 3-class LSTM sample weights
- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — lg-st dual `P_long` / `P_short`
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — Custom Test head slot
- [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md) — 15s hive GPU mirror
- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) — aux path labels + head-private `lg_*`/`st_*`
