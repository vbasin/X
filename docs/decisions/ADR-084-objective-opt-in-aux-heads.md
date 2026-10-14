# ADR-084: Objective-opt-in dual-binary aux heads

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

ADR-082 wrote ten path-quality columns onto the 5s and 15s label parquets and
grew Huber aux heads on any dual-binary LSTM that saw those columns. The
operator's intent was that **this exact subset** train only for **this
exact objective** (`nq-50pt-econ-lg-st`), not for every LSTM going forward.

3-class LSTMs never consumed `y_aux` (the plugin requires `head=dual_binary`).
The leak was: any later dual-binary LSTM on those files — including a Custom
Test head flip on another 50/6 TBM objective — would grow the same aux graph
without a YAML declaration.

Rejected: auto-on from parquet presence; a global LSTM flag.

## Decision

1. **`labels.aux`** on the objective YAML is the opt-in. Omitted / empty =
   no aux heads on any family. The list is the parquet column subset (schema
   enum of the ADR-082 names). `nq-50pt-econ-lg-st` lists the train-head
   subset ([ADR-086](ADR-086-side-specific-aux-targets.md) dropped
   `MAE_before_target_or_MFE` from the YAML list; the column stays on
   the parquet).

2. Fold cache attaches `y_aux` only when `labels.aux` is set. The dual-binary
   LSTM plugin further requires that list **and** `y_aux_train` before
   compiling Huber outputs. Trees / TCN / CNN / Transformer stay
   `p_long` / `p_short` only.

3. Label parquets may still carry the columns as a sidecar (rebuilds
   unchanged). Extra columns are skipped as features. They are not KPI,
   OOF ranking, fire, or serving targets.
   **Amended [ADR-088](ADR-088-aux-side-rank-decision.md):** `labels.aux`
   also turns on `side_rank` (relative `p_long` vs `p_short` from MFE
   winner). Still not a serve/KPI head.

4. Cache key for `encoding=dual_binary` includes the requested aux names so
   toggling `labels.aux` invalidates folds.

## Consequences

- Live pin unchanged. 3-class path unchanged.
- A dual-binary LSTM Custom Test on `nq-50pt-econ` / `nq-50pt-tbm` / etc.
  does not grow aux heads.
- Adding aux to another objective is a YAML list plus schema-valid names,
  not a code change.
- **Amended [ADR-095](ADR-095-regime-context-fire-aux.md) /
  [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md):** lg-st aux list is
  MFE / MAE_b4 / 10m return / `Time_to_TGT_*` (NaN on no-hit; dropped
  1m/5m returns). LSTM also trains binary `long_stop_first` /
  `short_stop_first` from MAE_b4 ≥ 6. Dual-binary trees use MFE
  near-miss sample weights; they still do not grow Keras heads.

## Related

- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) — columns + Huber
  weights; auto-on from parquet **amended** by this ADR
- [ADR-080](ADR-080-dual-binary-sequence-heads.md) — named `p_long` / `p_short`
- [ADR-086](ADR-086-side-specific-aux-targets.md) — side-signed forward-return aux; YAML list dropped shared MAE
