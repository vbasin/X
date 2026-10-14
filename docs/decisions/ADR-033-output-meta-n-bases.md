# ADR-033: N-base output-meta (first instance: XGBoost + TCN)

## Status
Accepted — implemented 2026-08-15.

## Date
2026-08-15

## Context

`nq-50pt-econ` uses `primary_meta`: one model, then a binary gate on that
model's OOF side/confidence. Operators wanted a **parallel** tree + sequence
stack on the **same** TBM +50/−6 labels and feature view, with a gate that
sees **upstream prediction vectors** (optional extra data). The combo must
not hardcode family names.

Existing Stage-2 stacking blends sides (3-class) with shuffled KFold. That is
another side picker, not a trade/no-trade gate.

## Decision

1. Topology mode `output_meta` with `bases: [family, …]`. Any catalog
   families. First objective: `nq-50pt-xgb-tcn`.
2. Same objective `labels` (TBM 50/6/10m) for every base. KPI stays
   `net_expectancy` on the composed tape.
3. Meta features = per-base probabilities/confidence/side indicators, plus
   optional `context_view`. Proposed side is resolved from base outputs;
   meta only approves or rejects. Default gate learner:
   `logistic_regression` (not a second XGBoost “family” identity).
4. Align base OOF on `ts`. Time-block meta CV with `embargo_bars` ≥ horizon.
5. Sequence input width = assigned column count after the view — never
   hardcoded `n_features: 154`.
6. Training is GPU-first when `GPU_INSTANCE_OCID` is set.
7. Finalize/serve loop `bases` then meta.

`nq-50pt-econ` is unchanged.

## Consequences

- Heterogeneous stacks are YAML, not a new code path per combo.
- Meta trains on walk-forward OOF vectors (not in-sample base homework).
- Same 50/6 labels leave FLAT imbalance on every base; the gate only
  filters proposed trades.
- Serious TCN fits need the GPU VM.

## Related

- [ADR-027](ADR-027-economics-meta-sparse-objective.md)
- Objective: `config/objectives/nq-50pt-xgb-tcn.yaml`
- Composer: `src/search/output_meta.py`
