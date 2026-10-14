# ADR-035: WebUI Custom Test compiles to a manual spec + topology overlay

## Status
Accepted — implemented 2026-08-15.
Amended by [ADR-040](ADR-040-custom-test-params-hpo.md): per-slot catalog
params + opt-in HPO on the Custom Test board.
Amended by [ADR-042](ADR-042-custom-test-calendar-walk-forward.md): expanding
calendar walk-forward overlay (`--walk-forward-overlay`) on the Custom Test board.
Amended by [ADR-071](ADR-071-optional-champion-gate.md): Custom Test does **not**
apply the champion gate; `run.json` is `status: report` with no `gate_result`.
Amended by [ADR-074](ADR-074-custom-test-saved-feature-views.md): model cards
expand views/groups to columns; **save as view** persists a named subset in
the feature bank (`ops/webui/custom_feature_views/`).

## Date
2026-08-15

## Context

Operators needed a console path to pick catalog families, assign feature views /
prefix groups / columns, mark units for ablation, and optionally sit a meta
gate on the selected models — without editing the immutable Trade Objective
(ADR-018) and without going through automated model-selection search.

Two existing pieces already do the scoring:

- ADR-019 manual candidates spec (`--manual-spec`)
- Objective `search.topology` (`meta_stack` / `output_meta` / `primary_meta`)

The latest stack (`nq-50pt-xgb-tcn`) is the default testing configuration:
XGBoost + TCN on their assigned views, plus a `meta_stack` logistic gate
(`walk_forward_OOF`, `linear_weights`, τ=0.55, 5 folds, 120-bar embargo).

A **feature builder** (type existing feature names to define a new column, then
compose that into a later feature) was requested as a future concept only.

## Decision

1. **New operator entry path: Custom Test.** From an objective, **Custom test**
   opens `#/objectives/<id>/test`. The board compiles to a manual-candidates
   spec plus an ephemeral topology overlay and launches
   `src.search.controller` (same walk-forward and scoring as search;
   **no champion gate** — [ADR-071](ADR-071-optional-champion-gate.md)).
2. **Do not mutate the objective YAML.** Topology for the run is applied
   in-memory via `run_search(..., topology_override=...)` /
   `--topology-overlay`. `mode: none` turns off the objective's default meta
   so an xgboost-only custom test on `nq-50pt-xgb-tcn` does not still expect
   a TCN base.
3. **Ablation is per-slot and multiplicative.** Each flagged item is a unit
   (whole view, whole prefix group, or one column). *k* flags on a slot yield
   `2^k` candidates. Caps: 6 flags/slot, 64 candidates/run.
4. **Meta uses the full (no-omission) variant** of each selected family.
   Ablation variants are scored independently. Default meta fields come from
   the objective's `search.topology` when present, else the nq-50pt-xgb-tcn
   stack defaults.
5. **Feature builder is parked.** The UI shows a disabled card; new columns
   are added to the catalog on request, not composed in the console.

## Consequences

- Custom Test is a compiler in front of ADR-019 + existing topology layers,
  not a second training engine.
- Board + compiled spec + overlay persist under
  `ops/webui/custom_tests/<objective>/<run_id>.*`.
- Each successful tree candidate also writes a promote-ready serving folder
  (last walk-forward fold). **Superseded for location by
  [ADR-036](ADR-036-custom-test-lab-artifacts.md):**
  `research/custom_test/<objective>/<run_ts>/models/<candidate_id>/`
  (weights + `oof.parquet`). `meta_stack` is persisted as its own bundle.
- Feature-builder compose-from-names remains backlog.

## Related

- [ADR-018](ADR-018-objective-doc-single-source-labels.md)
- [ADR-019](ADR-019-user-override-manual-candidate-spec.md)
- [ADR-027](ADR-027-economics-meta-sparse-objective.md)
- [ADR-033](ADR-033-output-meta-n-bases.md)
- Compiler: `src/search/custom_test.py`
- Schema: `config/schemas/custom_test.schema.json`
- [ADR-040](ADR-040-custom-test-params-hpo.md)
- [ADR-042](ADR-042-custom-test-calendar-walk-forward.md)
- [ADR-074](ADR-074-custom-test-saved-feature-views.md)
