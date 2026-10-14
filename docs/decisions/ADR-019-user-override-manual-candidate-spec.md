# ADR-019: User-Override Mode — Manual Candidate Spec Preserves the Declarative Objective

## Status
Accepted — extends the Phase-23 search controller; upholds the "objective never encodes a
model choice" invariant from [ADR-018](ADR-018-objective-doc-single-source-labels.md) and the
objective schema, and complements the Stage-0 feature-view mechanism of
[ADR-017](ADR-017-requirements-first-stage0-feature-views.md).
Amended by [ADR-040](ADR-040-custom-test-params-hpo.md): an explicit candidate
`search_space` is honoured (Custom Test opt-in HPO). Default remains
`search_space: {}` so `--tune-trials` does not override pinned params.
Amended by [ADR-043](ADR-043-cudnn-lstm-gpu-train-path.md): Custom Test
`max_rows` is opt-in (default off = full train). Automated search still uses
catalog `search_max_rows`. Trees / explicit `--manual-spec` files stay
full-matrix by default.

## Date
2026-07-03 (Phase 23)

---

## Context

Phase 23's search controller (`src/search/controller.run_search`) is built around **automated
model selection**: it enumerates the `config`-declared catalog (`src/models/catalog.py`),
optionally narrowed by the Model Scout (W5) or a `--families` whitelist, trains every family
at full walk-forward fidelity, and lets the champion gate pick a winner. The trade objective
(`config/objectives/<id>.yaml`) is deliberately declarative — its schema states it
"**NEVER encodes a model choice**"; it owns only labels, KPI, targets, and eligibility.

Users need a complementary path: *skip* the automated selection and fit **exactly** the
model(s) they choose, each with its **own hyperparameters and its own feature subset**, while
still scoring against the objective's labels/KPI/targets. Three gaps blocked this:

1. There was no way to pin an explicit *set* of models + params (only a family whitelist that
   still ran the full scout/gate machinery on catalog defaults).
2. `CatalogEntry.name` served double duty as both the plugin key **and** the candidate id, so
   two candidates of the same family (e.g. two XGBoost configs) could not coexist.
3. Feature selection only supported **named views** (`feature_view`); there was no user-facing
   way to pick arbitrary feature prefixes/columns. (The `objective["search"]["feature_views"]`
   override read by the controller is still rejected by the objective schema's
   `additionalProperties: false`, so it was never a usable user surface.)

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Embed a `models:` block in the objective doc | One file | **Violates** the objective invariant (ADR-018); pollutes a declarative contract with imperative model choices |
| CLI flags only (`--family --params …`) | No new file | Cannot cleanly express multiple models, per-model params, and per-model feature sets; not reproducible/versionable |
| Reuse the `--families` whitelist | No new code | Still runs Model Scout + catalog defaults; cannot set params or features per candidate |
| **Separate `manual_candidates` spec file + `--manual-spec`** (chosen) | Keeps the objective pure; versionable, schema-validated; expresses N models × params × features; reuses the whole fit/score/gate pipeline | One more schema + loader to maintain |

---

## Decision

Add an opt-in **user-override mode** driven by a **separate** manual-candidates spec
(`config/manual_candidates.schema.json`), loaded by `src/search/manual_spec.load_manual_spec`
and passed via `run_search(..., manual_spec=...)` / `--manual-spec`. The objective doc stays
untouched — it still owns labels/KPI/targets; the manual spec only replaces the
*which-model(s)* decision. Specifically:

1. **Bypass, don't rewrite.** Each user candidate is materialised into a synthetic
   `CatalogEntry` (via `dataclasses.replace` on the family's catalog defaults) and fed through
   the existing CPU/GPU pipeline. `manual_spec` is **mutually exclusive** with `--families`.
2. **Decouple plugin family from candidate id.** `CatalogEntry` gains `plugin_family` and a
   `family_key` property (`plugin_family or name`). `name` becomes a unique candidate id (used
   for leaderboard rows + OOF/diagnostics filenames); `family_key` is the registered plugin to
   build. Backward compatible: catalog entries leave `plugin_family=None` ⇒ `family_key == name`.
3. **Explicit feature selection.** `CatalogEntry.feature_selection` and
   `feature_views.apply_feature_selection` support exactly one of `{"view"|"prefixes"|"columns"}`,
   masking the feature axis for both 2-D tabular and 3-D sequence folds. `feature_selection`
   supersedes the named `feature_view` when present.
4. **Manual-mode semantics.** Candidates default to the **full matrix** (`search_max_rows=None`,
   no search sub-sample) and **no auto-tuning** (`search_space` cleared) — the user deliberately
   chose these params, so `--tune-trials` must not override them. The Stage-2 ensemble layer runs
   only if **both** the caller's `ensembles` flag and the spec's own `ensembles` flag are true.
5. **GPU parity.** The Object-Storage job payload carries `candidate_id`, merged `spec_params`,
   `feature_selection`, and `search_max_rows`; the GPU runner (`src/gpu_runner/runner.py`)
   reconstructs the exact candidate via `dataclasses.replace`, so overrides apply identically on
   the A10.

---

## Consequences

- **+** Users can fit an explicit set of models with per-candidate params **and** features,
  scored on the same frozen contract as the automated search — no objective edits required.
- **+** The declarative objective invariant (ADR-018) is preserved; model choices live in a
  clearly-separate, versionable, schema-validated artifact.
- **+** `plugin_family`/`family_key` finally allow multiple candidates of one family; the change
  is inert for every existing catalog entry.
- **+** Explicit `prefixes`/`columns`/`view` feature selection is delivered through a
  schema-supported surface, sidestepping the still-unwired objective `search.feature_views` gap.
- **−** A second schema + loader to keep in sync with the catalog (family names must exist in
  `catalog.py`; validated at load time).
- **−** The GPU-side reconstruction relies on the family being catalogued (`get_entry`); a manual
  candidate for an un-catalogued family is rejected up front (by design).

### Known follow-ups (not addressed here)
- The objective schema's `search` block still forbids a `feature_views` override
  (`additionalProperties: false`); manual mode makes that path unnecessary for users but the
  controller code that reads it remains dormant.
- **Autonomous iterate-until-target** (`src/search/iterate.py`, `config/iterate_spec.schema.json`,
  W-iterate): builds on this ADR's manual-spec + feature-selection foundations to search
  hyperparameters and feature choices (Optuna TPE) until `kpi.target` is met or a trial/time/cost
  budget is exhausted; added in the same Phase-23 session.
