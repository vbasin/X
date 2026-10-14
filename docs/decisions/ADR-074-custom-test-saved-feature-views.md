# ADR-074: Custom Test saved feature views + expand-on-assign

## Status
Accepted — implemented 2026-08-27.
Amends [ADR-035](ADR-035-webui-custom-test.md): a Custom Test model card
shows the assigned **columns**, not a single view/group chip; operators can
save that subset as a named view in the feature bank.

## Date
2026-08-27

---

## Context

Custom Test let operators drop a catalog view or prefix group onto a model
as one chip. The card then hid the actual columns, and there was no way to
reuse an ad-hoc subset (for example “these 16 LSTM columns”) on another
model or later run without dragging each feature again.

## Decision

1. **Expand on assign.** Dropping or clicking a view or group onto a model
   card materializes every matching catalog column as its own chip. Default
   boards (objective view names) expand the same way when the page hydrates.
   Ablation flags stay per-column (a view is no longer one ablation unit).
2. **Save as view.** Top-right of each model card: **save as view**. The
   operator names the current column list. Persistence is
   `ops/webui/custom_feature_views/<name>.json` (global to the console, not
   per-objective). Catalog `FEATURE_VIEWS` names are reserved.
3. **Feature bank.** Saved views appear first on the Views tab (source
   `saved`) and expand to columns when assigned, same as catalog views.
   Delete is only offered on saved views.
4. **Compile.** A saved view used as `kind: view` compiles to
   `{"columns": [...]}`. Catalog views used alone still compile to
   `{"view": name}`. After the UI expands, the launch payload is usually
   already `kind: feature` items.

Rejected: writing operator views into `FEATURE_VIEWS` / objective YAML
(ADR-018); keeping a collapsed view chip with a tooltip of names.

## Consequences

- Assigning `regime_core` snapshots the current catalog columns for that
  prefix set. Later columns added under those prefixes are **not** picked
  up unless the operator drops the catalog view again.
- Ablating a whole view in one click is gone; flag individual columns
  (still capped at 6 flags / slot).
- Saved views are operator state under `$VTECH_DATA_DIR`, not git.

## Related

- [ADR-017](ADR-017-requirements-first-stage0-feature-views.md)
- [ADR-035](ADR-035-webui-custom-test.md)
- Compiler: `src/search/custom_test.py`
- Console: `src/webui/static/app.js`, `POST/DELETE /api/custom-test/views`
