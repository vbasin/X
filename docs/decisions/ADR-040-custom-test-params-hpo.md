# ADR-040: Custom Test per-family parameters and opt-in HPO

## Status
Accepted — implemented 2026-08-21.

## Date
2026-08-21

## Context

WebUI Custom Test (ADR-035) compiled a board into an ADR-019 manual spec: catalog
family + feature units + optional meta. Hyperparameters were always the catalog
`params_default`. Operators who wanted a one-off `colsample_bytree=1` or a
bounded Optuna sweep had to edit YAML. ADR-019 also **clears** `search_space` on
every manual candidate so `--tune-trials` cannot override params the user
pinned.

Operators asked to click into a Custom Test model (e.g. xgboost), see catalog
defaults, edit them for that run, tick which knobs belong in an HPO run, and
enable HPO with a checkbox — without mutating the Trade Objective (ADR-018).

## Decision

1. **Catalog remains the SSOT for defaults and HPO ranges.** The feature bank
   exposes each family's `params_default` and `search_space`. The board does
   not invent ranges.
2. **Per-slot `params`** merge over catalog defaults on the compiled manual
   spec (same as handwritten ADR-019 specs).
3. **Per-slot `hpo` + `hpo_params`.** When `hpo` is true, the compiler copies
   the catalog `search_space` subset for the selected keys (empty/omitted
   `hpo_params` = all catalog knobs). Families with no catalog space stay
   fixed and emit a warning.
4. **Board `tune_trials`** becomes `--tune-trials` only when at least one
   candidate has a non-empty `search_space`. `0` with HPO on defaults to 20
   (max 200). Ablation variants each get the same trial budget.
5. **ADR-019 default is unchanged:** omitting `search_space` still clears it.
   An explicit `search_space` on a manual candidate is now legal (opt-in).
   GPU `job_exec` reconstructs that space when the payload carries it; otherwise
   it still clears auto-tuning on user-override candidates.
6. **Do not mutate the objective YAML.** Params/HPO live on the Custom Test
   board and the persisted `ops/webui/custom_tests/<obj>/<run>.*` artifacts.

Rejected: a second HPO engine, editing catalog ranges in the console, and a
board-level HPO switch that would tune families the operator did not open.

## Consequences

- Clicking a Custom Test model card opens the catalog parameter table.
- One Custom Test can mix a fixed xgboost config with an HPO'd copy (two slots)
  or tune a subset of knobs while pinning the rest (`colsample_bytree=1`).
- HPO × ablation is multiplicative (warned). Meta still trains on the full
  (no-omission) variant of each family.
- Iterate-until-target and handwritten manual specs may now carry
  `search_space` without going through Custom Test.

## Related

- [ADR-018](ADR-018-objective-doc-single-source-labels.md)
- [ADR-019](ADR-019-user-override-manual-candidate-spec.md) (amended: opt-in space)
- [ADR-035](ADR-035-webui-custom-test.md)
- Compiler: `src/search/custom_test.py`
- Schema: `config/schemas/custom_test.schema.json`, `manual_candidates.schema.json`
