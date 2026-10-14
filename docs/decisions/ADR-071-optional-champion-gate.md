# ADR-071: Champion gate is YAML opt-in; Custom Test never gates

## Status
Accepted — implemented 2026-08-27.

Amends [ADR-035](ADR-035-webui-custom-test.md) (Custom Test no longer uses
"the same … gate as search") and [ADR-014](ADR-014-requirement-target-kpi.md)
(requirement/target stay on the objective; the *picker* is optional).

## Date
2026-08-27

## Context

The champion gate (`apply_champion_gate`) is a **picker**: among scored
candidates, those that meet `kpi.requirement` are passers and the best passer
is the champion. Custom Test is not a picker. Operators launch named models
(and optional meta) to inspect OOF, diagnostics, and last-fold bundles.

Custom Test still called the gate. For `nq-50pt-econ` that marked
`tcn_so_mbp_vix` as `failed_ids` / `requirement_unmet` because economics were
scored at the implicit τ=0.8 fire cut (`primary_tau` fallback), even though
the YAML requirement has no precision@τ condition. `run.json` looked like a
failed search.

A related diagnostics gap: `confusion_matrix_raw` is ungated argmax (no τ),
but NE / n_signals / hit_rate were only published at `gate_tau`.

## Decision

1. **`kpi.champion_gate` is opt-in.** Search / HPO / scoping run
   `apply_champion_gate` only when the objective YAML sets
   `champion_gate: true` **and** defines `kpi.requirement` conditions (the
   gate). Omitted or `false` → no picker; the run status is `report` and
   `gate_result` / `champion` are not written. Loader rejects
   `champion_gate: true` with an empty requirement.
2. **Custom Test never gates**, even when the objective has
   `champion_gate: true`. `research/custom_test/<obj>/<run_ts>/run.json` is
   `{kind, status: "report", objective_id, run_ts, out_dir, n_candidates,
   n_ok, n_failed}` — no `gate_result`, no `champion`. Headline scoring is
   the argmax book. Every fitted base and meta still lands under
   `models/<cand>/` (ADR-036).
3. **No inferred confidence cut.** `primary_tau()` never defaults to 0.8.
   A fire-cut is used only when the YAML declares one: optional `kpi.tau`,
   or a `precision_at_tau.tau` on `kpi.requirement`. Otherwise NE /
   n_signals / hit_rate count every argmax up/down (same population as
   `confusion_matrix_raw`). Diagnostics always include `trades_at_raw` /
   `pnl_at_raw`.

Existing objectives that already had a `requirement` set
`champion_gate: true`. Those without a declared τ (e.g. `nq-50pt-econ`)
now score the requirement on the argmax book, not a hidden 0.8.

## Consequences

- Custom Test `run.json` is a report, not a pass/fail. Train failures remain
  `candidates[].status == "failed"` (fit errors), not requirement misses.
- An objective that wants no picker omits `champion_gate` or sets it false;
  `kpi.requirement` can stay as documentation of the bar.
- Iterate-until-target still reads `gate_result.target_ids`; that path needs
  `champion_gate: true` on the objective.

## Related

- [ADR-014](ADR-014-requirement-target-kpi.md) — requirement vs target
- [ADR-035](ADR-035-webui-custom-test.md) — Custom Test compiler
- [ADR-036](ADR-036-custom-test-lab-artifacts.md) — lab bundles
- `src/search/gate.py` (`champion_gate_enabled`, `resolve_champion_gate`)
- `src/search/controller.py` (`custom_test_run_doc`)
