# ADR-014 — Two-Tier Success Criteria: `requirement` (gate) vs `target` (goal)

**Date**: 2026-07-01  
**Status**: Accepted (design). Amended by [ADR-071](ADR-071-optional-champion-gate.md):
the champion *picker* is opt-in (`kpi.champion_gate`); Custom Test never applies it.  
**Relates to**: [PHASE-23-MODEL-SELECTION-SEARCH.md](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md),
[ADR-012](./ADR-012-gated-promotion-mlflow-lineage.md) (champion/promotion gate it feeds)  
**Supersedes**: nothing (first ADR to formalize the objective KPI as a two-tier
requirement/target contract; refines the single-threshold gate sketched in ADR-012)

---

## Context

The `nq-50pt` objective originally encoded success as a single flat threshold block:

```yaml
kpi:
  primary: precision_at_tau
  thresholds:
    dir_f1: 0.75
    precision_at_tau: {"0.8": 0.85}
```

Two problems surfaced once real runs started (Phase 23 W3/W8):

1. **`dir_f1: 0.75` is the wrong bar.** The trade objective is *precision at a confidence
   threshold* (only fire when confident, tolerate low recall). Maximizing directional F1 pushes
   the opposite way: the W8 HPO run (60 trials, xgboost) hit `dir_f1 = 0.888` by firing on
   ~everything (22k up / 17k down) at **precision ≈ 0.32/0.36**. A single F1 threshold rewards
   the fire-on-everything regime the objective is meant to avoid.
2. **One number conflates two distinct decisions.** "Is this model good enough to *consider*?"
   and "Has the program *reached its goal*?" are different bars. The first selects a champion
   from a search; the second says the whole experimentation effort can stop. Collapsing them
   into one threshold makes the gate either too strict (nothing is ever a champion) or too lax
   (a mediocre model is declared a success).

## Decision

Split the objective's `kpi` block into two declarative condition blocks:

- **`requirement`** — the bar a candidate must clear to be *considered* by the champion gate.
  Pass = candidate is a *passer*; the highest-ranked passer becomes the champion. For `nq-50pt`,
  ANY of: up/down `precision@0.8 ≥ 0.85`, or up/down `precision@0.9 ≥ 0.90`.
- **`target`** — the experimentation goal, surfaced as a flag (`meets_target` /
  `champion_meets_target`), **never a gate**. For `nq-50pt`, ALL of: both directions
  `precision@0.9 ≥ 0.95` AND both directions `recall > 0.5`.

Each condition is `{metric, direction?, tau?, min, op?, min_signals?}`:
- `metric ∈ {precision_at_tau, recall, directional_f1, f1}`; `op ∈ {gte (default), gt}`.
- A block is exactly one of `any` (OR) / `all` (AND).
- `precision_at_tau` conditions enforce a `min_signals` floor (default 10) so a lucky handful of
  high-confidence fires cannot satisfy a precision bar (e.g. down`@0.9 = 1.0` on 3 signals fails).

Mechanics:
- `src/evaluation/kpi_criteria.py` is the single evaluator (`meets_requirement`, `meets_target`,
  `objective_taus`, `primary_tau`). Metric values are read from each candidate's **diagnostics**
  (τ sweep + per-class recall), so the gate needs no extra computation.
- `config/objectives/objective.schema.json` validates the structure (`$defs.kpi_condition`,
  `$defs.criteria_block`; `requirement` + `target` required).
- The champion gate (`src/search/gate.py`) uses `requirement` for pass/fail and annotates
  `target_ids` / `champion_meets_target`. `directional_f1` remains a *ranking tiebreaker* and a
  reported diagnostic — it is no longer a gate.
- `primary_tau` = the lowest `requirement` τ (0.8) — the headline scoring τ; `objective_taus`
  drives the diagnostics sweep.

## Consequences
- The gate now matches the objective's economics: precision-at-confidence decides who is
  considered; recall enters only at the (stricter) target bar.
- Clear separation of "champion of this search" from "program goal reached" — the search can
  promote a best-so-far champion every iteration without ever falsely declaring success.
- The W8 HPO objective is now known to be misaligned (it maximizes `directional_f1`); follow-up
  is to re-point HPO at the `requirement` metric (`precision@τ` with a min-signal floor).
- Metrics are **not comparable** to any pre-2026-07-01 gate decision that used the `dir_f1: 0.75`
  threshold; the old `kpi.thresholds` shape is removed (readers migrated to `kpi_criteria`).
- Adding a metric or bar is now a config edit (new condition) rather than a code change.
