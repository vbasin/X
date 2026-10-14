# ADR-027: Economics-Primary Ranking + Primary/Meta Topology + Sparse Feature Views

## Status
Accepted — implemented 2026-07-15. Builds on [ADR-014](ADR-014-requirement-target-kpi.md)
(requirement/target), [ADR-018](ADR-018-objective-doc-single-source-labels.md) (objective SSOT),
[ADR-023](ADR-023-triple-barrier-label-scheme.md) (TBM labels), [ADR-026](ADR-026-pnl-economics-and-predictions-backtest.md)
(label-implied PnL), and [ADR-017](ADR-017-requirements-first-stage0-feature-views.md) (feature views).

## Date
2026-07-15

---

## Context

Architecture review of the VTech research stack identified three decision-surface gaps that
prevent the platform from optimizing *trading* rather than *classification*:

1. **W1 — Classification-first champion selection.** ADR-026 already computes
   `net_expectancy` on every run, but the champion gate ranked passers only by
   `precision_at_tau` then `directional_f1`. A high-precision / low-expectancy model
   could outrank a better economic policy.
2. **W3 — Meta-labeling prepared but inert.** `src/training/meta_labels.py` and
   `config/meta_ab.yaml` existed; nothing in search composed a primary + trade/no-trade
   gate. Sparse directional barriers are the textbook use case for López de Prado
   meta-labeling.
3. **W4 — Full feature matrix by default.** Curated views existed for regime families,
   but tree/sequence search usually trained on the full ~183-col matrix unless a manual
   override said otherwise.

These needed to be proven **in code** as an opt-in objective, without breaking
`nq-50pt` / `nq-50pt-tbm` semantics.

---

## Decision

### 1. New objective `nq-50pt-econ` (same TBM barriers, different decision surface)

- `labels.scheme: triple_barrier` (50 / 6 / 10m) — same ground truth as `nq-50pt-tbm`.
- `costs.cost_per_trade: 1.0` so expectancy is net of friction.
- `kpi.primary: net_expectancy` — the gate **ranks** passers by economics.
- `kpi.requirement` (ALL): `net_expectancy ≥ 12`, `n_signals ≥ 30`, overall
  `precision_at_tau@0.8 ≥ 0.70`, `max_drawdown ≥ -250` (floor).
- `kpi.target` (ALL): stronger expectancy + hit rate + both-side precision@0.9.
- Hold-out pin reuses the TBM contract hash `e68510982d9f` (same parquet vintage).

### 2. Schema + criteria + gate (W1)

- `objective.schema.json`: economics metrics in conditions
  (`net_expectancy`, `hit_rate`, `sharpe`, `max_drawdown`, `n_signals`);
  `kpi.primary` enum includes economics; `search.feature_views` /
  `default_feature_view` / `topology` formalized.
- `kpi_criteria.py`: reads economics from the compact score (and `pnl_at_gate` fallback);
  overall `precision_at_tau` when `direction` is omitted.
- `gate.py`: ranks passers by `kpi.primary` — economics primaries use
  `(net_expectancy|hit_rate|sharpe, precision)` instead of `(precision, dir_f1)`.

Legacy objectives keep `primary: precision_at_tau` → **no ranking regression**.

### 3. Primary + meta topology (W3)

- `src/search/meta_compose.py`: time-ordered K-fold cross-fitted binary meta-gate on
  primary OOF; features = confidence + side (+ optional `context_view` columns).
- Controller Stage 2b: when `search.topology.mode: primary_meta`, emit
  `meta__<primary_id>` candidates scored on the same contract.
- `config/meta_ab.yaml` status → `active` for the economics objective.

### 4. Sparse feature views as default (W4)

- New view `momentum_edge` in `feature_views.py`
  (`of_ bp_ vw_ rv_ ht_ dc_ sd_ tc_ tl_ ss_`).
- Objective declares per-family views + `default_feature_view: momentum_edge`.
- Controller resolution order:
  explicit selection → objective family view → catalog view →
  **objective default** → full matrix.

---

## Options considered

| Option | Why not chosen |
|--------|----------------|
| Change `nq-50pt-tbm` in place | Breaks historical leaderboard comparability |
| End-to-end RL as default | Sample-hungry; unstable on rare signals |
| Path-PnL inside every trial | Too expensive; Tier A expectancy is the right screen |
| Always-on meta for all objectives | Surprising; keep opt-in via topology |

---

## Consequences

- **+** Economics-aware selection is real, not only a diagnostic column.
- **+** Meta-labeling is an executable search stage, leakage-controlled via cross-fit.
- **+** Sparse views are objective-owned defaults, not tribal knowledge in manual specs.
- **−** `nq-50pt-econ` needs its first full search run before a champion exists
  (dry-run + unit tests prove plumbing; live training is operator-initiated).
- **−** Meta context join loads recent feature parquets (capped); large caches add I/O
  when topology is on.

## Proof artifacts

- Objective: `config/objectives/nq-50pt-econ.yaml`
- Manual spec: `config/manual_candidates.nq-50pt-econ.yaml`
- Tests: `tests/test_econ_objective.py`
- Contract: `data/evaluation/nq-50pt-econ/contract_v1.json`
