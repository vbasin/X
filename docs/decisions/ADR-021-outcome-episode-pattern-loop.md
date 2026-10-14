# ADR-021: Outcome Episode Store + Pattern Mining Loop Feeds the Research Director

## Status
Accepted — closes the Phase-23/Phase-20 feedback gap: the Research Director and its two
agents ([ADR-013](ADR-013-two-expert-panel.md)) now consume mined success/failure patterns
from the latest model and strategy outputs. Complements the target-pursuit contract
([ADR-014](ADR-014-requirement-target-kpi.md)) — the target says *how far* the baseline is
from the goal; this says *where and why* it wins, loses, and is blind. Consumes the canonical
prediction frame produced by both model candidates and rule strategies
([ADR-020](ADR-020-rule-strategy-backtest.md)).

## Date
2026-07-04 (Phase 23)

---

## Context

The Research Director loop injects an OBJECTIVE PROGRESS block (per-condition distance to the
target) into every agent's Context Pack, and the agents propose one experiment per cycle. But
progress was the only quantitative signal they received about the *current* model: nothing
told them **which scenarios the model gets right, which it gets wrong, and which true moves it
misses entirely**. The user's requirement was explicit: the Director and its two subagents
should take the latest outputs of the ML model *or* the user's strategy, "carefully assess the
patterns which explain any success and/or failures, and identify how to optimize the missing
piece(s)" — which "implies having specific metadata or some other reference to both successful
and unsuccessful scenarios, in order to find any patterns with the model or strategy behavior."

Aggregate metrics (precision, F1) cannot answer *why*. A per-scenario, context-annotated
record of outcomes is required, plus a deterministic miner that separates the wins from the
losses and misses.

---

## Options Considered

| Option | Pros | Cons |
|---|---|---|
| **A. Let the LLM read raw OOF parquet each cycle** | No new code | Huge token cost; LLM computing statistics is unreliable + non-reproducible; violates "LLM proposes, deterministic code disposes" |
| **B. Only surface aggregate diagnostics (status quo)** | Cheap | Cannot explain *why* a model fails; no per-scenario evidence; the stated requirement unmet |
| **C. Episode store + deterministic miner → compact report into the Context Pack (chosen)** | Reproducible; cheap tokens; keeps the LLM as interpreter; works identically for models & strategies | New store + miner modules to maintain |

---

## Decision

Add two modules under `src/evaluation/` and one Context-Pack section.

### 1. Episode store (`src/evaluation/episodes.py`)
`build_episodes()` derives one row per **scenario** from a canonical OOF frame, classifying
each into a four-way outcome taxonomy:

- `win` — fired, direction matched the true move;
- `wrong_direction` — fired, a true move went the other way;
- `false_positive_flat` — fired, no qualifying move (type-1);
- `missed` — a true move occurred but nothing fired (type-2).

Each episode is annotated with the **market context at that bar** (regime/time descriptors by
default, or the model's own feature list for refinement) and, for strategies, the realised
trade economics. Episodes are persisted append-only under
`data/evaluation/episodes/<objective_id>/` with a JSONL index. `latest_outcome_reports()`
resolves a **program_id → objective_id** prefix fallback (the Director keys context by
`program_id` e.g. `nq-50pt-08-15-cst`; the store keys by `objective_id` e.g. `nq-50pt`).

### 2. Pattern miner (`src/evaluation/patterns.py`)
`mine_patterns()` is **pure pandas/scipy** (no LLM, no network) and computes:

- **win-vs-loss separations** — Cohen's d + Mann-Whitney p per context feature;
- **blind spots** — the same separation between *missed* and *caught* true moves;
- **time-of-day / day-of-week win-rate lift** vs the base rate;
- **confidence calibration** (models) and **directional asymmetry**;
- machine-readable **suggestions** (`add_filter_or_regime_gate`, `add_feature_or_data`,
  `avoid_time_window`, `rebalance_direction`) with cited evidence.

The report carries a compact `summary_text` for prompt injection.

### 3. Context-Pack integration (`src/research_director/context_assembler.py`)
`assemble()` loads the latest outcome reports for the program's objective and the pack's
`render()` emits an **`## OUTCOME PATTERNS`** section. Both agent personas
(`quant_trading.md`, `ml_data_scientist.md`) are instructed to aim each experiment at an
**observed** failure mode and cite the specific pattern (feature, effect size, support) it
attacks. The search controller (`_record_outcome_episodes`) and the strategy runner both write
episodes + reports automatically after every evaluation.

### The invariant
The LLM never computes these statistics — it **interprets** them (LLM proposes, deterministic
code disposes). The same episode/pattern machinery serves ML models and user strategies
identically, because both emit the canonical prediction frame.

---

## Consequences

**Positive**
- The two agents now aim experiments at measured failure modes rather than intuition.
- One reproducible analysis path for models and strategies alike.
- Powers the feature-refinement loop (`src/search/feature_refine.py`): the same separations
  drive add/remove-feature proposals on a pinned model.

**Negative / limits**
- `missed` episodes dominate on long windows; capped by a deterministic thinning
  (`max_missed`) to bound storage — the total count is preserved in `attrs`.
- Cohen's d / Mann-Whitney assume enough per-group support; small cohorts are skipped
  (`_MIN_GROUP`) rather than reported noisily.

**Neutral**
- `scipy` (already a core dependency) supplies Mann-Whitney; no new runtime dependencies.

---

## Related
- [ADR-013](ADR-013-two-expert-panel.md) — the two independent agents that consume this
- [ADR-014](ADR-014-requirement-target-kpi.md) — target distance (complementary signal)
- [ADR-020](ADR-020-rule-strategy-backtest.md) — the strategy path that also emits episodes
- [PHASE-23-MODEL-SELECTION-SEARCH.md](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md) — W-outcome + W-refine workstreams
