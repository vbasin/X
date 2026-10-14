# ADR-136: Canonical Research Architecture

**Status**: Accepted
**Date**: 2026-09-22
**Supersedes**: none (additive restructuring; no prior ADR is invalidated)

## Context

An independent, code-verified audit (2026-09-22, six parallel adversarial research passes against executable code, `systemctl` state, and real data artifacts — not documentation) established that this repository contains three architecturally disconnected model-development efforts:

- **Track A** — legacy 3-class detection models (`model_v001`-`model_v015`), Phase 1-22 roadmap. Confirmed dead: 3.5 months stale, zero live code references, lives under `data/legacy/`.
- **Track B** — `nq-50pt-econ`, pinned by `config.yaml`'s `objective_id` (ADR-006 single source of truth), evaluated via Research Director / Custom Test Lab (ADR-025, ADR-027, ADR-036). The only track wired to live serving (`src/webui/live.py`) and promotion (ADR-071). Confirmed currently non-functional as a predictive system: every recent candidate run (Aug 2026) collapses toward the majority class; the meta-stacking layer produces zero signals in every run inspected; the one model ever promoted (2026-08-05) realized ~26x worse expectancy in paper trading than its own backtest claimed, and paper trading has been inactive since.
- **Track C** — the MNQ Auction Cascade (ADR-127, ADR-130, ADR-134). The most actively developed track and, on independent re-audit, the one with the strongest causal-engineering hygiene in the repository — but not wired to `config.yaml`, live serving, or promotion at all, and its headline economics are optimistic-case-only: its WebUI-facing backtest has zero execution-cost modeling, and its own console-only cost-sensitivity check shows the edge is not statistically significant at 1 tick of slippage and goes negative at 2 ticks.

No document anywhere in the repository compares or reconciles these three tracks. A fresh agent (or this session, before the audit) had to reconstruct this state through substantial code archaeology.

Beyond the three-track fragmentation, the audit found: configuration fields that are silently dead or contradicted by actual runtime behavior (`config.yaml`'s `attention_heads`/`class_weight_directional` block; `slippage_model` never translated in `src/backtest/engine.py`); four mutually inconsistent implementations of economic evaluation; no calibration anywhere in either live track; validation infrastructure stronger than what real runs actually use (`n_folds=1` despite genuine walk-forward machinery existing); and a confirmed, untooled population-shift confound in Track C's gated cascade.

## Decision

Adopt a **track-agnostic canonical architecture** organized around 12 durable research responsibilities (data contracts, feature builder, outcome builder, model strategy, validator, calibrator, decision policy, economic evaluator, experiment, model lifecycle, serving, orchestrator — full definitions in `docs/ARCHITECTURE.md`), derived independently from the project's actual trading objective and MLOps/quant-research best practice, **not** by renaming Track B's or Track C's existing module boundaries. Track-specific vocabulary ("Stage 1", "gate", "meta model", "champion", "objective") remains implementation-specific — describing one strategy inside a responsibility, not architecture that future work must conform to.

The canonical architecture lives under `src/`, alongside (not replacing) Track A/B/C's existing code. This pass is structural only: it does not change any label, feature, model, threshold, or execution-behavior math (see Non-goals). New canonical modules are either (a) real, tested, and directly extracted from an already-verified-correct existing pattern (`data_contracts/causal.py`'s as-of primitive, ported from Track C's `_asof()`), (b) real, tested, and net-new because no working implementation of that responsibility exists anywhere (`evaluation/economic.py`, `experiment/manifest.py`, `lifecycle/status.py`), or (c) interface-only stubs documenting which existing Track B/C module currently fulfills that responsibility, pending future wiring.

Migration follows a strict copy-first discipline: nothing existing is moved, renamed, or deleted. The one exception — `AGENTS.md`, rewritten in place to add project orientation it previously lacked — has its pre-restructuring content preserved verbatim at `docs/archive/AGENTS.pre-canonical-2026-09-22.md` before any edit. Full accounting in `docs/MIGRATION_MANIFEST.md`.

Agent-instruction architecture is made vendor-neutral: `AGENTS.md` is the canonical cross-agent operating contract; `CLAUDE.md` is a minimal adapter that defers to it rather than duplicating project rules.

## Non-goals (explicitly out of scope for this ADR)

- Changing any label/outcome mathematics, target/stop mathematics, feature definitions, model algorithms, thresholds, gating mathematics, training objectives, or trade-selection logic on either live track.
- Wiring `src/experiment` or `src/evaluation` into any existing Track B/C script's actual execution path (flagged as the highest-value next step in the audit, deferred to a follow-up task with explicit behavior-preservation validation).
- Choosing a "winner" between Track B and Track C, or promoting either. This ADR resolves *structural* ambiguity about what's canonical architecture; it does not resolve the open scientific question of whether either track has real trading edge net of realistic costs (see the audit's Section L).
- Classifying the existing 136 ADRs by status (Accepted/Superseded/Deprecated) — flagged as follow-up in `docs/MIGRATION_MANIFEST.md` §5.
- Editing `docs/roadmap/ROADMAP.md`'s orphaned Track A Phase 20-22 entries.
- Any database access, data mutation, or physical relocation of `data/` — data is read-only external state for this ADR's scope per the task's data-safety boundary.

## Consequences

**Positive**: a fresh agent (or coding-agent session with zero conversational history) can now determine, from `AGENTS.md` → `project_manifest.yaml` → `docs/ARCHITECTURE.md`, what's active/experimental/production/deprecated, what config is authoritative, and where known dead/contradictory config fields are — without the code archaeology this ADR's founding audit required. Economic-evaluation and experiment-identity now have a single, tested, correct reference implementation to converge on, even though nothing consumes it yet.

**Negative / residual risk**: three tracks remain functionally disconnected — this ADR documents and structurally accommodates that fact, it does not resolve it. `src/`'s interface-only subpackages could drift out of sync with Track B/C's actual implementations if `project_manifest.yaml` isn't kept current (explicit maintenance note left in that file). The real, high-value wiring work (Economic Evaluator, Experiment manifest, config validator) remains undone; this ADR makes that debt visible (`docs/MIGRATION_MANIFEST.md` §5) rather than paying it down.

## References

- Full audit (six-agent adversarial pass, this session, 2026-09-22): summarized in `project_manifest.yaml`'s `open_critical_findings`.
- ADR-006 (config.yaml single source of truth), ADR-018 (objective doc single source of labels/KPI), ADR-025 (Scoping v2), ADR-026 (PnL economics), ADR-027 (economics/meta/sparse objective), ADR-028 (structure-first layout), ADR-071 (optional champion gate), ADR-127/130/134 (cascade architecture) — none superseded by this ADR; all remain accurate implementation-level history for their respective tracks.
