# ADR-013 — Two Independent End-to-End Expert Agents (Supersedes the Four-Expert Panel)

**Date**: 2026-06-30  
**Status**: Accepted (design)  
**Relates to**: [RESEARCH_DIRECTOR.md](../architecture/RESEARCH_DIRECTOR.md)  
**Supersedes**: the four-expert advisory **panel + Director synthesis** design in
[PHASE-19-RESEARCH-DIRECTOR.md](../roadmap/PHASE-19-RESEARCH-DIRECTOR.md) (#9, #10, #26, #29, #30)

---

## Context

The Phase-19 build introduced a **four**-expert advisory panel — `quant_trading`,
`ml_data_scientist`, `systems_architect`, `data_engineer` — where each expert only mines
*findings* and a central **Director synthesis** step clusters/corroborates them and a single
central **Hypothesis Planner** turns the top cluster into one experiment. The redesign
target mandates a different topology: **exactly two** expert agents, each running a **fully
independent end-to-end workflow** (assess → research → hypothesize → **plan its own
experiment** → execute → evaluate → post-analyze → journal), with **no merge, clustering,
or corroboration** between them.

Empirically, the `systems_architect` and `data_engineer` lenses overlap heavily with the
`ml_data_scientist` lens (leakage, reproducibility, scaler choice, fold integrity are all
data-science evaluation concerns), and their *actionable* output is mostly deterministic
checks (split integrity, provenance completeness, locked-section safety) that the Policy
Engine and Evaluator already enforce in code. Four LLM lenses also multiply token cost. A
central synthesis/planner adds a coupling point that conflicts with the requirement that
each agent independently arrive at and own its experiment.

## Decision

1. **Two independent agents**: `quant_trading` (alpha / tradeability / market structure /
   label economics) and `ml_data_scientist` (architecture / training / labels /
   evaluation / **data-quality + leakage + reproducibility checklist**).
2. **Fold-in, not drop**: the `systems_architect` reproducibility/leakage concerns and the
   `data_engineer` data-quality/feature-staleness concerns become a **mandatory checklist
   section** inside the `ml_data_scientist` agent. Infra/cost-feasibility checks remain
   deterministic Policy-Engine gates, not an LLM lens.
3. **Each agent owns a full end-to-end workflow** — independent assessment, bounded
   research (tools + read-only external literature search), a falsifiable hypothesis,
   **its own experiment plan/spec**, a self-critique/reflection pass, execution, and
   post-analysis — all journaled to the agent's own thread + vector memory.
4. **No synthesis / no corroboration**: the Director **coordinates** (launches both agents,
   owns the single-writer journal, splits the per-cycle budget, schedules the GPU queue)
   but does **not** merge, cluster, corroborate, or adjudicate the agents' ideas. If both
   agents produce a viable experiment in a cycle, **both** are validated and run
   independently (budget permitting). Resource contention is resolved deterministically
   (FIFO GPU queue, per-agent budget slice), never by reconciling ideas.
5. **Agentic harness**: each agent is a tool-using loop (plan → act → observe → reflect)
   with a typed tool registry (context/inventory/research/attribution/memory readers),
   skills, just-in-time context, bounded passes, and structured-output validation.

## Consequences
- ~50% fewer cognitive-layer lenses and a simpler control flow (no clustering/adjudication
   code path).
- `config/research_director.yaml` `panel.experts` shrinks to two; the `synthesis.*` block
   (cluster threshold, `w_relevance`/`w_issue_level`/`w_corroboration`, `round2_cross_review`)
   is **deprecated**; `schemas.Lens` drops two members;
   `config/agents/{systems_architect,data_engineer}.md` are retired (content merged into
   `ml_data_scientist.md`).
- `planner.py` logic moves **inside** each agent (per-agent hypothesis planning);
   `experts.py` becomes an agentic runner; `tools.py` gains DB-inventory, dataset-catalog,
   and gated external-research tools.
- Loss of two *independent* infra/data lenses is mitigated by moving those checks into
   deterministic gates (Policy Engine, Evaluator) where they are enforced, not merely advised.
- Trade-off accepted: dropping cross-agent corroboration removes the "independent agreement"
   signal, but the requirement prioritizes **two independent, diverse experiment streams**
   per cycle over a single corroborated one — diversity of experiments is the goal.
