"""Research Director — agentic model-improvement orchestration.

Two field-expert agents (`quant_trading`, `ml_data_scientist`) study the EXISTING model and the
patterns in its results and prescribe the single highest-leverage change; a cross-model critic
forces deeper thinking; a judge synthesizes one ranked ModelImprovementPlan; the top pick is
auto-validated (trained + scored vs the baseline) and the outcome is journaled. Mirrors the
Scoping-v2 loop (`src/scoping/`), reusing its critic + RAG + bounded-round primitives.

Modules
-------
orchestrator        The 5-stage loop: diagnose → cross_review → synthesis → validate → handoff.
director            Thin cycle driver: governor gate → assemble → stand-down → orchestrator → journal.
run                 Production entrypoint (`python -m src.research_director.run`).
config              RDConfig.load(): program context + Governor/MLflow knobs (rd_loop lives in YAML).
context_assembler   Builds the Context Pack (existing model + result patterns) handed to both agents.
memory              Per-agent semantic memory (public API over rag_memory).
rag_memory          pgvector per-agent memory substrate (research_director.agent_memory).
tools               Context / inventory / research readers available to agents.
registry            In-process typed tool registry (no MCP) + the critic verification registry.
skills              Reusable agent procedures (playbooks).
llm                 Shim → src.common.llm (OpenAI-compatible client); prefer make_llm.
cost                Shim → src.common.llm.cost (CostMeter bands).
journal             Single-writer journal: PROPOSED → PLAN_SYNTHESIZED → terminal verdict.
governor            Pre-cycle AND-gate (cost caps, suppression windows, advisory lock).
evaluator           Deterministic baseline-relative gate (reused by the validation stage).
objective_progress  Distance-to-target assessment (ADR-014 stand-down).
sentinel            Observability + tiered alerting.
mlflow_tracking     MLflow run-nesting + cost/verdict logging (no-op fallback).
"""

from __future__ import annotations

__all__ = [
    "orchestrator",
    "director",
    "run",
    "config",
    "context_assembler",
    "memory",
    "rag_memory",
    "tools",
    "registry",
    "skills",
    "llm",
    "cost",
    "journal",
    "governor",
    "evaluator",
    "objective_progress",
    "sentinel",
    "mlflow_tracking",
]
