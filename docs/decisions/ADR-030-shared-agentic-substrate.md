# ADR-030: Shared Agentic Substrate (`src/agentic`)

## Status
Accepted — implemented 2026-08-05 (P1). Builds on ADR-028 (structure-first),
ADR-025 (scoping RAG + critic), and the LLM consolidation under `src.common.llm`.

## Date
2026-08-05

---

## Context

Scoping and Research Director each grew overlapping agent infrastructure:

| Concern | Pre-P1 home | Problem |
|---------|-------------|---------|
| LLM clients / cost | Split then moved to `src.common.llm` (P0) | Solved |
| Tool registry | `research_director.registry` | Scoping built ad-hoc registries |
| Research retrieval | RD: substring `ResearchTool`; Scoping: pgvector | Dual quality / no shared import |
| Critic harness | `scoping.critic` only | RD reused via deep imports |
| Embeddings | `scoping.embeddings` | RD memory imported private RAG helpers |

P1 needs a **single import surface** for anything an agent harness shares, without
forcing a big-bang move of Scoping/RD plane logic into one mega-package.

---

## Decision

1. **Package `src/agentic/`** is the shared substrate (not plane-specific):
   - `registry` — `Tool` / `ToolRegistry` / `obj_schema`
   - `rag` — `research_search` (pgvector **primary**, substring **fallback**),
     re-exports `search_research` / `citation_exists`
   - `critic` — re-exports Scoping critic harness
   - `embeddings` — re-exports embedder factories
   - `memory` — re-exports RD agent memory API

2. **LLM stays in `src.common.llm`** (providers are infrastructure, not “agent plane”).

3. **RD default/critic registries** register `research_search` / `search_research`
   against `src.agentic.rag.research_search` (not grep-first `ResearchTool`).

4. **`ResearchTool`** remains as a thin shim for legacy call sites (Scoping agent)
   and delegates to `research_search`.

5. **Scoping RAG table vs RD agent_memory** stay separate tables (rebuildable corpus
   vs durable primary memory) — shared is the **retrieval API + embedder**, not one table.

---

## Consequences

**Easier**

- New agents import `src.agentic` + `src.common.llm` only for harness plumbing.
- One research quality bar (pgvector) for both loops.

**Harder**

- `src.agentic.rag` depends on `src.scoping.rag` implementation (acceptable; can invert
  later by moving pgvector core under agentic if scoping shrinks).

**Out of scope for this ADR**

- Enabling the RD systemd timer (P3).
- Paper blotter (P2).
- Physical nest `src/agents/{scoping,research_director}` rename (optional later).

---

## Related

- [WORKFLOW.md](../architecture/WORKFLOW.md)
- ADR-025 scoping v2 · ADR-013 two-expert panel · ADR-028 structure-first
