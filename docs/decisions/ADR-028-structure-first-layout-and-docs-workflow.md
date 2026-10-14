# ADR-028: Structure-First Layout + Documentation Workflow

## Status
Accepted — 2026-08-05. Inventory: [TREE_INVENTORY.md](../architecture/TREE_INVENTORY.md). Workflow map: [WORKFLOW.md](../architecture/WORKFLOW.md).

## Date
2026-08-05

---

## Context

The platform has grown through Phases 17–27 into a capable multi-plane system (objectives, search, scoping, research director, GPU queue, serving, economics KPI). Capability is ahead of **repository structure and living documentation**:

1. **Directory drift** — `SYSTEM.md` still describes an Attention-LSTM-centric tree; major packages (`search/`, `scoping/`, `serving/`, `gpu_runner/`, `evaluation/`, `backtest/`) are absent from the living layout map.
2. **Import dualism** — Research Director and some services use `from research_director…` with `PYTHONPATH=…/src`, while tests and most code use `from src.…`.
3. **Agent dual stacks** — Scoping v2 and RD each carry LLM/journal/tools/memory concerns with only partial reuse; docs still describe deleted RD modules (`agent.py`, `policy_engine`, `experiment_spec`).
4. **Config/scripts sprawl** — schemas, run specs, runtime YAML, and 47 scripts share flat directories; screenshots landed in `docs/` and `config/agents/`.
5. **Doc type contamination** — `ROADMAP.md` holds long completed Phase 17 archaeology; architecture docs mix target design with present behavior.

Operator direction: **clean structure and docs before** further agentic (G2) and live paper-trading (G3) build-out so those systems grow on a coherent substrate.

---

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Big-bang rename `src/` + rewrite all imports in one PR | Clean end state fast | High break risk; blocks concurrent work |
| B. **Structure-first ADR + inventory; incremental moves; docs truth first** | Safe; reviewable; enables G2/G3 design without freeze | Temporary dual paths during migration |
| C. Docs-only polish, leave tree alone | Cheap | Does not fix import/config/script friction |

---

## Decision

**Option B.** Adopt a **workflow-aligned target layout** and a **strict past / present / future documentation workflow**. Execute P0 as:

1. Inventory every major path (role, plane, disposition) — `TREE_INVENTORY.md`.
2. Document the **as-built** system in living architecture docs (no ghost modules).
3. Standardize on **`from src.<pkg>…` imports** (package root = repo root; `python -m src.…`). Systemd/scripts should set `PYTHONPATH=/opt/vtech` (repo root), not `/opt/vtech/src`, once RD imports are fixed.
4. Relocate non-source noise; group `config/` into schemas/specs/runtime and `scripts/` by plane **incrementally** with loader updates in the same change.
5. Defer physical nest `src/agents/{common,scoping,research_director}` until **P1** extracts a shared agentic substrate (RAG, critic, cost, tool registry).
6. Defer data-storage campaigns and large domain rewrites until P0 map is stable (**P5**).

### Target layout (binding intent; names may refine)

```
src/     acquisition ingestion features objectives models training
         evaluation search backtest serving scoping research_director
         gpu_runner webui common
         + paper/ (P2) + agents/common (P1 extract)

config/  objectives/  agents/  schemas/  specs/  runtime/

docs/    architecture/ (present)  decisions/ (why)  roadmap/ (future)
         changelog/ (past)  reference/  operations/

scripts/ data/ features/ search/ agents/ ops/ gpu/ serving/
```

### Documentation workflow (binding)

| Kind | Content | Update rule |
|------|---------|-------------|
| `architecture/*` | **Present only** | Same change set as behavior/layout |
| `decisions/ADR-*` | **Why** | Append-only; new ADR supersedes |
| `roadmap/ROADMAP.md` | **Future** — active + next | No completed-phase novels |
| `changelog/*` | **Past** | Append on capability complete |
| `reference/*`, `operations/*` | Stable how/what | Edit in place when code changes |

Every `architecture/*.md` carries a short header:

```markdown
> **Present**: …
> **Supersedes**: …
> **Next**: …
```

### Research Director topology (documentation truth)

**As-built (2026-08):** RD is a Scoping-style loop (`orchestrator`: diagnose → cross_review → synthesis → validate → handoff) producing a ranked `ModelImprovementPlan`, not the deleted dual-independent `ExperimentSpec` + `policy_engine` pipeline. ADR-013’s “no merge” principle is **partially superseded in practice** by synthesis; a future ADR may either restore dual-independent experiments or formally accept synthesizer topology. Living `RESEARCH_DIRECTOR.md` must describe **present** code first.

### Import convention

- Preferred: `from src.research_director.x import Y`, `python -m src.research_director.run`
- Disallowed for new code: bare `from research_director…`
- Migration: fix RD package + `scripts/run_rd_on_model.py` + systemd unit in a dedicated follow-up PR; keep temporary dual import shims only if required for one release.

---

## Consequences

**Easier**

- Operators and agents share one workflow map (`WORKFLOW.md`).
- G2 (shared agentic infra) and G3 (paper blotter) have explicit package homes.
- Doc drift has an enforceable rule set in `DOC_LAYOUT.md` / `AGENTS.md`.

**Harder / constraints**

- Incremental moves require dual-path tolerance briefly.
- No deletes without operator approval (AGENTS.md safety).
- Phase-20 roadmap text remains historical; do not execute it literally against deleted files.

**Follow-ups (ordered)**

1. ~~RD import SSOT + systemd `PYTHONPATH`~~ done
2. ~~`config/{schemas,specs,runtime}` moves~~ done
3. ~~`scripts/<plane>/` grouping + Makefile~~ done
4. ~~LLM consolidation under `src/common/llm/`~~ done (plane files = shims)
5. P1 remaining agentic substrate (shared RAG, tools, journals beyond LLM)
6. P2 `src/paper/` + policy bundle
7. P3 enable RD timer
8. P5 storage/data capacity

---

## Related

- Inventory: [TREE_INVENTORY.md](../architecture/TREE_INVENTORY.md)
- Workflow: [WORKFLOW.md](../architecture/WORKFLOW.md)
- Supersedes layout sections of older SYSTEM snapshots (edit living SYSTEM in place)
- Related: ADR-013 (panel topology), ADR-022 (GPU), ADR-025 (scoping), ADR-027 (econ objective)
