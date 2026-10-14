# ADR-025: Scoping v2 — Dual-Perspective, Evidence-Tiered Agentic Research Loop

## Status
Accepted & **IMPLEMENTED 2026-07-14** (all work items below ticked; suite 146 green); supersedes the **single-agent** Stage-0 design of Phase-23 **W2.5** (`src/scoping/runner.py`, `strategy_research.md`). Builds on [ADR-013](ADR-013-two-expert-panel.md) (two-expert panel, Director-as-coordinator), [ADR-021](ADR-021-outcome-episode-pattern-loop.md) (episodes/patterns), [ADR-024](ADR-024-unified-experiment-artifact-flow.md) (MLflow lineage, durability). Aligns with Phase-20 **W2** (agentic harness with per-agent self-critique pass, in-process tools/ACI, no MCP).

## Date
2026-07-14

---

## Context

Stage-0 scoping today is a single `strategy_research` agent that (1) derives a data scope and
(2) runs one external research pass into a scored publications library, then a deterministic
data-gap diff. It does **not** propose the ML starting point (model types/combos land later, in
the search stage), does not consult the system's own accumulated evidence (results_index,
MLflow's 380+ logged runs, episode/pattern mining), has no proven-live memory tier, and its
consumers ingest research by **whole-file prompt-stuffing** (no retrieval). The goal: refine
scoping into an iterative, dual-perspective agentic workflow that digests an objective and
produces the **optimal starting point** — model type/combo, data/features/symbols/timeframes —
with full evidence provenance, feeding the existing full-fidelity search controller.

Design constraints fixed during review (2026-07-14 design session):

- **Perspectives**: ML data science + finance quant trading — the existing ADR-013 personas
  (`ml_data_scientist.md`, `quant_trading.md`), not new agents.
- **Pass isolation**: each research pass sees only its own sources + the prior pass's output
  (Pass A starts blank-slate) — for *drafting*; synthesis stages see everything.
- LLM cost / call count is explicitly **not** a constraint; iteration is bounded by
  **convergence detection + caps**, not budget (unbounded loops fail by oscillation, not cost).

## Decision

### 1. Stage structure (deterministic orchestrator; agents only where reasoning lives)

```
Objective digest
  → Pass A   External research (T1)              [dual persona, blank slate]
  → Pass B1  Internal experimental evidence (T2) [automatic sources]
  → Pass B2  Proven-live journal (T3)            [user-confirmed entries only]
  → Preliminary Research Results (synthesis)     [panel debate → scoping_plan.json]
  → Pass C   Deterministic gap check             [code, no LLM]
  → Final Research Report                        [human gate]
  → existing search controller                   [the arbiter — scoping only proposes]
```

Workflow-over-autonomy: the stage graph is known, so a deterministic state machine sequences
bounded agent stages (orchestrator-workers + evaluator-optimizer patterns); every artifact is
schema-validated; "LLM proposes, controller disposes" holds throughout — scoping's plan is a
**starting-point hypothesis set**, never a verdict.

### 2. Evidence tiers with explicit trust ordering — T3 > T2 > T1

| Tier | Source | Write policy |
|------|--------|--------------|
| **T1** external | scored publications library (`research_library.json`, append-only accumulation across objectives) + `research_cases/` corpus + new reputable-source research | agents append freely, with notes + scores |
| **T2** internal experimental | `results_index/` warm-start priors, leaderboards, episode/pattern reports, MLflow `search/<objective_id>` runs | automatic — machine-generated, no gate |
| **T3** proven-live journal | validated live-run results (config, live period, realized metrics, `mlflow_run_id` lineage) | **user-confirmed writes only** (`python -m src.scoping.journal add --from-run <id>`, interactive confirmation) — guards the highest-trust memory against self-reinforcing noise / memory poisoning |

Pass B is split B1/B2 so the gated T3 tier never blocks the rich automatic T2 evidence.

### 3. Retrieval substrate: pgvector RAG + file/object-storage durability (complementary layers)

Bare object storage is key-addressed — not a retrieval system; whole-file prompt-stuffing does
not scale with an accumulating corpus (context dilution). Therefore: **artifacts** stay
file-based (`data/scoping/…`, `research_cases/`) synced to `vtech_archive` (ADR-024 pattern;
`data/scoping` added to sync targets), while a **pgvector** index in the existing Postgres
(extension install required; inherits pgBackRest durability; SQL-joinable metadata) holds
embedded chunks tagged `(tier, traits, source_ref, objective_id)`. Agents retrieve via one tool
— `search_research(query, tier, traits, k)` — making research **agentic RAG** (retrieval inside
the reasoning loop). Index is rebuildable from artifacts. Embedding backend: local
sentence-transformers (CPU) default, API embeddings (Voyage/OpenAI) configurable — Anthropic
has no embeddings endpoint.

### 4. Review topology: author self-critique → parameterized critic harness → interface-scoped cross-review → judge

Rationale: LLMs are weak at finding their *own* reasoning errors (error-generating and
error-checking distributions coincide); decorrelation is what buys detection.

1. **Author self-critique** (cheap pre-pass, ≤3 iterations — already planned in Phase-20 W2):
   surface slips only.
2. **Single parameterized critic harness** — one `critic.md` role (shared adversarial
   instructions + objection schema), **composed at runtime with the reviewed field's persona
   as reference knowledge** (`critic(field=quant)` / `critic(field=ml)`), run **fresh-context**
   (sees the deliverable + the author's **retrieval trace**, never the reasoning scratchpad —
   detects both "retrieved-and-ignored" and "never-looked" while preventing anchoring) and on
   a **different model family** than the author (both providers already supported by
   `llm_factory`). Supersedes the interim "two critic personas" idea: critique decomposes into
   a domain-general layer (sharable) + a domain-specific layer (injected expertise); a
   knowledge-free generic critic cannot catch deep field errors, and two hand-maintained
   critics duplicate ~70% shared machinery. Per-field red-flag checklists are **config**, not
   separate personas. Precedent: `code_reviewer.md` author–critic split.
3. **Critic methodology: artifact-first → targeted verification — never full re-derivation.**
   Full replication collapses the author/critic asymmetry (two rival plans + a reconciliation
   problem instead of focused objections); artifact-only misses errors of **omission**.
   Sequence: (a) artifact-first read → structured flags; (b) targeted tool use per flag:
   *citation grounding* (citation-exists is a deterministic RAG-index check, not an LLM call),
   *omission probes* (coverage queries on the objective's key axes), *adversarial
   counter-evidence search* (actively try to falsify the headline recommendation — the search
   intent the author structurally under-runs); (c) objections must carry severity + evidence
   citation; "no objections" requires completing an explicit verification checklist
   (anti-sycophancy).
4. **Interface-scoped cross-review** (not unscoped): quant reviews the ML plan's **market/data
   realism**; ML reviews the quant plan's **learnability/statistical feasibility**. Every
   cross-objection must tag the interface assumption it attacks; out-of-domain objections are
   dropped by schema. Preserves ADR-013 independence (no merged outputs).
5. **Judge** = Director/coordinator (ADR-013 role): weighs objections under T3>T2>T1 trust,
   convergence = no material objections (cap 4 rounds).
6. **One full-replication second opinion** at the **synthesis node only** (the highest-stakes
   decision gating data purchases / model-combo commitment): a single independent derivation,
   reconciled by the judge. Replication is spent where being wrong is expensive — not per claim.

### 5. Synthesis + gap check + report

- **Preliminary Research Results**: panel debate → `scoping_plan.json` (schema'd): model plan
  (families / combos / ensemble topology with per-choice rationale + tier-tagged evidence
  citations) and per-model data requirements (symbols, market/non-market data types, raw vs
  engineered features, timeframes), plus KPI sanity check.
- **Pass C (deterministic, code)**: diff the plan against `data/catalog/data_capability.json`
  (exists) **+ model catalog / plugin coverage (new) + feature catalog coverage (new)** →
  criticality-ranked `data_gap_report.json` v2 (criticality = impact on plan feasibility).
- **Final Research Report**: `final_research_report.md` (plan, evidence trail with T1/T2/T3
  citations, gaps + criticality, acquisition cost estimates, outstanding user items) **+
  machine-consumable outputs**: draft `manual_candidates.yaml`, suggested objective-doc patch
  (`eligible_model_families`, `search.feature_views`), aux-dataset specs → **human gate**
  (acquire / proceed) → search controller.
- **Corpus write-back**: critic-surfaced omissions and new findings are appended to the T1
  library + indexed, so every objective's scoping permanently improves the corpus.

### 6. Observability

Every scoping session logs to MLflow experiment `scoping/<objective_id>` — parent run per
session, nested runs per pass/agent round, all JSON/MD artifacts attached (consistent with
ADR-024's "verify everything in one UI"; W7's MLflow-3 tracing/prompt-registry slots in).

## Options Considered

| Option | Verdict |
|--------|---------|
| Single autonomous mega-agent for all of scoping | rejected — known stage structure ⇒ workflow beats free-form autonomy on reliability/reproducibility |
| Two hand-maintained critic personas (`quant_critic.md`, `ml_critic.md`) | rejected — duplicates the shared critique layer; superseded by the parameterized harness |
| One knowledge-free "senior critic" applied to any agent | rejected — cannot catch domain-specific errors (the ones self-critique also misses) |
| Critic fully re-runs the author's research | rejected as default — collapses author/critic asymmetry, convergence noise; kept **once** at synthesis as the second opinion |
| Unscoped cross-review between fields | rejected — out-of-domain objections are noise; interface-scoped only |
| RAG in bare object storage / FAISS file / managed vector DB | rejected — no semantic retrieval / single-writer ops burden / new infra+cost; pgvector in existing Postgres wins |
| A→B order inverted (internal first) | kept A-first — blank-slate external avoids anchoring on possibly-stale internal wins |

## Consequences

- **+** Scoping produces a defensible, evidence-cited starting point (model combo + data plan)
  instead of a data scope alone; the search stage starts warm.
- **+** Provenance is explicit at every hop (tiered citations, retrieval traces, MLflow runs).
- **+** The corpus compounds across objectives (append-only library + write-back + RAG).
- **−** New moving parts: pgvector + embedding backend, critic harness, journal CLI.
- **−** Cross-model critique requires a second provider key configured.
- **Constraint**: scoping never decides — the deterministic gap check and the full-fidelity
  search controller remain the arbiters; the human gate remains at the irreversible boundary
  (data acquisition / plan commit).

---

## Implementation Plan (full work-item list)

> **Implemented 2026-07-14** — every item below shipped and tested (`tests/test_scoping_v2.py`, 12 tests;
> full suite 146 green). Deviations from the letter of the plan (intent preserved):
> - **Persona prompt sections** live in `config/agents/scoping_duties.yaml` (runtime-composed with the
>   base personas by the orchestrator) rather than appended to `quant_trading.md`/`ml_data_scientist.md` —
>   keeps Research-Director cycles free of scoping-only prompt text; red-flag checklists share that file.
> - **Embedding default**: the adapter's auto order is local sentence-transformers → OpenAI-compatible API
>   (`RD_EMBED_MODEL`) → deterministic hash fallback. sentence-transformers is not installed on the CPU VM
>   (torch weight), so auto currently resolves to the API backend (`text-embedding-3-large`, 3072-dim);
>   installing the package flips the default to local with no code change. Corpus indexed live: 95 chunks.
> - **Pass B1/B2** are two evidence inputs (automatic T2 digest + gated T3 journal read) integrated in one
>   per-persona revision round, not two separate LLM rounds — the split that matters (write-gating) is
>   preserved in full.
> - Entry point: `python -m src.scoping.orchestrator --objective <id> [--dry-run]`.

**Schemas & config**
- [x] `config/scoping_plan.schema.json` — model plan (families/combos/ensemble topology, per-choice rationale + tier-tagged evidence citations), per-model data requirements (symbols, data types, timeframes, raw-vs-engineered features), KPI sanity block
- [x] `config/proven_journal.schema.json` — T3 entry: model config, live period, realized metrics, `mlflow_run_id`, `user_confirmed_at`
- [x] `config/critic_objection.schema.json` — target_claim, severity, evidence citation, `interface_assumption` tag (cross-review), verification checklist for "no objections"
- [x] `research_library.schema.json`: append-mode accumulation, per-publication notes, chunk/index status flags
- [x] `config/research_director.yaml` scoping block: iteration caps (self-critique ≤3, critic rounds ≤4, synthesis replication =1), convergence rules, provider matrix (author vs critic model family), embedding backend, raised cost caps

**Personas / roles**
- [x] `config/agents/critic.md` — single parameterized critic role: shared adversarial instructions + objection schema + artifact-first→targeted-verification method; composed at runtime with the reviewed field's persona; fresh context; input = deliverable + retrieval trace (never scratchpad); cross-model provider
- [x] Scoping-pass prompt sections for `quant_trading.md` + `ml_data_scientist.md` (Pass A/B duties, self-critique pre-pass, interface-review scope)
- [x] Per-field red-flag checklists as config data (not separate personas)

**RAG substrate**
- [x] Install pgvector (`CREATE EXTENSION vector`) — verified absent today
- [x] `scoping.research_chunks` table (chunk, embedding, tier, traits, source_ref, objective_id) + migration
- [x] `src/scoping/embeddings.py` — backend adapter (local sentence-transformers default; Voyage/OpenAI configurable)
- [x] Indexer: chunk+embed `research_library.json`, `research_cases/*.md`, T3 journal; incremental, rebuildable from artifacts
- [x] Agent tool `search_research(query, tier, traits, k)`

**Evidence tiers**
- [x] T2 read-only adapter tools: `query_results_index`, `query_leaderboard`, `query_episodes_patterns`, `query_mlflow`
- [x] T3 journal: `data/evaluation/proven_journal/<objective_id>.jsonl` + `python -m src.scoping.journal add --from-run <mlflow_run_id>` with **interactive user confirmation**; write-gated
- [x] T3>T2>T1 trust rule encoded in synthesis + judge prompts

**Orchestrator (`src/scoping` v2)**
- [x] 5-stage deterministic state machine (A → B1/B2 → synthesis → C → report), artifacts per stage to `data/scoping/<obj>/<run_ts>/`, all schema-validated
- [x] Per-pass agent loop: PLAN → ACT (tools) → DRAFT → self-critique (≤3) → critic rounds (≤4, converge on no material objections); authors log **retrieval traces** (required critic input)
- [x] Critic verification tools: deterministic citation-exists check, omission probes, counter-evidence search
- [x] Interface-scoped cross-review with schema-enforced assumption tags (out-of-domain objections dropped)
- [x] Synthesis: panel debate → `scoping_plan.json`; judge = Director role; **one** full independent replication reconciled by the judge
- [x] Pass C differ v2: + model catalog/plugin coverage, + feature catalog coverage; criticality-ranked report
- [x] Final report generator: `final_research_report.md` + draft `manual_candidates.yaml` + objective-doc patch (`eligible_model_families`, `search.feature_views`) + aux-dataset specs → human gate
- [x] Wire approved outputs into the search controller inputs (scoping→search handoff)
- [x] Corpus write-back: append critic-surfaced omissions/new findings to T1 + re-index

**Observability & durability**
- [x] MLflow experiment `scoping/<objective_id>`: parent run per session, nested runs per pass/round, artifacts attached
- [x] `oci_sync.conf`: add `data/scoping|vtech_archive|scoping/`

**Governance & tests**
- [x] Tests: state machine (stub LLM), objection-schema validation, RAG round-trip, gap differ v2, journal write-gating, convergence caps, retrieval-trace capture
- [x] ROADMAP entry + README ADR index; W2.5 marked superseded by this ADR
- [x] Update `docs/architecture/RESEARCH_DIRECTOR.md` scoping section on implementation

Cross-links: [ADR-013](ADR-013-two-expert-panel.md), [ADR-021](ADR-021-outcome-episode-pattern-loop.md), [ADR-024](ADR-024-unified-experiment-artifact-flow.md), Phase-20 W2/W2.5/W7 ([PHASE-20-RESEARCH-DIRECTOR-REBUILD.md](../roadmap/PHASE-20-RESEARCH-DIRECTOR-REBUILD.md))
