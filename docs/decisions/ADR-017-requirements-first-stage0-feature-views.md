# ADR-017 — Requirements-First Stage 0 + Curated Per-Family Feature Views

**Date**: 2026-07-03
**Status**: Accepted
**Relates to**: [PHASE-23-MODEL-SELECTION-SEARCH.md](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md)
(W2.5 Stage 0, W3 controller), [ADR-014](./ADR-014-requirement-target-kpi.md) (objective spec),
`research_cases/Agent Summaries/Three_ML_Model_approaches.md` (model-component taxonomy)

---

## Context

Two coupled problems surfaced in Phase 23 testing:

1. **Inventory-anchored scoping.** The Stage-0 Strategy Research Agent's prompt instructed it
   to inventory existing data (`db_inventory`, `dataset_catalog`) **first** and derive the
   Data Requirements Spec from what it found. That anchors every new trade objective's scope
   to whatever previous objectives happened to compute — the opposite of the intended
   methodology, where the case for every data/feature class must be made per objective from
   the strategy's mechanics and the securities' characteristics. It also duplicated work: the
   runner already performs a **deterministic** inventory diff (`_gap_check`) to build the Data
   Gap Report for the human acquire/proceed gate.

2. **One matrix for all families.** Every model family received the full ~175-column parquet
   feature matrix. Distance/density/covariance families (KMeans, DBSCAN, GMM, HMM) degrade or
   fail in that dimensionality — GMM's `covariance_type="full"` was observed failing on the
   full float32 matrix (ill-conditioned covariance → unfitted model → zero-signal output).
   Canonical regime detection feeds these models a *small curated descriptor set* (volatility,
   trend strength, range compression, volume behaviour), not the supervised superset.

## Decision

1. **Requirements-first Stage 0.** The scoping agent holds **no inventory tools**. It derives
   the Data Requirements Spec from first principles (objective + securities + research
   evidence), explicitly instructed not to assume any existing dataset. The inventory
   comparison is exclusively the runner's deterministic `_gap_check`, run **after** the
   artifacts exist. Reviewing existing parquet/DB holdings is the LAST step of the workflow,
   never an input to scoping. (Persona, prompts, and tool registry updated accordingly.)

2. **Curated per-family feature views** (`src/models/feature_views.py`). A view is a named
   list of feature-prefix groups; the controller masks fold `X` arrays per family. Resolution:
   `objective["search"]["feature_views"][family]` → `CatalogEntry.feature_view` → full matrix.
   Initial views: `regime_core` (rv_/ht_/dc_/vw_/cs_ — assigned to hmm/gmm/kmeans/dbscan),
   `sequence_baseline` (standard-indicator set for sequence models — available, not yet the
   default: the LSTM champion baseline stays on the full matrix for comparability),
   `microstructure` (order-flow experiment view). Views select by prefix so they survive
   column additions and parquet-vintage drift; empty selection falls back to the full matrix.

3. **GMM/HMM numerical hardening.** `covariance_type` defaults to `diag` (catalog + plugin),
   `reg_covar=1e-4`, and regime fits cast to float64 — full covariance on wide float32 was the
   crash vector.

4. **Provider-agnostic LLM factory** (`src/common/llm_factory.py`). All agentic processes
   (Stage-0 agent, Model Scout, future agents) construct their LLM via `make_llm()`, selected
   by `VTECH_LLM_PROVIDER` (auto-detect from available creds when unset). Both clients
   (`AnthropicLLM`, `LLMClient`) satisfy one duck-typed contract (`chat` / `chat_json` /
   `chat_tools` / cost-band introspection); shared helpers (`strip_code_fence`) are deduped
   into the factory module. Fitting a different LLM to any agent is a config change.

## Consequences

- New objectives get an unbiased, evidence-cited data scope; reuse of existing parquet
  collections is a deliberate post-gap-check decision, not a default.
- **Regime-family metrics are not comparable** with prior full-matrix regime runs (input
  dimensionality changed ~175 → ~40); leaderboard entries are keyed by run, and no prior
  regime candidate had passed the gate, so nothing promoted is invalidated.
- Supervised families' inputs are unchanged (no comparability impact); `sequence_baseline`
  adoption for DL families is deferred until it can be A/B'd against the full matrix.
- One construction point for LLMs removes per-agent provider lock-in and duplicate client
  bootstrapping; the Research Director's internal `LLMClient()` construction sites can migrate
  to the factory incrementally.
