"""Research Director tool registries — builds on the shared agentic registry.

Core :class:`ToolRegistry` lives in :mod:`src.agentic.registry`. This module wires
RD-specific context tools and the unified research retrieval path
(:func:`src.agentic.rag.research_search` — pgvector first, substring fallback).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable

from src.agentic.rag import research_search as _research_search
from src.agentic.registry import Tool, ToolRegistry, obj_schema
from src.research_director import tools

# Re-export for callers that imported these from research_director.registry
_obj = obj_schema
__all__ = [
    "Tool",
    "ToolRegistry",
    "build_critic_registry",
    "build_default_registry",
    "_obj",
    "_scoping_rag_search",
]


def build_default_registry(
    *,
    registry_dir: str | Path,
    program_id: str,
    model_name: str,
    external_research: bool = False,
) -> ToolRegistry:
    """Wire context / research / skill tools for one Research Director cycle.

    ``research_search`` uses the shared agentic RAG (pgvector) with substring
    fallback — agents no longer depend on the legacy internal-only grep path as primary.
    """
    from src.research_director import skills as _skills  # noqa: PLC0415

    reg = ToolRegistry()
    reg.add(
        "db_inventory",
        "List database tables with approximate row counts and column counts for a schema "
        "(default 'databento'). Use to learn what raw data already exists.",
        _obj({"schema": {"type": "string", "default": "databento"}}),
        tools.db_inventory,
    )
    reg.add(
        "data_capability_catalog",
        "Pre-built data capability catalog (data/ops/catalog/data_capability.json): for each "
        "internal dataset — instruments, data_type, granularity, date_range, and fields. Also "
        "covers the parquet feature matrix. Use before proposing data-dependent experiments. "
        "Returns {} if the catalog has not been built (run scripts/features/build_data_capability.py).",
        _obj(),
        tools.data_capability_catalog,
    )
    reg.add(
        "dataset_catalog",
        "Parquet/snapshot inventory for the program: latest snapshot pointer + CATALOG.json "
        "(date coverage, row counts, columns). No arguments.",
        _obj(),
        partial(tools.dataset_catalog, program_id),
    )
    reg.add(
        "read_attribution",
        "Feature attribution for the current baseline model: per-group importance and the "
        "top integrated-gradients features for BIG_UP / BIG_DOWN. No arguments.",
        _obj(),
        partial(tools.read_attribution, Path(registry_dir), program_id, model_name),
    )
    reg.add(
        "read_reference",
        "Read one repo doc by relative path (e.g. docs/architecture/ML_MODEL.md, an ADR, "
        "docs/reference/FEATURE_CATALOG.md). Returns truncated text.",
        _obj({"rel_path": {"type": "string"}}, ["rel_path"]),
        tools.read_reference,
    )
    reg.add(
        "load_docs",
        "Load several repo docs at once given a list of relative paths.",
        _obj({"rel_paths": {"type": "array", "items": {"type": "string"}}}, ["rel_paths"]),
        tools.load_docs,
    )
    reg.add(
        "recent_journal",
        "Recent experiments and their outcomes (incl. DISPROVEN) so you never re-propose a "
        "dead idea. No arguments.",
        _obj(),
        partial(tools.load_recent_journal, program_id),
    )
    reg.add(
        "research_search",
        "Search the tiered research corpus (pgvector T1 literature + T3 proven-live; falls "
        "back to internal docs/papers scan if the index is cold). Read-only — never downloads data. "
        "Optional tier ('t1'|'t3') and k.",
        _obj({
            "query": {"type": "string"},
            "tier": {"type": "string"},
            "k": {"type": "integer", "default": 8},
        }, ["query"]),
        lambda query, tier=None, k=8: _research_search(
            query, tier=tier, k=int(k), allow_substring_fallback=True),
    )
    # Alias used by Scoping-style prompts / critic
    reg.add(
        "search_research",
        "Alias of research_search — semantic search over the tiered research corpus.",
        _obj({
            "query": {"type": "string"},
            "tier": {"type": "string"},
            "k": {"type": "integer", "default": 8},
        }, ["query"]),
        lambda query, tier=None, k=8: _research_search(
            query, tier=tier, k=int(k), allow_substring_fallback=True),
    )
    reg.add(
        "get_skill",
        "Fetch a named reusable procedure (skill): one of "
        "'design_walkforward', 'feature_group_ablation', 'audit_calibration', 'estimate_cost'.",
        _obj({"name": {"type": "string"}}, ["name"]),
        _skills.get_skill,
    )
    # external_research reserved for future web path; RAG covers internal+indexed external T1
    _ = external_research
    return reg


def _scoping_rag_search(query: str, tier: str | None = None, k: int = 6) -> Any:
    """Bridge critic tools to pgvector (no substring fallback — verification wants corpus hits)."""
    try:
        return _research_search(
            query, tier=tier, k=int(k), allow_substring_fallback=False)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"research RAG unavailable: {exc}"}


def build_critic_registry(
    *,
    registry_dir: str | Path,
    program_id: str,
    model_name: str,
) -> ToolRegistry:
    """Read-only verification tools for the cross-model critic reviewing a prescription.

    Mirrors the scoping critic registry: journal, data existence, attribution, RAG grounding.
    """
    reg = ToolRegistry()
    reg.add(
        "recent_journal",
        "Recent experiments and their outcomes (incl. DISPROVEN). Use to check whether the "
        "prescription re-proposes a dead idea. No arguments.",
        _obj(),
        partial(tools.load_recent_journal, program_id),
    )
    reg.add(
        "dataset_catalog",
        "Parquet/snapshot inventory for the program. Use to verify the data / history the "
        "prescription assumes actually exists. No arguments.",
        _obj(),
        partial(tools.dataset_catalog, program_id),
    )
    reg.add(
        "read_attribution",
        "Feature attribution for the current baseline model — verify feature claims against "
        "measured importance. No arguments.",
        _obj(),
        partial(tools.read_attribution, Path(registry_dir), program_id, model_name),
    )
    reg.add(
        "research_search",
        "Search the tiered research corpus for grounding or counter-evidence.",
        _obj({
            "query": {"type": "string"},
            "tier": {"type": "string"},
            "k": {"type": "integer", "default": 6},
        }, ["query"]),
        lambda query, tier=None, k=6: _research_search(
            query, tier=tier, k=int(k), allow_substring_fallback=True),
    )
    reg.add(
        "search_research",
        "Search the research RAG corpus (T1 literature + proven-live T3). Optional tier.",
        _obj({
            "query": {"type": "string"},
            "tier": {"type": "string"},
            "k": {"type": "integer", "default": 6},
        }, ["query"]),
        _scoping_rag_search,
    )
    return reg
