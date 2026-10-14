"""Shared research retrieval for all agents.

Primary path: Scoping-v2 **pgvector** corpus (``search_research``) — T1 literature
and T3 proven-live journal, tier-filterable.

Fallback: lightweight internal substring scan over ``research_cases/`` + ``docs/decisions/``
when the vector index is empty or unavailable (never raises to the agent loop).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.agentic.rag")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def search_research(
    query: str,
    *,
    tier: str | None = None,
    k: int = 8,
) -> list[dict[str, Any]]:
    """Semantic search over the tiered research corpus (pgvector).

    Thin wrapper so agents import from :mod:`src.agentic.rag` instead of
    reaching into ``src.scoping`` implementation details.
    """
    from src.scoping.rag import search_research as _sr  # noqa: PLC0415
    return _sr(query, tier=tier, k=k)


def citation_exists(source_ref: str) -> bool:
    from src.scoping.rag import citation_exists as _ce  # noqa: PLC0415
    return _ce(source_ref)


def research_search(
    query: str,
    *,
    tier: str | None = None,
    k: int = 8,
    allow_substring_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Primary agent research tool: RAG first, optional substring fallback.

    Returns a list of hit dicts. Shapes may mix RAG hits
    (``source_ref``, ``snippet``, ``tier``, …) and fallback hits
    (``source``, ``snippet``). Agents should treat both as citation candidates.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        hits = search_research(q, tier=tier, k=int(k))
        if hits:
            return hits if isinstance(hits, list) else [hits]
    except Exception as exc:  # noqa: BLE001 — agent tool must never hard-fail
        logger.debug("agentic_rag_unavailable: %s", exc)

    if not allow_substring_fallback:
        return []
    return _substring_internal(q, limit=min(5, int(k)))


def _substring_internal(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    """Legacy internal corpus scan (pre-pgvector). Keep as cold fallback only."""
    hits: list[dict[str, str]] = []
    terms = [t.lower() for t in query.split() if len(t) > 3]
    if not terms:
        return []
    for d in ("research_cases/papers", "docs/decisions", "docs/architecture"):
        root = _REPO_ROOT / d
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".txt"}:
                continue
            try:
                text = path.read_text(errors="ignore").lower()
            except OSError:
                continue
            if any(term in text for term in terms):
                hits.append({
                    "source": str(path.relative_to(_REPO_ROOT)),
                    "source_ref": str(path.relative_to(_REPO_ROOT)),
                    "snippet": query,
                    "tier": "t1",
                    "via": "substring_fallback",
                })
            if len(hits) >= limit:
                return hits
    return hits
