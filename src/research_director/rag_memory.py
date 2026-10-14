"""pgvector substrate for the Research Director's per-agent semantic memory.

Replaces the legacy OCI-blob + in-memory numpy scan (``memory.py`` pre-modernization) with a
Postgres/pgvector table — the same retrieval technology the Scoping-v2 loop uses
(:mod:`src.scoping.rag`), reusing its :class:`~src.scoping.embeddings.Embedder` singleton and
cosine-query pattern.

Why a **dedicated** table (``research_director.agent_memory``) instead of reusing
``scoping.research_chunks``:
  * The scoping table is explicitly *rebuildable from artifacts* — losing it loses nothing. RD
    agent memory (findings / experiment outcomes) is **durable primary state**; it exists nowhere
    else, so co-locating it would break that invariant and make a scoping reindex unsafe.
  * The scoping ``tier`` CHECK is ``('t1','t2','t3')`` and its rows are exposed by the agent-facing
    ``search_research`` tool — RD memory must not leak into that surface.
  * RD needs ``agent_id`` / ``kind`` / ``experiment_id`` as first-class, indexed, always-filtered
    columns; ``agent_id`` is the mandatory recall partition (one lens never reads another's memory).

Vectors are ``embed_model``-scoped exactly as in :mod:`src.scoping.rag`: every row records the
resolved backend id and every query filters on the active one, so vectors from different embedding
backends are never compared (a cold model simply yields empty recall — memory starts cold, which is
not an error).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("vtech.research_director.rag_memory")

_DDL = """
CREATE SCHEMA IF NOT EXISTS research_director;
CREATE TABLE IF NOT EXISTS research_director.agent_memory (
    id             bigserial   PRIMARY KEY,
    agent_id       text        NOT NULL,           -- the lens (quant_trading | ml_data_scientist)
    kind           text        NOT NULL,           -- finding | conclusion | experiment_outcome
    experiment_id  bigint,
    content        text        NOT NULL,
    content_sha256 text        NOT NULL,
    metadata       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    embed_model    text        NOT NULL,           -- dim varies by backend; queries filter on it
    embedding      vector      NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (agent_id, content_sha256, embed_model)
);
CREATE INDEX IF NOT EXISTS agent_memory_agent_idx
    ON research_director.agent_memory (agent_id, embed_model);
"""


def _connect():
    import psycopg  # noqa: PLC0415
    from src.common.db import get_dsn  # noqa: PLC0415
    return psycopg.connect(get_dsn())


def ensure_schema() -> None:
    """Create the research_director schema + agent_memory table (idempotent).

    Extension install (``CREATE EXTENSION vector``) is ops-level, shared with the scoping RAG.
    """
    with _connect() as conn:
        conn.execute(_DDL)
        conn.commit()


def default_embedder() -> Any:
    """Return the process-level embedder singleton, shared with the scoping RAG so a process
    resolves its backend exactly once."""
    # Prefer agentic re-export surface; implementation remains scoping.embeddings.
    from src.scoping.rag import _default_embedder  # noqa: PLC0415
    return _default_embedder()


def _vec_literal(vec: list[float]) -> str:
    from src.scoping.rag import _vec_literal as _vl  # noqa: PLC0415
    return _vl(vec)


def write_memory(
    agent_id: str,
    content: str,
    kind: str,
    *,
    experiment_id: int | None = None,
    metadata: dict[str, Any] | None = None,
    embedder: Any = None,
) -> bool:
    """Embed + upsert one memory item. Returns True if a new row was inserted (idempotent: an
    identical (agent_id, content, embed_model) is skipped via ON CONFLICT)."""
    content = (content or "").strip()
    if not content:
        return False
    emb = embedder or default_embedder()
    sha = hashlib.sha256(content.encode()).hexdigest()
    vec = emb.embed([content])[0]
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO research_director.agent_memory"
                " (agent_id, kind, experiment_id, content, content_sha256, metadata,"
                "  embed_model, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)"
                " ON CONFLICT (agent_id, content_sha256, embed_model) DO NOTHING",
                (agent_id, kind, experiment_id, content, sha,
                 json.dumps(metadata or {}), emb.model_id, _vec_literal(vec)))
            inserted = cur.rowcount > 0
        conn.commit()
    if inserted:
        logger.debug("rd_memory_write agent=%s kind=%s eid=%s", agent_id, kind, experiment_id)
    return inserted


def recall_memory(
    agent_id: str,
    query: str,
    *,
    k: int = 6,
    min_similarity: float = 0.7,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """Top-k cosine recall of ONE agent's own prior memory (agent_id partition, embed_model-scoped).

    Returns dicts ``{content, kind, similarity, metadata}`` with similarity >= ``min_similarity``,
    ranked most-similar first. Empty on a cold model / empty partition / index unavailable — recall
    is best-effort and must never block a research cycle.
    """
    emb = embedder or default_embedder()
    try:
        qv = _vec_literal(emb.embed([query])[0])
        sql = ("SELECT content, kind, metadata,"
               "       1 - (embedding <=> %s::vector) AS similarity"
               "  FROM research_director.agent_memory"
               " WHERE agent_id = %s AND embed_model = %s"
               " ORDER BY embedding <=> %s::vector LIMIT %s")
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(sql, [qv, agent_id, emb.model_id, qv, int(k)])
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 — index down / no backend ≠ recall failure
        logger.warning("rd_memory_recall_failed agent=%s: %s", agent_id, exc)
        return []
    hits: list[dict[str, Any]] = []
    for content, kind, metadata, sim in rows:
        if float(sim) < min_similarity:
            break
        hits.append({"content": content, "kind": kind,
                     "similarity": round(float(sim), 4), "metadata": metadata or {}})
    return hits
