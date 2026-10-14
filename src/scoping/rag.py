"""Scoping-v2 RAG substrate — pgvector index over the tiered research corpus (ADR-025 §3).

Artifacts stay file/object-storage-durable (``data/scoping``, ``research_cases/``, the T3
journal); this module maintains the *retrieval* layer: ``scoping.research_chunks`` in the
existing Postgres (inherits pgBackRest durability), embedded with any backend from
:mod:`src.scoping.embeddings`. The index is REBUILDABLE from artifacts at any time —
losing it loses nothing.

Agent-facing surface (one tool + one deterministic check):

    search_research(query, tier=None, traits=None, k=8)  → top-k chunks (agentic RAG)
    citation_exists(source_ref)                          → deterministic citation grounding

Vectors from different embedding backends are never mixed: rows carry ``embed_model`` and
every query filters on the active one. Exact cosine scan (no ANN index) — the corpus is
hundreds-to-thousands of chunks; add ivfflat only if that changes.

CLI::

    python -m src.scoping.rag ensure                     # create schema/table
    python -m src.scoping.rag index [--objective ID]     # (re)index the corpus
    python -m src.scoping.rag search "query" [--tier t1] [--k 8]
    python -m src.scoping.rag status
"""

from __future__ import annotations

from src.common.data_paths import data_root, evaluation_dir, scoping_dir

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.scoping.rag")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DDL = """
CREATE SCHEMA IF NOT EXISTS scoping;
CREATE TABLE IF NOT EXISTS scoping.research_chunks (
    id             bigserial PRIMARY KEY,
    source_ref     text        NOT NULL,          -- publication id / file path / journal ref
    tier           text        NOT NULL CHECK (tier IN ('t1','t2','t3')),
    objective_id   text,
    traits         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    chunk_ix       int         NOT NULL DEFAULT 0,
    content        text        NOT NULL,
    content_sha256 text        NOT NULL,
    embed_model    text        NOT NULL,
    embedding      vector      NOT NULL,           -- dim varies by backend; queries filter embed_model
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (content_sha256, embed_model)
);
CREATE INDEX IF NOT EXISTS research_chunks_source_ref_idx ON scoping.research_chunks (source_ref);
CREATE INDEX IF NOT EXISTS research_chunks_tier_idx       ON scoping.research_chunks (tier, embed_model);
"""


def _connect():
    import psycopg
    from src.common.db import get_dsn
    return psycopg.connect(get_dsn())


def ensure_schema() -> None:
    """Create the scoping schema + chunks table (idempotent). Extension install is ops-level."""
    with _connect() as conn:
        conn.execute(_DDL)
        conn.commit()


# ── chunking ──────────────────────────────────────────────────────────────────

def _cfg_rag() -> dict[str, Any]:
    """The scoping ``rag`` knobs (chunk_chars / chunk_overlap / search_k) — YAML-authoritative."""
    try:
        from src.scoping.orchestrator import load_scoping_cfg
        return load_scoping_cfg().get("rag") or {}
    except Exception:  # noqa: BLE001
        return {}


_EMBEDDER_CACHE: Any = None  # module-level singleton — resolved once per process


def _default_embedder() -> Any:
    """Return the process-level embedder singleton (backend resolved once, then cached)."""
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is None:
        from src.scoping.embeddings import make_embedder
        _EMBEDDER_CACHE = make_embedder()
    return _EMBEDDER_CACHE


def chunk_text(text: str, *, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
    """Paragraph-aware chunker: pack whole paragraphs up to ``max_chars``; hard-split
    oversized paragraphs with ``overlap`` carry-over. Deterministic, whitespace-normalized.
    Defaults come from config/scoping.yaml ``rag`` block (1200/150 when unset)."""
    if max_chars is None or overlap is None:
        cfg = _cfg_rag()
        max_chars = int(max_chars or cfg.get("chunk_chars", 1200))
        overlap = int(overlap if overlap is not None else cfg.get("chunk_overlap", 150))
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        while len(p) > max_chars:                      # oversized paragraph → hard split
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(p[:max_chars])
            p = p[max_chars - overlap:]
        if buf and len(buf) + 2 + len(p) > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


# ── indexing ──────────────────────────────────────────────────────────────────

def index_document(
    source_ref: str,
    text: str,
    *,
    tier: str,
    objective_id: str | None = None,
    traits: dict[str, Any] | None = None,
    embedder: Any = None,
) -> int:
    """Chunk + embed + upsert one document. Returns the number of NEW chunks inserted
    (existing (sha, embed_model) pairs are skipped — incremental by construction)."""
    emb = embedder or _default_embedder()
    chunks = chunk_text(text)
    if not chunks:
        return 0
    shas = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content_sha256 FROM scoping.research_chunks"
                " WHERE embed_model = %s AND content_sha256 = ANY(%s)",
                (emb.model_id, shas))
            existing = {r[0] for r in cur.fetchall()}
        new = [(i, c, s) for i, (c, s) in enumerate(zip(chunks, shas)) if s not in existing]
        if not new:
            return 0
        vecs = emb.embed([c for _, c, _ in new])
        with conn.cursor() as cur:
            for (ix, content, sha), vec in zip(new, vecs):
                cur.execute(
                    "INSERT INTO scoping.research_chunks"
                    " (source_ref, tier, objective_id, traits, chunk_ix, content,"
                    "  content_sha256, embed_model, embedding)"
                    " VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::vector)"
                    " ON CONFLICT (content_sha256, embed_model) DO NOTHING",
                    (source_ref, tier.lower(), objective_id,
                     json.dumps(traits or {}), ix, content, sha,
                     emb.model_id, _vec_literal(vec)))
        conn.commit()
    logger.info("rag_indexed source=%s tier=%s new_chunks=%d", source_ref, tier, len(new))
    return len(new)


def index_corpus(objective_id: str | None = None, *, embedder: Any = None,
                 data_root: Path | None = None) -> dict[str, int]:
    """(Re)index the tiered corpus incrementally. Sources:

    T1 — research_cases/*.md + research_cases/papers/*, every cumulative/pass
         research_library.json under data/research/scoping/ (publication → one doc)
    T3 — the proven-live journal entries (data/evaluation/proven_journal/*.jsonl)

    T2 is NOT indexed: it is structured/queryable directly via src.scoping.evidence.
    """
    emb = embedder or _default_embedder()
    root = data_root or _data_root()
    stats = {"t1_files": 0, "t1_pubs": 0, "t3_entries": 0, "new_chunks": 0}

    # T1: internal research corpus (markdown)
    cases = _REPO_ROOT / "research_cases"
    for md in sorted(list(cases.glob("*.md")) + list((cases / "papers").glob("**/*.md"))):
        try:
            n = index_document(str(md.relative_to(_REPO_ROOT)), md.read_text(errors="ignore"),
                               tier="t1", embedder=emb)
            stats["t1_files"] += 1
            stats["new_chunks"] += n
        except Exception as exc:  # noqa: BLE001 — one bad file never kills the index run
            logger.warning("rag_index_skip file=%s err=%s", md, exc)

    # T1: scored research libraries (accumulated + per-run)
    scoping_dir = root 
    if scoping_dir.exists():
        for lib in sorted(scoping_dir.glob("**/research_library.json")):
            if objective_id and objective_id not in lib.parts:
                continue
            try:
                data = json.loads(lib.read_text())
                oid = data.get("objective_id")
                for pub in data.get("publications") or []:
                    text = _pub_text(pub)
                    if not text:
                        continue
                    n = index_document(pub.get("id") or pub.get("url") or lib.name, text,
                                       tier="t1", objective_id=oid,
                                       traits=data.get("traits"), embedder=emb)
                    stats["t1_pubs"] += 1
                    stats["new_chunks"] += n
            except Exception as exc:  # noqa: BLE001
                logger.warning("rag_index_skip lib=%s err=%s", lib, exc)

    # T3: proven-live journal
    jdir = root / "evaluation" / "proven_journal"
    if jdir.exists():
        for jf in sorted(jdir.glob("*.jsonl")):
            oid = jf.stem
            if objective_id and oid != objective_id:
                continue
            for i, line in enumerate(jf.read_text().splitlines()):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    n = index_document(f"proven_journal:{oid}#{i}", _journal_text(entry),
                                       tier="t3", objective_id=oid, embedder=emb)
                    stats["t3_entries"] += 1
                    stats["new_chunks"] += n
                except Exception as exc:  # noqa: BLE001
                    logger.warning("rag_index_skip journal=%s line=%d err=%s", jf, i, exc)

    logger.info("rag_index_corpus %s", stats)
    return stats


# ── retrieval ─────────────────────────────────────────────────────────────────

def search_research(
    query: str,
    *,
    tier: str | None = None,
    traits: dict[str, Any] | None = None,
    k: int | None = None,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """THE agent retrieval tool (agentic RAG): top-k chunks by cosine similarity, optionally
    filtered by tier ('t1'|'t2'|'t3') and/or a traits subset match. ``k`` defaults from
    config/scoping.yaml. When the ACTIVE embed_model has no indexed rows (e.g. the embedding
    endpoint just changed), returns a reindex hint instead of silently empty results."""
    emb = embedder or _default_embedder()
    if k is None:
        k = int(_cfg_rag().get("search_k", 8))
    qv = _vec_literal(emb.embed([query])[0])

    sql = ("SELECT source_ref, tier, objective_id, content,"
           "       1 - (embedding <=> %s::vector) AS similarity"
           "  FROM scoping.research_chunks WHERE embed_model = %s")
    params: list[Any] = [qv, emb.model_id]
    if tier:
        sql += " AND tier = %s"
        params.append(tier.lower())
    if traits:
        sql += " AND traits @> %s::jsonb"
        params.append(json.dumps(traits))
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([qv, int(k)])

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            # Distinguish "no match" from "index not built for THIS embed_model" (the
            # silent-degradation trap when the embedding endpoint/model changes).
            cur.execute("SELECT count(*) FILTER (WHERE embed_model = %s), count(*)"
                        " FROM scoping.research_chunks", (emb.model_id,))
            n_model, n_total = cur.fetchone()
            if int(n_model) == 0 and int(n_total) > 0:
                logger.warning("rag_index_empty_for_model embed_model=%s total_chunks=%s — "
                               "run `python -m src.scoping.rag index`", emb.model_id, n_total)
                return [{"error": f"no chunks indexed under embed_model '{emb.model_id}' "
                                  f"({n_total} exist under other models) — the corpus needs "
                                  "re-indexing: python -m src.scoping.rag index"}]
    return [{"source_ref": r[0], "tier": r[1], "objective_id": r[2],
             "content": r[3], "similarity": round(float(r[4]), 4)} for r in rows]


def citation_exists(source_ref: str) -> dict[str, Any]:
    """Deterministic citation-grounding check for the critic (ADR-025 §4.3): does the cited
    source exist in the RAG index or the cumulative research library? No LLM involved."""
    n = 0
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM scoping.research_chunks WHERE source_ref = %s",
                        (source_ref,))
            n = int(cur.fetchone()[0])
    except Exception as exc:  # noqa: BLE001 — index down ≠ citation missing; report distinctly
        return {"source_ref": source_ref, "exists": False, "chunks": 0, "index_error": str(exc)}
    in_lib = False
    lib = scoping_dir() / "_library" / "research_library.json"
    if lib.exists():
        try:
            pubs = json.loads(lib.read_text()).get("publications") or []
            in_lib = any(p.get("id") == source_ref or p.get("url") == source_ref for p in pubs)
        except Exception:  # noqa: BLE001
            pass
    return {"source_ref": source_ref, "exists": n > 0 or in_lib, "chunks": n, "in_library": in_lib}


def status() -> dict[str, Any]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT tier, embed_model, count(*) FROM scoping.research_chunks"
                    " GROUP BY tier, embed_model ORDER BY tier")
        rows = cur.fetchall()
    return {"chunks_by_tier_model": [{"tier": r[0], "embed_model": r[1], "n": int(r[2])}
                                     for r in rows]}


# ── helpers ───────────────────────────────────────────────────────────────────

def _data_root() -> Path:
    return data_root()


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def _pub_text(pub: dict[str, Any]) -> str:
    parts = [pub.get("title", ""), pub.get("summary", ""), pub.get("notes", "")]
    ideas = pub.get("data_feature_ideas") or []
    if ideas:
        parts.append("Data/feature ideas: " + "; ".join(ideas))
    return "\n\n".join(p for p in parts if p).strip()


def _journal_text(entry: dict[str, Any]) -> str:
    model = entry.get("model") or {}
    lp = entry.get("live_period") or {}
    return (f"PROVEN LIVE RESULT — objective {entry.get('objective_id')}: "
            f"{model.get('family')} ({model.get('candidate_id') or 'n/a'}) "
            f"live {lp.get('start')}..{lp.get('end')} realized "
            f"{json.dumps(entry.get('realized_metrics') or {})}. "
            f"MLflow run {entry.get('mlflow_run_id')}. {entry.get('notes', '')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Scoping v2 RAG index (ADR-025)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure")
    ix = sub.add_parser("index")
    ix.add_argument("--objective", default=None)
    se = sub.add_parser("search")
    se.add_argument("query")
    se.add_argument("--tier", default=None)
    se.add_argument("--k", type=int, default=8)
    sub.add_parser("status")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.cmd == "ensure":
        ensure_schema()
        print("scoping.research_chunks ready")
    elif args.cmd == "index":
        ensure_schema()
        print(json.dumps(index_corpus(args.objective), indent=2))
    elif args.cmd == "search":
        for hit in search_research(args.query, tier=args.tier, k=args.k):
            print(f"[{hit['tier']}] {hit['similarity']:.3f}  {hit['source_ref']}\n"
                  f"    {hit['content'][:180].replace(chr(10), ' ')}…")
    elif args.cmd == "status":
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    _cli()
