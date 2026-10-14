"""Per-agent semantic memory for the Research Director panel.

Each expert (lens) has its own durable store of embedded findings, conclusions, and experiment
outcomes so the panel learns across cycles instead of restarting cold.

**Substrate: Postgres/pgvector** (``research_director.agent_memory`` via :mod:`.rag_memory`) — the
same retrieval technology the Scoping-v2 loop uses, sharing its
:class:`~src.scoping.embeddings.Embedder`. This replaces the legacy OCI Object-Storage JSON blob +
in-memory numpy cosine scan: pgvector does the top-k cosine in the database (indexed, embed_model-
scoped) rather than downloading the whole blob and scanning it in Python each recall.

Consumed by :mod:`src.research_director.orchestrator` (per-lens recall in the diagnose stage; a
write-back of each lens's prescription + validation verdict at handoff) for cross-cycle learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.research_director import rag_memory


@dataclass(frozen=True)
class MemoryHit:
    content: str
    kind: str
    similarity: float
    metadata: dict[str, Any]


class AgentMemory:
    """pgvector-backed semantic memory for a single agent (lens).

    ``llm`` is retained for backward-compatible construction (the legacy blob store embedded via
    ``LLMClient.embed``); embeddings now go through the shared scoping :class:`Embedder`, so the
    argument is ignored. ``embedder`` may be injected in tests.
    """

    def __init__(self, agent_id: str, llm: Any = None, *, recall_top_k: int = 6,
                 recall_min_similarity: float = 0.7, embedder: Any = None) -> None:
        self._agent_id = agent_id
        self._k = recall_top_k
        self._min_sim = recall_min_similarity
        self._embedder = embedder  # None → rag_memory resolves the shared singleton lazily

    # ── public API ──────────────────────────────────────────────────────────────

    def write(self, content: str, kind: str, *, experiment_id: int | None = None,
              metadata: dict[str, Any] | None = None) -> None:
        """Embed and persist a memory item (finding|conclusion|experiment_outcome)."""
        rag_memory.write_memory(
            self._agent_id, content, kind,
            experiment_id=experiment_id, metadata=metadata, embedder=self._embedder,
        )

    def recall(self, query: str) -> list[MemoryHit]:
        """Top-k cosine-similarity recall of this agent's own prior memory (embed_model-scoped)."""
        hits = rag_memory.recall_memory(
            self._agent_id, query,
            k=self._k, min_similarity=self._min_sim, embedder=self._embedder,
        )
        return [MemoryHit(content=h["content"], kind=h["kind"],
                          similarity=h["similarity"], metadata=h["metadata"]) for h in hits]

    def recall_text(self, query: str) -> str:
        """Recall formatted for prompt injection."""
        hits = self.recall(query)
        if not hits:
            return "(no relevant prior memory)"
        return "\n".join(
            f"- [{h.kind} | sim={h.similarity:.2f}] {h.content}" for h in hits
        )
