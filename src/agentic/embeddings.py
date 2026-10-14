"""Shared embedding backends — re-exports Scoping-v2 embedder.

Used by research RAG and Research Director agent memory so a process resolves
one embed backend (and one model id) consistently.
"""

from __future__ import annotations

from src.scoping.embeddings import Embedder, make_embedder  # noqa: F401

__all__ = ["Embedder", "make_embedder"]
