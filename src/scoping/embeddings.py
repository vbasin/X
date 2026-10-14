"""Embedding backend adapter for the Scoping-v2 RAG substrate (ADR-025 §3).

One tiny interface — ``Embedder.embed(texts) -> list[vector]`` — over three backends:

    local   sentence-transformers on CPU (default when importable)
    openai  any OpenAI-compatible embeddings endpoint (RD_LLM_BASE_URL / RD_LLM_API_KEY /
            RD_EMBED_MODEL — Anthropic has no embeddings endpoint, so the RD client's
            OpenAI-compatible path covers the API case)
    hash    deterministic token-hashing vectors — zero dependencies, zero network; used by
            tests and as the degraded-mode fallback so indexing never hard-fails

Backend selection: explicit arg > ``VTECH_EMBED_BACKEND`` env > ``auto`` (local → openai →
hash). Vectors from different backends are never compared: every chunk row stores its
``embed_model`` id and queries filter on it (see src.scoping.rag).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any

logger = logging.getLogger("vtech.scoping.embeddings")

_DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_HASH_DIM = 384


class Embedder:
    """Lazy, backend-agnostic embedding client. ``model_id`` keys rows in the RAG table."""

    def __init__(self, backend: str | None = None, *, local_model: str | None = None,
                 hash_dim: int | None = None) -> None:
        # Precedence: VTECH_EMBED_BACKEND env (ad-hoc override) > explicit/YAML arg > auto.
        self.backend = (os.environ.get("VTECH_EMBED_BACKEND") or backend or "auto").strip().lower()
        self._local_model_name = local_model or _DEFAULT_LOCAL_MODEL
        self._hash_dim = int(hash_dim or _DEFAULT_HASH_DIM)
        self._impl: Any = None          # resolved on first embed()
        self._resolved: str = ""

    # ── public API ────────────────────────────────────────────────────────────
    @property
    def model_id(self) -> str:
        """Stable id of the resolved backend+model, e.g. 'local:all-MiniLM-L6-v2'."""
        self._resolve()
        return self._resolved

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; every vector is L2-normalized."""
        self._resolve()
        if not texts:
            return []
        if self._resolved.startswith("local:"):
            vecs = self._impl.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [list(map(float, v)) for v in vecs]
        if self._resolved.startswith("openai:"):
            model = self._resolved.split(":", 1)[1]
            resp = self._impl.embeddings.create(model=model, input=texts)
            return [_l2([float(x) for x in d.embedding]) for d in resp.data]
        return [_hash_embed(t, self._hash_dim) for t in texts]

    # ── backend resolution ─────────────────────────────────────────────────────
    def _resolve(self) -> None:
        if self._resolved:
            return
        order = [self.backend] if self.backend != "auto" else ["local", "openai", "hash"]
        for b in order:
            try:
                if b == "local":
                    from sentence_transformers import SentenceTransformer
                    self._impl = SentenceTransformer(self._local_model_name)
                    self._resolved = f"local:{self._local_model_name}"
                    return
                if b == "openai":
                    # Endpoint preference: the OpenAI PLATFORM when OPENAI_API_KEY exists
                    # (no request quotas; used by the Scoping-v2 critic endpoint), else the
                    # shared RD_LLM_* endpoint (e.g. GitHub Models — 150 embed requests/day).
                    import openai
                    if os.environ.get("OPENAI_API_KEY"):
                        base = os.environ.get("OPENAI_BASE_URL")            # SDK default otherwise
                        key = os.environ["OPENAI_API_KEY"]
                        model = os.environ.get("VTECH_EMBED_MODEL", "text-embedding-3-large")
                    else:
                        base = os.environ.get("RD_LLM_BASE_URL")
                        key = os.environ.get("RD_LLM_API_KEY", "")
                        model = os.environ.get("RD_EMBED_MODEL", "text-embedding-3-large")
                        if not key:
                            raise ValueError("no OPENAI_API_KEY/RD_LLM_API_KEY for openai embeddings")
                    self._impl = openai.OpenAI(base_url=base, api_key=key) if base else openai.OpenAI(api_key=key)
                    self._resolved = f"openai:{model}"
                    return
                if b == "hash":
                    self._impl = None
                    self._resolved = f"hash:{self._hash_dim}"
                    return
            except Exception as exc:  # noqa: BLE001 — fall through the auto order
                if self.backend != "auto":
                    raise
                # DEBUG, not INFO: in 'auto' mode a missing local backend is expected and the
                # fall-through is silent-by-design — logging it per resolution spammed the run
                # log (the embedder is a process singleton, so this now fires at most once).
                logger.debug("embed backend '%s' unavailable (%s) — trying next", b, exc)
        raise ValueError(f"no embedding backend resolvable (requested '{self.backend}')")


def make_embedder(cfg: dict[str, Any] | None = None) -> Embedder:
    """Build an Embedder from the scoping config ``embedding`` block (config/scoping.yaml).

    This is the LIVE construction path (rag.py routes through it), so the YAML knobs
    (backend / local_model / hash_dim) are authoritative; the ``VTECH_EMBED_BACKEND`` env
    var still overrides for ad-hoc runs."""
    if cfg is None:
        try:
            from src.scoping.orchestrator import load_scoping_cfg
            cfg = load_scoping_cfg()
        except Exception:  # noqa: BLE001 — config unreadable → pure-default embedder
            cfg = {}
    e = (cfg or {}).get("embedding") or {}
    return Embedder(e.get("backend"), local_model=e.get("local_model"), hash_dim=e.get("hash_dim"))


# ── deterministic hashing backend ──────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_embed(text: str, dim: int) -> list[float]:
    """Feature-hashing bag-of-words vector: deterministic, dependency-free, L2-normalized."""
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:8], "big")
        vec[h % dim] += -1.0 if (h >> 63) & 1 else 1.0
    return _l2(vec)


def _l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 0 else vec
