"""Backward-compatible shim — prefer :mod:`src.common.llm`.

Historically this module lived at ``src.common.llm_factory``. The canonical home is now
``src.common.llm`` (factory + clients + cost). This file re-exports the public API so
existing ``from src.common.llm_factory import make_llm`` call sites keep working.
"""

from __future__ import annotations

from src.common.llm.factory import make_llm, strip_code_fence

__all__ = ["make_llm", "strip_code_fence"]
