"""User-authored chart-scoped trading rules — the Live Chart "Model design" section (ADR-117).

A rule spec is a JSON document (see :mod:`.schema`) that references only the
features currently rendered on the Live Chart: candles, active Volume
Profiles, active overlays, and Large Prints. :func:`simulate` runs the rule
tree bar-by-bar against the exact payload the chart displayed, and returns
markers (for the chart), a trade blotter, and a PnL summary.

The DSL is a **closed vocabulary**: unknown identifiers, unbalanced parens,
or unsupported nodes fail at compile time — never at run time.
"""

from __future__ import annotations

from src.webui.chart_models.interpreter import simulate
from src.webui.chart_models.schema import compile_model, validate_model
from src.webui.chart_models.store import (
    delete_model,
    list_models,
    load_model,
    save_model,
)
from src.webui.chart_models.vars import build_var_catalog

__all__ = [
    "simulate",
    "compile_model",
    "validate_model",
    "build_var_catalog",
    "list_models",
    "load_model",
    "save_model",
    "delete_model",
]
