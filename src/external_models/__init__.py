"""External (non-catalog) models — Stage-0 rule replicas with operator knobs.

This plane is intentionally separate from ``src.models`` (ML plugin catalog)
and ``src.backtest`` (feature-parquet rule DSL). External models read
PostgreSQL candles (``databento.tbbo_1min``), not the 5s feature matrix, and
they are not trained. See ADR-058 / ADR-096.
"""

from __future__ import annotations

from src.external_models.catalog import MODEL_ID_RE, get_model, list_models

__all__ = ["MODEL_ID_RE", "get_model", "list_models"]
