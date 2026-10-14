"""Live paper trading — same as live trading without real money (P2).

**Not** a bar-close path-backtest. Execution is driven by **Databento live (or
replayed) trade prints** (TBBO): when the model fires, a **limit entry** is
resting, then an **OCO bracket** (target limit + stop) manages the exit.

::

    # Live stream (Databento Live client → NQ.FUT tbbo)
    python -m src.paper.runner --stream

    # Offline: replay one session from TimescaleDB tbbo (true tick path)
    python -m src.paper.runner --replay-db 2026-08-04

See ``policy.PaperPolicy`` for barriers from the active objective (e.g. +50/−6).
"""

from __future__ import annotations

from src.paper.engine import FillEvent, MarketTick, PaperEngine, PaperTrade
from src.paper.policy import PaperPolicy, load_paper_policy

__all__ = [
    "FillEvent",
    "MarketTick",
    "PaperEngine",
    "PaperPolicy",
    "PaperTrade",
    "load_paper_policy",
]
