"""src.scoping — Phase 23 Stage 0: Strategy Research & Data-Scoping.

Two steps, driven by the Strategy Research Agent persona
(``config/agents/strategy_research.md``):

1. The **agent runner** invokes the LLM persona and emits two JSON artifacts:
   - ``DataRequirementsSpec``  (config/data_requirements.schema.json)
   - ``ScoredResearchLibrary`` (config/research_library.schema.json)

2. The **gap checker** deterministically diffs the spec against the live DB /
   dataset inventory and writes a ``DataGapReport``
   (config/data_gap_report.schema.json) for the human gate.

Public API::

    from src.scoping import run_stage0
    report = run_stage0("nq-50pt")   # → DataGapReport dict; blocks for human decision
"""

from .runner import run_stage0

__all__ = ["run_stage0"]
