"""Builds the Research Context Pack handed identically to every agent.

The pack grounds LLM proposals in the actual current system state rather than stale
static knowledge. It is assembled fresh each cycle and contains:
  - program objectives (the lens for Relevance scoring),
  - the LATEST_STABLE baseline artifacts (config / features / training_result),
  - recent journal entries (incl. DISPROVEN, so agents don't re-propose dead ideas),
  - reference corpus (FEATURE_CATALOG, BACKLOG, ADRs),
  - per-agent vector-memory recall (added per-expert by the runner, not here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.research_director import tools


@dataclass
class ContextPack:
    program_id: str
    program_objectives: dict[str, Any]
    baseline: dict[str, Any] | None
    recent_journal: list[dict[str, Any]]
    feature_catalog: str
    backlog: str
    extras: dict[str, str] = field(default_factory=dict)
    objective_progress: Any = None          # ObjectiveProgress (target distance; ADR-014)
    data_capability: dict[str, Any] | None = None  # data_capability.json artifact (Option A)
    outcome_patterns: list[dict[str, Any]] = field(default_factory=list)  # mined win/loss reports (ADR-021)

    def render(self, *, max_chars: int = 60000) -> str:
        """Flatten to a single text block for prompt injection (token-bounded)."""
        parts: list[str] = []
        parts.append("## PROGRAM OBJECTIVES\n" + json.dumps(self.program_objectives, indent=2))
        if self.objective_progress is not None:
            parts.append("## OBJECTIVE PROGRESS (distance to target — aim experiments at the "
                         "largest unmet gap)\n" + self.objective_progress.render())
        if self.outcome_patterns:
            blocks = []
            for r in self.outcome_patterns:
                head = (f"[{r.get('kind', 'model')}] {r.get('source_id', '?')} "
                        f"run {r.get('run_ts', '?')}")
                blocks.append(head + "\n" + (r.get("summary_text") or "")[:2500])
            parts.append(
                "## OUTCOME PATTERNS (mined success/failure scenarios — latest model &"
                " strategy runs)\n"
                "Deterministic win/loss/missed analysis of the most recent evaluated"
                " candidates. Use these separations, blind spots and seeds to aim your"
                " experiment at an OBSERVED failure mode — cite the pattern you attack.\n"
                + "\n\n".join(blocks)
            )
        if self.data_capability:
            summary_text = (self.data_capability.get("summary") or {}).get("text", "")
            cap_ts = self.data_capability.get("generated_at", "unknown")
            parts.append(
                f"## DATA CAPABILITY (internal data catalog, built {cap_ts[:10]})\n"
                "Use this to ground data-availability claims. "
                "Required/nice-to-have judgements for experiments must cite a row below.\n"
                + summary_text
            )
        if self.baseline is None:
            parts.append(
                "## BASELINE\n(cold start — no registry LATEST for this objective; "
                "promote a staged bundle before RD can improve it)")
        else:
            tr = self.baseline.get("training_result") or {}
            feat = self.baseline.get("features") or {}
            parts.append(
                "## BASELINE MODEL (registry LATEST v"
                + str(self.baseline.get("version")) + ")\n"
                + "### config_snapshot\n" + json.dumps(self.baseline.get("config_snapshot"), indent=2)[:8000]
                + "\n### feature_schema (groups enabled, prefixes)\n"
                + json.dumps(_feature_schema(feat), indent=2)[:2000]
                + "\n### training_result (key metrics + attribution)\n"
                + json.dumps(_slim_training_result(tr), indent=2)[:16000]
            )
        parts.append("## RECENT JOURNAL (avoid re-proposing DISPROVEN)\n"
                     + json.dumps(self.recent_journal, indent=2, default=str)[:8000])
        parts.append("## FEATURE CATALOG (candidate features to add)\n" + self.feature_catalog[:12000])
        parts.append("## BACKLOG\n" + self.backlog[:4000])
        for k, v in self.extras.items():
            parts.append(f"## {k}\n{v[:4000]}")
        return "\n\n".join(parts)[:max_chars]


def assemble(
    *,
    program_id: str,
    program_objectives: dict[str, Any],
    registry_dir: Path,
    model_name: str,
    journal_limit: int = 25,
    objective_id: str | None = None,
) -> ContextPack:
    # Production baseline only: model_registry/<objective>/LATEST.json
    oid = objective_id or program_id
    baseline = tools.resolve_baseline(
        Path(registry_dir), program_id, model_name or "", objective_id=oid,
    )

    # Target-pursuit contract (ADR-014): score the baseline against the objective's
    # `target` block so every agent sees the per-condition distance-to-goal, and the
    # Director can stand the loop down once the target is achieved.
    from src.research_director.objective_progress import assess_progress
    progress = assess_progress(
        program_objectives,
        (baseline or {}).get("diagnostics"),
        (baseline or {}).get("training_result"),
    )

    # Data capability catalog (Option A): load the pre-built artifact so every
    # agent pack includes a compact DATA CAPABILITY section with instruments,
    # granularity, and date ranges — no live DB queries at pack-assembly time.
    try:
        data_cap = tools.data_capability_catalog()
    except Exception:
        data_cap = None

    # Outcome patterns (ADR-021): the latest mined success/failure reports for this
    # program's objective — model OOF runs and user-strategy backtests alike — so both
    # agents see WHERE the current system wins, loses, and is blind, every cycle.
    try:
        try:
            from evaluation.episodes import latest_outcome_reports
        except ImportError:
            from src.evaluation.episodes import latest_outcome_reports
        outcome_patterns = latest_outcome_reports(program_id, limit=3)
    except Exception:
        outcome_patterns = []

    return ContextPack(
        program_id=program_id,
        program_objectives=program_objectives,
        baseline=baseline,
        recent_journal=tools.load_recent_journal(program_id, limit=journal_limit),
        feature_catalog=tools.read_reference("docs/reference/FEATURE_CATALOG.md"),
        backlog=tools.read_reference("docs/roadmap/BACKLOG.md"),
        objective_progress=progress,
        data_capability=data_cap,
        outcome_patterns=outcome_patterns,
    )


def _slim_training_result(tr: dict[str, Any]) -> dict[str, Any]:
    """Keep the evidence-rich fields; drop the bulky per-timestep IG profiles."""
    if not tr:
        return {}
    slim = {
        k: tr.get(k)
        for k in (
            "test_f1_macro", "test_f1_per_class", "confusion_matrix",
            "precision_at_tau", "feature_group_importance", "n_signal_test",
            # Fields the validation stage's Evaluator gate needs on the BASELINE side to compare a
            # re-trained candidate apples-to-apples (multi-fold consistency + bootstrap CI + the
            # identical-window guard). Absent them, validation can only return INCONCLUSIVE.
            "test_folds", "test_window_hash", "guardrails", "n_signals",
        )
        if k in tr
    }
    ig = tr.get("integrated_gradients", {})
    if isinstance(ig, dict):
        slim["integrated_gradients"] = {
            cls: {"top_features": (v or {}).get("top_features", [])[:10], "groups": (v or {}).get("groups", [])}
            for cls, v in ig.items() if cls in ("BIG_UP", "BIG_DOWN")
        }
    return slim


def _feature_schema(feat: dict[str, Any]) -> dict[str, Any]:
    """Compact feature schema for ContextPack: which groups are enabled + column counts.

    Derived from the baseline model's features.json rather than exposing the full
    ~180-column list (too verbose for a context pack). Agents can use read_reference
    on docs/reference/FEATURE_CATALOG.md for the full column-level detail.
    """
    if not feat:
        return {}
    groups = feat.get("enabled_groups") or {}
    cols = feat.get("feature_columns") or feat.get("features") or []
    prefixes: dict[str, int] = {}
    for col in (cols if isinstance(cols, list) else []):
        if "_" in str(col):
            p = str(col).split("_")[0] + "_"
            prefixes[p] = prefixes.get(p, 0) + 1
    return {
        "n_features": len(cols) if isinstance(cols, list) else feat.get("n_features"),
        "enabled_groups": {k: v for k, v in groups.items() if v} if groups else None,
        "column_counts_by_prefix": prefixes or None,
        "snapshot_id": feat.get("snapshot_id"),
    }
