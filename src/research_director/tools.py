"""Tools available to the cognitive layer.

Two kinds:
  1. Context readers — deterministic loaders for the registry baseline artifacts,
     journal history, and reference corpus. Used by the context assembler.
  2. Bounded research tool — literature/web lookup with a hard per-finding query cap,
     internal-corpus-first. External search is wired behind a flag.

All readers are pure functions over the filesystem / DB so they are safe to call from
any agent without side effects.
"""

from __future__ import annotations

from src.common.data_paths import catalog_dir, model_registry_dir

import json
from pathlib import Path
from typing import Any

from src.common.db import get_pool

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Registry baseline (RD only uses promoted LATEST) ─────────────────────────
def load_registry_latest(
    objective_id: str,
    *,
    registry_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load the production baseline for *objective_id* from registry LATEST.

    Layout::

        model_registry/<objective_id>/LATEST.json
        model_registry/<objective_id>/<version>/   # serve bundle

    Returns None if nothing is promoted (RD cold-start / stand-down).
    Does **not** fall back to staging — only promoted models enter RD.
    """
    from src.serving.promote import load_latest, resolve_latest_bundle

    root = Path(registry_dir) if registry_dir else model_registry_dir()
    pointer = load_latest(objective_id, registry_root=root)
    if not pointer:
        return None
    bundle = resolve_latest_bundle(objective_id, registry_root=root)
    if bundle is None:
        return None
    man = _safe_json(bundle / "finalize_manifest.json") or {}
    schema = _safe_json(bundle / "feature_schema.json") or {}
    score = man.get("search_score") or man.get("metrics") or {}
    return {
        "version": pointer.get("version") or man.get("version") or bundle.name,
        "source": "registry",
        "path": str(bundle),
        "candidate_id": pointer.get("candidate_id") or man.get("candidate_id"),
        "objective_id": objective_id,
        "family": pointer.get("family") or man.get("family"),
        "config_snapshot": man.get("config_snapshot") or {
            "objective_id": objective_id,
            "family": man.get("family"),
            "candidate_id": man.get("candidate_id"),
        },
        "features": {
            "feature_cols": schema.get("feature_cols") or schema.get("columns") or [],
            "n_features": len(schema.get("feature_cols") or schema.get("columns") or []),
        },
        "training_result": score if isinstance(score, dict) else {},
        "provenance": _safe_json(bundle / "provenance.json") or pointer,
        "latest_pointer": pointer,
    }


def load_latest_stable(registry_dir: Path, program_id: str, model_name: str) -> dict[str, Any] | None:
    """Backward-compatible name: prefer objective-keyed LATEST, else legacy program= tree."""
    # New layout: program_id was often == objective_id
    base = load_registry_latest(program_id, registry_dir=registry_dir)
    if base is not None:
        return base
    # Legacy LSTM layout
    legacy = Path(registry_dir) / f"program={program_id}" / f"model_name={model_name}"
    pointer = legacy / "LATEST_STABLE.json"
    if not pointer.exists():
        return None
    latest = json.loads(pointer.read_text())
    version = latest.get("version")
    vdir = legacy / f"version={version}"
    return {
        "version": version,
        "source": "registry_legacy",
        "path": str(vdir),
        "config_snapshot": _safe_json(vdir / "config_snapshot.json"),
        "features": _safe_json(vdir / "features.json"),
        "training_result": _safe_json(vdir / "training_result.json"),
        "provenance": _safe_json(vdir / "provenance.json"),
    }


def resolve_baseline(
    registry_dir: Path,
    program_id: str,
    model_name: str = "",
    *,
    objective_id: str | None = None,
) -> dict[str, Any] | None:
    """RD baseline = registry LATEST for *objective_id* only (no staging fallback)."""
    oid = objective_id or program_id
    return load_registry_latest(oid, registry_dir=Path(registry_dir))


def load_recent_journal(program_id: str, limit: int = 25) -> list[dict[str, Any]]:
    """Recent experiments + outcomes so agents don't re-propose DISPROVEN ideas."""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, state, scope, title, hypothesis, closure_reason, tweaks
            FROM research_director.research_journal
            WHERE program_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (program_id, limit),
        ).fetchall()
    cols = ["experiment_id", "state", "scope", "title", "hypothesis", "closure_reason", "tweaks"]
    return [dict(zip(cols, r)) for r in rows]


def read_reference(rel_path: str, max_chars: int = 20000) -> str:
    """Read a repo reference doc (FEATURE_CATALOG, BACKLOG, an ADR), truncated."""
    p = _REPO_ROOT / rel_path
    if not p.exists():
        return ""
    return p.read_text()[:max_chars]


# ── Data / codebase inventory readers (agent context tools) ─────────────────
def db_inventory(schema: str = "databento") -> list[dict[str, Any]]:
    """High-level DB table inventory: name, approx row count, column count.

    Server-side aggregation over the catalog — no raw-row scans. Row counts are the
    planner's `n_live_tup` estimate (exact counts on billion-row hypertables are wasteful).
    """
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT t.table_name,
                   COALESCE(s.n_live_tup, 0)                                  AS approx_rows,
                   (SELECT count(*) FROM information_schema.columns c
                     WHERE c.table_schema = t.table_schema
                       AND c.table_name  = t.table_name)                      AS n_cols
            FROM information_schema.tables t
            LEFT JOIN pg_stat_user_tables s
                   ON s.schemaname = t.table_schema AND s.relname = t.table_name
            WHERE t.table_schema = %s AND t.table_type = 'BASE TABLE'
            ORDER BY approx_rows DESC
            """,
            (schema,),
        ).fetchall()
    return [
        {"table": f"{schema}.{r[0]}", "approx_rows": int(r[1]), "n_cols": int(r[2])}
        for r in rows
    ]


def dataset_catalog(program_id: str, features_root: str | None = None) -> dict[str, Any]:
    """Parquet/snapshot inventory for a program: latest snapshot pointer + catalog.

    Hive export (legacy) lived under ``data/features/program=…``; local hive was removed
    in the 2026-08 reorg. Returns {} unless *features_root* points at an existing tree.
    """
    if features_root is None:
        return {}
    base = Path(features_root)
    if not base.is_absolute():
        base = _REPO_ROOT / features_root
    base = base / f"program={program_id}"
    if not base.exists():
        return {}
    latest_files = sorted(base.rglob("LATEST.json"))
    if not latest_files:
        return {}
    latest_path = latest_files[-1]                       # newest schema_version dir
    snapshot_dir = latest_path.parent
    return {
        "snapshot_dir": str(snapshot_dir.relative_to(_REPO_ROOT)),
        "latest": _safe_json(latest_path),
        "catalog": _safe_json(snapshot_dir / "CATALOG.json"),
    }


def data_capability_catalog(
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the pre-built data capability catalog (ops/catalog/data_capability.json).

    Returns a dict with ``datasets`` (one entry per hypertable: data_type, granularity,
    instruments, date_range, fields) and ``parquet_features`` (parquet matrix coverage).
    Returns ``{}`` if the catalog has not been built yet; call
    ``python scripts/build_data_capability.py`` to generate it.

    Agents can use this to answer:
    - What top-level instruments are available (excluding granular option strikes)?
    - What is the date range / history depth for each dataset?
    - What granularity is available for a given data type?
    - What fields (columns) are in each dataset?
    """
    import json as _json
    from src.common.data_paths import catalog_dir
    path = Path(catalog_path or (catalog_dir() / "data_capability.json"))
    if not path.exists():
        return {}
    try:
        return _json.loads(path.read_text())
    except Exception:
        return {}


def read_attribution(registry_dir: Path, program_id: str, model_name: str) -> dict[str, Any]:
    """Feature attribution for the LATEST_STABLE model (group importance + IG top features).

    Pulls the evidence-rich attribution subset from `training_result.json` so an agent can
    ground a feature/architecture hypothesis without loading the full per-timestep profiles.
    """
    baseline = load_latest_stable(registry_dir, program_id, model_name)
    if not baseline:
        return {}
    tr = baseline.get("training_result") or {}
    ig = tr.get("integrated_gradients", {})
    ig_slim = {}
    if isinstance(ig, dict):
        ig_slim = {
            cls: (v or {}).get("top_features", [])[:10]
            for cls, v in ig.items() if cls in ("BIG_UP", "BIG_DOWN")
        }
    return {
        "feature_group_importance": tr.get("feature_group_importance"),
        "integrated_gradients_top": ig_slim,
    }


def load_docs(rel_paths: list[str], max_chars: int = 8000) -> dict[str, str]:
    """Load several repo docs (architecture / ADRs / external-research notes) by path."""
    return {p: read_reference(p, max_chars=max_chars) for p in rel_paths}


# ── Bounded research tool (shim → agentic RAG) ───────────────────────────────
class ResearchTool:
    """Literature lookup with a per-finding query budget.

    **Primary path (P1):** :func:`src.agentic.rag.research_search` (pgvector tiered
    corpus + substring fallback). The old internal-only grep is no longer the default.
    """

    def __init__(self, *, queries_per_finding: int = 3, prefer_internal: bool = True,
                 internal_dirs: list[str] | None = None, external_enabled: bool = False) -> None:
        self._budget = queries_per_finding
        self._prefer_internal = prefer_internal
        self._internal_dirs = internal_dirs or ["research_cases/papers", "docs/decisions"]
        self._external_enabled = external_enabled  # reserved; RAG already covers indexed T1

    def search(self, query: str, *, used: int = 0) -> list[dict[str, Any]]:
        """Return citation candidates; respects the per-finding query budget."""
        if used >= self._budget:
            return []
        from src.agentic.rag import research_search  # noqa: PLC0415
        return research_search(
            query, allow_substring_fallback=True, k=5)


def _safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
