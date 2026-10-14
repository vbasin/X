"""Stage 0 runner — strategy research + data-scoping + gap check.

Orchestrates the two duties of the Strategy Research Agent and emits three
versioned artifacts to ``data/research/scoping/<objective_id>/<run_ts>/``:

    data_requirements.json   — DataRequirementsSpec
    research_library.json    — ScoredResearchLibrary
    data_gap_report.json     — DataGapReport (for the human gate)

The agent LLM call is made through the existing ``src.research_director.llm``
client so cost governance and per-run budgets are inherited automatically.

Usage (CLI)::

    python -m src.scoping.runner --objective nq-50pt [--dry-run]

Usage (Python)::

    from src.scoping.runner import run_stage0
    report = run_stage0("nq-50pt")
"""

from __future__ import annotations

from src.common.data_paths import catalog_dir, data_root, scoping_dir

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("vtech.scoping.runner")

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _REPO_ROOT / "config"
_DATA_ROOT   = data_root()

# Schema paths
from src.common.config_paths import schema_path as _schema_path
_REQ_SCHEMA   = _schema_path("data_requirements.schema.json")
_LIB_SCHEMA   = _schema_path("research_library.schema.json")
_GAP_SCHEMA   = _schema_path("data_gap_report.schema.json")
_AGENT_PROMPT = _CONFIG_ROOT / "agents" / "strategy_research.md"


# ── Public entry point ───────────────────────────────────────────────────────

def run_stage0(
    objective_id: str,
    *,
    dry_run: bool = False,
    llm_budget_usd: float = 5.0,
) -> dict[str, Any]:
    """Run Stage 0 for *objective_id*.

    Parameters
    ----------
    objective_id:
        ID matching a ``config/objectives/<id>.yaml`` file.
    dry_run:
        If True, skip the LLM call and emit empty placeholder artifacts.
    llm_budget_usd:
        Maximum LLM spend for this run (passed to the LLM client).

    Returns
    -------
    dict
        The completed DataGapReport.
    """
    from src.objectives.loader import load_objective

    objective = load_objective(objective_id)
    run_ts    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir   = scoping_dir() / objective_id / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("stage0_start", extra={"objective_id": objective_id, "dry_run": dry_run, "out_dir": str(out_dir)})

    # ── Step 1: Agent run (Duties 1 + 2) ────────────────────────────────────
    if dry_run:
        req_spec = _empty_requirements(objective)
        lib      = _empty_library(objective)
    else:
        req_spec, lib = _run_agent(objective, budget_usd=llm_budget_usd)

    _write_json(out_dir / "data_requirements.json", req_spec, _REQ_SCHEMA)
    _write_json(out_dir / "research_library.json",  lib,      _LIB_SCHEMA)

    # ── Step 2: Deterministic gap check ─────────────────────────────────────
    report = _gap_check(objective, req_spec, out_dir)
    _write_json(out_dir / "data_gap_report.json", report, _GAP_SCHEMA)

    _print_gap_summary(report)
    logger.info("stage0_complete", extra={"out_dir": str(out_dir), "rec": report["summary"].get("recommendation")})
    return report


# ── Agent runner ─────────────────────────────────────────────────────────────

def _run_agent(objective: dict[str, Any], *, budget_usd: float) -> tuple[dict, dict]:
    """Invoke the recursive ReAct Strategy Research Agent (Duties 1 + 2).

    Delegates to ``src.scoping.agent.StrategyResearchAgent`` — a bounded plan→act→observe loop
    with read-only tools (inventory, internal corpus, web search/fetch, cost estimate). On any
    failure the agent returns None and we fall back to schema-shaped empty placeholders so
    Stage 0 always completes and the deterministic gap-check still runs.
    """
    try:
        from src.scoping.agent import StrategyResearchAgent
    except ImportError as exc:
        logger.warning("StrategyResearchAgent unavailable (%s) — returning empty placeholders", exc)
        return _empty_requirements(objective), _empty_library(objective)

    result = StrategyResearchAgent(objective, budget_usd=budget_usd).run()
    if result is None:
        logger.warning("strategy_research agent produced no result — using empty placeholders")
        return _empty_requirements(objective), _empty_library(objective)
    return result


# ── Gap check ────────────────────────────────────────────────────────────────

def _gap_check(
    objective: dict[str, Any],
    req_spec: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Diff req_spec against the Data Capability Catalog + live DB / parquet inventory.

    Gap evaluation is performed in two tiers:
    1. **Catalog tier** (preferred): reads ``data/ops/catalog/data_capability.json`` built by
       ``scripts/build_data_capability.py`` — supports range-aware and field-aware checks.
    2. **Fallback tier**: legacy live ``db_inventory()`` / ``dataset_catalog()`` queries
       used when the catalog file is absent (e.g., first run before the catalog is built).

    Each requirement data class is checked against the catalog and classified as:
    - ``have``    — the required instruments + date range are present
    - ``partial`` — the data type exists but date/instrument coverage is narrower than needed
    - ``missing`` — no matching dataset exists
    """
    catalog = _load_capability_catalog()
    if catalog:
        return _gap_check_from_catalog(objective, req_spec, out_dir, catalog)
    return _gap_check_legacy(objective, req_spec, out_dir)


def _load_capability_catalog() -> dict[str, Any]:
    """Load data/ops/catalog/data_capability.json, silently returning {} if absent."""
    try:
        from scripts.build_data_capability import load_catalog
        return load_catalog()
    except Exception:
        cap_path = catalog_dir() / "data_capability.json"
        if cap_path.exists():
            import json as _json
            return _json.loads(cap_path.read_text())
        return {}


# Category → data_types that satisfy it (matched against catalog dataset data_type field)
_CATEGORY_TO_DATA_TYPES: dict[str, list[str]] = {
    "primary_microstructure":  ["tick_bid_offer", "options_bbo_1s"],
    "aggregated_price":        ["ohlcv_1s", "equity_ohlcv", "tick_bid_offer"],
    "related_cross_asset":     ["ohlcv_1s"],
    "derivatives_positioning": ["options_bbo_1s", "statistics_daily"],
    "macro_government":        [],  # not yet ingested
    "other":                   [],
}


def _gap_check_from_catalog(
    objective: dict[str, Any],
    req_spec: dict[str, Any],
    out_dir: Path,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Catalog-backed gap check: range-aware and field-aware."""
    datasets = {ds["data_type"]: ds for ds in catalog.get("datasets", [])}
    parquet = catalog.get("parquet_features") or {}
    has_parquets = parquet.get("available", False)
    catalog_ts = catalog.get("generated_at", "unknown")

    items: list[dict[str, Any]] = []
    n_req_missing = 0
    n_opt_missing = 0
    n_req_partial = 0
    total_cost_usd = 0.0

    for dc in req_spec.get("data_classes", []):
        cat = dc.get("category", "other")
        req = dc.get("required", False)
        data_types = _CATEGORY_TO_DATA_TYPES.get(cat, [])

        # Check parquet as a special aggregated source for aggregated_price
        if cat == "aggregated_price" and has_parquets:
            data_types = data_types + ["__parquet__"]

        matched_ds = None
        for dt in data_types:
            if dt == "__parquet__":
                matched_ds = {"data_type": "__parquet__", "date_range": parquet.get("date_range"),
                              "instruments": [], "fields": parquet.get("feature_groups", [])}
                break
            if dt in datasets and datasets[dt].get("date_range"):
                matched_ds = datasets[dt]
                break

        if matched_ds is None:
            status = "missing"
            if req:
                n_req_missing += 1
            else:
                n_opt_missing += 1
            if dc.get("acquisition"):
                total_cost_usd += dc["acquisition"].get("est_cost_usd", 0.0)
            coverage_detail = None
        else:
            dr = matched_ds.get("date_range") or {}
            # Check whether the required instruments are present (if specified)
            req_instruments = dc.get("instruments") or []
            avail_symbols = {i["symbol"] for i in matched_ds.get("instruments", [])}
            missing_instruments = [s for s in req_instruments if s not in avail_symbols]
            # Check field coverage (if the requirement specifies fields)
            req_fields = dc.get("fields") or []
            avail_fields = set(matched_ds.get("fields") or [])
            missing_fields = [f for f in req_fields if f not in avail_fields]

            if missing_instruments or missing_fields:
                status = "partial"
                if req:
                    n_req_partial += 1
            else:
                status = "have"

            coverage_detail = {
                "table": matched_ds.get("table") or matched_ds.get("data_type"),
                "date_range": dr,
                "missing_instruments": missing_instruments,
                "missing_fields": missing_fields,
            }

        items.append({
            "data_class": dc.get("name", cat),
            "category": cat,
            "status": status,
            "required": req,
            "coverage": coverage_detail,
            "gap": dc.get("rationale", "") if status != "have" else None,
            "acquisition": dc.get("acquisition"),
        })

    recommendation = "proceed"
    if n_req_missing > 0:
        recommendation = "acquire"
    elif n_req_partial > 0:
        recommendation = "proceed_with_partial"

    return {
        "objective_id": objective.get("objective_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_ts": catalog_ts,
        "requirements_ref": str(out_dir / "data_requirements.json"),
        "items": items,
        "summary": {
            "n_required_missing": n_req_missing,
            "n_required_partial": n_req_partial,
            "n_optional_missing": n_opt_missing,
            "total_est_cost_usd": total_cost_usd or None,
            "recommendation": recommendation,
        },
        "human_decision": None,
    }


def _gap_check_legacy(
    objective: dict[str, Any],
    req_spec: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Legacy gap-check fallback (used when data_capability.json is absent).

    Shallow category-level check only. Rebuild with:
        python scripts/build_data_capability.py
    """
    try:
        from src.research_director.tools import db_inventory, dataset_catalog
        program_id = objective.get("objective_id", "")
        inv_tables  = db_inventory()
        inv_catalog = dataset_catalog(program_id)
        inventory_desc = (
            f"DB tables: {[t['table'] for t in inv_tables]}, "
            f"snapshot: {inv_catalog.get('latest_snapshot_id', 'none')}"
        )
    except Exception as exc:
        logger.warning("inventory_unavailable: %s — defaulting to 'missing'", exc)
        inv_tables, inv_catalog = [], {}
        inventory_desc = "unavailable"

    have_tables  = {t["table"].split(".")[-1] for t in inv_tables if t.get("approx_rows", 0) > 0}
    has_parquets = bool(inv_catalog.get("latest_snapshot_id"))

    _COVERAGE_MAP = {
        "primary_microstructure":  "tbbo" in have_tables,
        "aggregated_price":        "tbbo" in have_tables or has_parquets,
        "related_cross_asset":     "tbbo" in have_tables,
        "derivatives_positioning": "bbo_1s" in have_tables,
        "macro_government":        False,
    }

    items: list[dict[str, Any]] = []
    n_req_missing = 0
    n_opt_missing = 0
    total_cost_usd = 0.0
    for dc in req_spec.get("data_classes", []):
        cat     = dc.get("category", "other")
        present = _COVERAGE_MAP.get(cat, False)
        req     = dc.get("required", False)
        if not present:
            if req:
                n_req_missing += 1
            else:
                n_opt_missing += 1
            if dc.get("acquisition"):
                total_cost_usd += dc["acquisition"].get("est_cost_usd", 0.0)
        items.append({
            "data_class": dc.get("name", cat), "category": cat,
            "status": "have" if present else "missing", "required": req,
            "coverage": inventory_desc if present else None,
            "gap": dc.get("rationale", "") if not present else None,
            "acquisition": dc.get("acquisition"),
        })
    return {
        "objective_id": objective.get("objective_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements_ref": str(out_dir / "data_requirements.json"),
        "inventory_ref": inventory_desc,
        "items": items,
        "summary": {
            "n_required_missing": n_req_missing,
            "n_optional_missing": n_opt_missing,
            "total_est_cost_usd": total_cost_usd or None,
            "recommendation": "acquire" if n_req_missing > 0 else "proceed",
            "note": "legacy shallow check — run scripts/build_data_capability.py for full catalog",
        },
        "human_decision": None,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_requirements(objective: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective_id": objective.get("objective_id", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "traits": {
            "asset_class":    (objective.get("traded products") or [{}])[0].get("asset_class", ""),
            "strategy_type":  (objective.get("strategy_keywords") or [""])[0],
            "horizon_bucket": (objective.get("horizon") or {}).get("max_trade_horizon", ""),
        },
        "data_classes": [],
    }


def _empty_library(objective: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective_id": objective.get("objective_id", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "traits": {
            "asset_class":    (objective.get("traded products") or [{}])[0].get("asset_class", ""),
            "strategy_type":  (objective.get("strategy_keywords") or [""])[0],
            "horizon_bucket": (objective.get("horizon") or {}).get("max_trade_horizon", ""),
        },
        "publications": [],  # will fail schema minItems:10 — placeholder only
    }


def _write_json(path: Path, obj: dict[str, Any], schema_path: Path) -> None:
    """Write *obj* as JSON; warn if it fails schema validation."""
    path.write_text(json.dumps(obj, indent=2, default=str))
    try:
        from jsonschema import Draft202012Validator as V
        schema = json.loads(schema_path.read_text())
        errs = list(V(schema).iter_errors(obj))
        if errs:
            logger.warning("artifact_schema_errors path=%s count=%d first=%s",
                           path.name, len(errs), errs[0].message)
    except Exception:
        pass


def _print_gap_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print(f"\n── Data Gap Report ──────────────────────────────────────────")
    print(f"  Required missing : {s['n_required_missing']}")
    print(f"  Optional missing : {s['n_optional_missing']}")
    if s.get("total_est_cost_usd"):
        print(f"  Est. acq. cost   : ${s['total_est_cost_usd']:.2f}")
    print(f"  Advisory         : {s['recommendation'].upper()}")
    print(f"────────────────────────────────────────────────────────────\n")
    if s["n_required_missing"] > 0:
        print("⚠️  Required data is missing. Review the gap report and decide: acquire or proceed.\n")
    else:
        print("✅  All required data is present. Awaiting human confirmation to proceed.\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Phase 23 Stage 0 — strategy research + data-scoping")
    p.add_argument("--objective", required=True, help="Objective id, e.g. nq-50pt")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM; emit empty placeholders")
    p.add_argument("--budget", type=float, default=5.0, help="LLM budget USD (default 5.0)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_stage0(args.objective, dry_run=args.dry_run, llm_budget_usd=args.budget)


if __name__ == "__main__":
    _cli()
