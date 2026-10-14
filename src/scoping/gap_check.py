"""Pass C — deterministic gap check v2 (ADR-025 §5). Code, no LLM.

Diffs the synthesized ScopingPlan against what the system can actually run TODAY:

    data tier      reuse of the existing catalog-backed differ (src.scoping.runner._gap_check
                   over data/ops/catalog/data_capability.json) — range/instrument/field aware
    model tier     every proposed family vs src/models/catalog.py + the objective's
                   eligible_model_families pin; ensemble kinds vs src.search.ensembles
    feature tier   feature hints (view / prefixes) vs src/models/feature_views.py and the
                   capability catalog's parquet feature groups

Each finding carries a ``criticality`` 1–5 (impact on plan feasibility): a required data
class or the primary model missing = 5; optional/nice-to-have = 2. The report ranks items
so the human gate reads the binding constraints first. Recommendation ∈
acquire | revise_plan | proceed_with_partial | proceed — advisory only, the human decides.
"""

from __future__ import annotations

from src.common.data_paths import catalog_dir, data_root

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.scoping.gap_check")

_ENSEMBLE_KINDS = {"consensus", "vote", "stacking", "gate_agree", "gate_veto"}


def gap_check_v2(objective: dict[str, Any], plan: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Diff *plan* against data capability + model catalog + feature coverage."""
    from src.scoping.runner import _gap_check, _load_capability_catalog

    # ── data tier: reuse the existing catalog differ over the plan's data classes ──
    req_spec = {"data_classes": (plan.get("data_requirements") or {}).get("data_classes") or []}
    report = _gap_check(objective, req_spec, out_dir)
    for item in report.get("items", []):
        item["criticality"] = _data_criticality(item)

    catalog = _load_capability_catalog()
    report["model_coverage"] = _model_coverage(objective, plan)
    report["feature_coverage"] = _feature_coverage(plan, catalog)
    report["inventory_ref"] = report.get("inventory_ref") or (
        f"data_capability.json@{report.get('catalog_ts', 'unknown')}")

    # ── roll-up ────────────────────────────────────────────────────────────────
    s = report["summary"]
    s["n_model_missing"] = sum(1 for m in report["model_coverage"] if m["status"] != "have")
    s["n_feature_unknown"] = sum(1 for f in report["feature_coverage"] if f["status"] == "missing")
    n_candidates = len(((plan.get("model_plan") or {}).get("candidates")) or [])
    if n_candidates == 0:
        s["recommendation"] = "revise_plan"
        s["note"] = "plan has no model candidates — degenerate (dry-run or failed synthesis)"
    elif s["n_model_missing"] > 0:
        s["recommendation"] = "revise_plan"
    # else: keep the data-tier recommendation (acquire / proceed_with_partial / proceed)

    report["items"].sort(key=lambda i: -i.get("criticality", 1))
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


# ── model tier ────────────────────────────────────────────────────────────────

def _model_coverage(objective: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from src.models.catalog import get_catalog
        known = {e.name: e for e in get_catalog()}
    except Exception as exc:  # noqa: BLE001 — catalog import failure = everything unverifiable
        logger.warning("model_catalog_unavailable: %s", exc)
        known = {}
    eligible = objective.get("eligible_model_families") or []      # empty = all eligible

    out: list[dict[str, Any]] = []
    mp = plan.get("model_plan") or {}
    for c in mp.get("candidates") or []:
        fam = c.get("family", "")
        entry = known.get(fam)
        if entry is None:
            status, note = "missing", "family not in src/models/catalog.py — no plugin to run it"
            crit = 5 if c.get("role") == "primary" else 4
        elif eligible and fam not in eligible:
            status, note = "partial", "family exists but objective pins eligible_model_families without it"
            crit = 4
        else:
            status = "have"
            note = f"catalogued ({entry.family_group}, {entry.compute_class})"
            crit = 1
        out.append({"candidate": c.get("id") or fam, "family": fam, "role": c.get("role"),
                    "status": status, "note": note, "criticality": crit})
    for e in mp.get("ensembles") or []:
        kind = e.get("kind", "")
        ok = kind in _ENSEMBLE_KINDS
        out.append({"candidate": f"{kind}({','.join(e.get('members') or [])})",
                    "family": kind, "role": "ensemble",
                    "status": "have" if ok else "missing",
                    "note": "known combiner" if ok else
                            f"unknown ensemble kind — supported: {sorted(_ENSEMBLE_KINDS)}",
                    "criticality": 1 if ok else 3})
    return out


# ── feature tier ──────────────────────────────────────────────────────────────

def _feature_coverage(plan: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from src.models.feature_views import FEATURE_VIEWS
        views = dict(FEATURE_VIEWS)
    except Exception:  # noqa: BLE001
        views = {}
    known_prefixes = {p for plist in views.values() for p in plist}
    parquet_groups = set(((catalog.get("parquet_features") or {}).get("feature_groups")) or [])

    out: list[dict[str, Any]] = []
    for c in ((plan.get("model_plan") or {}).get("candidates")) or []:
        feats = c.get("features") or {}
        cid = c.get("id") or c.get("family")
        if not feats:
            out.append({"candidate": cid, "selector": "full-matrix", "status": "have",
                        "note": "no explicit selection — full feature matrix", "criticality": 1})
            continue
        if feats.get("view"):
            v = feats["view"]
            ok = v in views
            out.append({"candidate": cid, "selector": f"view:{v}",
                        "status": "have" if ok else "missing",
                        "note": "known view" if ok else
                                f"view not in feature_views.py — known: {sorted(views)}",
                        "criticality": 1 if ok else 4})
        if feats.get("prefixes"):
            unknown = [p for p in feats["prefixes"]
                       if not any(k.startswith(p) or p.startswith(k) for k in known_prefixes)
                       and p.rstrip("_") not in parquet_groups]
            out.append({"candidate": cid, "selector": f"prefixes:{feats['prefixes']}",
                        "status": "have" if not unknown else "partial",
                        "note": "all prefixes recognised" if not unknown else
                                f"unrecognised prefixes {unknown} — verify against the feature matrix",
                        "criticality": 1 if not unknown else 3})
        if feats.get("columns"):
            out.append({"candidate": cid, "selector": f"columns({len(feats['columns'])})",
                        "status": "partial",
                        "note": "explicit columns can only be verified against a built matrix",
                        "criticality": 2})
    return out


def _data_criticality(item: dict[str, Any]) -> int:
    status, required = item.get("status"), item.get("required", False)
    if status == "missing":
        return 5 if required else 2
    if status == "partial":
        return 4 if required else 2
    return 1
