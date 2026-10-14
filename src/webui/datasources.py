"""Read/act adapters between the web console and the existing workflow surfaces.

Everything here is a thin, defensive wrapper over artifacts and modules that already
exist: objective YAMLs (`src.objectives.loader`), scoping run dirs (ADR-025 orchestrator),
search run dirs (Phase-23 controller), the Research Director systemd units, and the
`research_director` Postgres schema (same queries as ``scripts/rd_dashboard.py``).
DB and systemd probes are best-effort: they degrade to ``{"available": False}`` instead
of failing the endpoint.
"""

from __future__ import annotations

from src.common.data_paths import (
    data_root, model_registry_dir, model_staging_dir, runs_dir,
    scoping_dir, search_dir,
)

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("vtech.webui.datasources")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = data_root()
OBJECTIVES_DIR = _REPO_ROOT / "config" / "objectives"
SCOPING_ROOT = scoping_dir()
SEARCH_ROOT = search_dir()

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RUN_TS_RE = re.compile(r"^[0-9]{8}T[0-9]{6}(\.[0-9]+)?Z$")
ARTIFACT_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

SCOPING_STAGES = ("pass_a", "pass_b", "synthesis", "gap_check", "final_report")

RD_TIMER = "vtech-research-director.timer"
RD_SERVICE = "vtech-research-director.service"
_SERVICES = [
    "vtech-mlflow.service", RD_TIMER, RD_SERVICE, "vtech-oci-sync.timer",
    "vtech-live-streamer.service", "vtech-feature-builder.timer", "postgresql.service",
]

_MLFLOW_URL = os.environ.get("VTECH_MLFLOW_URL", "http://127.0.0.1:5000")


def check_id(value: str, what: str = "objective_id") -> str:
    if not ID_RE.match(value or ""):
        raise ValueError(f"invalid {what}: must match {ID_RE.pattern}")
    return value


def check_run_ts(value: str) -> str:
    if not RUN_TS_RE.match(value or ""):
        raise ValueError("invalid run timestamp")
    return value


# ── objectives ────────────────────────────────────────────────────────────────

def objective_schema() -> dict[str, Any]:
    return json.loads((OBJECTIVES_DIR / "objective.schema.json").read_text())


def _kpi_summary(obj: dict[str, Any]) -> str:
    kpi = obj.get("kpi") or {}
    return str(kpi.get("primary", ""))


def list_objectives() -> list[dict[str, Any]]:
    active = active_objective_id()
    out = []
    for f in sorted(OBJECTIVES_DIR.glob("*.yaml")):
        try:
            obj = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError as exc:
            out.append({"objective_id": f.stem, "error": str(exc)})
            continue
        prods = obj.get("traded products") or []
        primary = next((p for p in prods if p.get("role") == "primary"), prods[0] if prods else {})
        oid = obj.get("objective_id", f.stem)
        rd = {}
        try:
            from src.research_director.rd_enable import status_for_objective
            rd = status_for_objective(oid)
        except Exception:  # noqa: BLE001
            rd = {}
        out.append({
            "objective_id": oid,
            "description": (obj.get("description") or "").strip(),
            "symbol": primary.get("symbol"),
            "scheme": (obj.get("labels") or {}).get("scheme", "detection_2"),
            "move_threshold": (obj.get("labels") or {}).get("move_threshold"),
            "mae_tolerance": (obj.get("labels") or {}).get("mae_tolerance"),
            "kpi_primary": _kpi_summary(obj),
            "keywords": obj.get("strategy_keywords") or [],
            "active": oid == active,
            "file": f.name,
            "has_registry_latest": bool(rd.get("has_registry_latest")),
            "registry_version": rd.get("registry_version"),
            "rd_enabled": bool(rd.get("rd_enabled")),
        })
    return out


def get_objective(objective_id: str) -> dict[str, Any]:
    check_id(objective_id)
    path = OBJECTIVES_DIR / f"{objective_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(objective_id)
    rd = {}
    try:
        from src.research_director.rd_enable import status_for_objective
        rd = status_for_objective(objective_id)
    except Exception:  # noqa: BLE001
        rd = {}
    text = path.read_text()
    parsed = None
    parse_error = None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        parse_error = str(exc)
    return {"objective_id": objective_id, "yaml": text,
            "parsed": parsed, "parse_error": parse_error,
            "active": objective_id == active_objective_id(),
            **rd}


def validate_objective(obj: dict[str, Any]) -> list[str]:
    """Schema-validate a candidate objective dict; returns [] when valid."""
    import jsonschema
    validator = jsonschema.Draft202012Validator(objective_schema())
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(obj)]


def create_objective(obj: dict[str, Any]) -> Path:
    """Validate and persist a NEW objective YAML. Never overwrites (ADR-018: the
    objective doc is the single source of labels/KPI — edits are ADR-level events)."""
    oid = check_id(str(obj.get("objective_id", "")))
    errors = validate_objective(obj)
    if errors:
        raise ValueError("; ".join(errors))
    path = OBJECTIVES_DIR / f"{oid}.yaml"
    if path.exists():
        raise FileExistsError(f"objective '{oid}' already exists — objectives are "
                              f"immutable once created (write a new id instead)")
    header = (f"# Trade Objective — {oid}\n"
              f"# Created via the VTech web console. Validates against objective.schema.json.\n"
              f"# NOTE: config/objectives/ holds multiple objectives; the ACTIVE one is\n"
              f"# pinned in config.yaml via `objective_id`.\n")
    path.write_text(header + yaml.safe_dump(obj, sort_keys=False, allow_unicode=True,
                                            default_flow_style=False))
    return path


def active_objective_id() -> str | None:
    try:
        cfg = yaml.safe_load((_REPO_ROOT / "config.yaml").read_text()) or {}
        return cfg.get("objective_id")
    except (OSError, yaml.YAMLError):
        return None


# ── scoping runs (ADR-025) ────────────────────────────────────────────────────

def _scoping_status(state: dict[str, Any] | None, has_live_job: bool) -> tuple[str, int]:
    """Map state.json to (status, done_count)."""
    if not state:
        return ("running" if has_live_job else "empty"), 0
    stages = state.get("stages") or {}
    done = sum(1 for s in SCOPING_STAGES if (stages.get(s) or {}).get("status") == "done")
    if any((stages.get(s) or {}).get("status") == "failed" for s in SCOPING_STAGES):
        return "failed", done
    if done == len(SCOPING_STAGES):
        return "complete", done
    return ("running" if has_live_job else "aborted"), done


def list_scoping_runs(objective_id: str | None = None,
                      live_objectives: set[str] | None = None,
                      limit: int | None = None) -> list[dict[str, Any]]:
    live = live_objectives or set()
    if not SCOPING_ROOT.exists():
        return []
    roots = ([SCOPING_ROOT / check_id(objective_id)] if objective_id
             else [p for p in SCOPING_ROOT.iterdir() if p.is_dir()])
    # Enumerate run dirs (names only) and sort by run_ts desc BEFORE parsing, so a bounded
    # `limit` (e.g. the dashboard's newest-6) reads only the top-N state.json files instead
    # of every run's on each poll.
    pairs: list[tuple[Path, Path]] = []
    for obj_dir in roots:
        if not obj_dir.is_dir():
            continue
        for run_dir in obj_dir.iterdir():
            if run_dir.is_dir() and RUN_TS_RE.match(run_dir.name):
                pairs.append((obj_dir, run_dir))
    pairs.sort(key=lambda p: p[1].name, reverse=True)
    if limit is not None:
        pairs = pairs[:limit]
    runs: list[dict[str, Any]] = []
    for obj_dir, run_dir in pairs:
        state = None
        state_f = run_dir / "state.json"
        if state_f.exists():
            try:
                state = json.loads(state_f.read_text())
            except json.JSONDecodeError:
                state = None
        status, done = _scoping_status(state, obj_dir.name in live)
        runs.append({
            "objective_id": obj_dir.name, "run_ts": run_dir.name,
            "status": status, "stages_done": done, "stages_total": len(SCOPING_STAGES),
            "dry_run": bool((state or {}).get("dry_run")),
            "elapsed_s": (state or {}).get("elapsed_s"),
            "has_report": (run_dir / "final_research_report.md").exists(),
            "has_plan": (run_dir / "scoping_plan.json").exists(),
        })
    return runs


def scoping_run_dir(objective_id: str, run_ts: str) -> Path:
    d = SCOPING_ROOT / check_id(objective_id) / check_run_ts(run_ts)
    if not d.is_dir():
        raise FileNotFoundError(f"scoping run not found: {objective_id}/{run_ts}")
    return d


def scoping_run_detail(objective_id: str, run_ts: str,
                       live_objectives: set[str] | None = None) -> dict[str, Any]:
    d = scoping_run_dir(objective_id, run_ts)
    state = None
    if (d / "state.json").exists():
        try:
            state = json.loads((d / "state.json").read_text())
        except json.JSONDecodeError:
            state = None
    status, done = _scoping_status(state, objective_id in (live_objectives or set()))
    artifacts = [{"name": f.name, "size": f.stat().st_size,
                  "mtime": f.stat().st_mtime} for f in sorted(d.iterdir()) if f.is_file()]
    return {"objective_id": objective_id, "run_ts": run_ts, "status": status,
            "stages_done": done, "stages_total": len(SCOPING_STAGES),
            "stage_order": list(SCOPING_STAGES), "state": state, "artifacts": artifacts,
            "mlflow_url": f"{_MLFLOW_URL}/#/experiments"}


def read_scoping_artifact(objective_id: str, run_ts: str, name: str,
                          max_bytes: int = 2_000_000) -> dict[str, Any]:
    if not ARTIFACT_RE.match(name or ""):
        raise ValueError("invalid artifact name")
    d = scoping_run_dir(objective_id, run_ts)
    path = (d / name).resolve()
    if not path.is_relative_to(d.resolve()) or not path.is_file():
        raise FileNotFoundError(name)
    data = path.read_bytes()[:max_bytes]
    return {"name": name, "size": path.stat().st_size, "truncated": path.stat().st_size > max_bytes,
            "content": data.decode("utf-8", errors="replace")}


# ── search runs (Phase 23 controller) ────────────────────────────────────────

def _sanitize_nan(value: Any) -> Any:
    """Replace NaN/±Inf with None recursively — controller JSON contains NaN, which
    strict encoders (Starlette JSONResponse, browser JSON.parse) reject."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nan(v) for v in value]
    return value


def list_search_runs(objective_id: str | None = None,
                     limit: int | None = None) -> list[dict[str, Any]]:
    if not SEARCH_ROOT.exists():
        return []
    roots = ([SEARCH_ROOT / check_id(objective_id)] if objective_id
             else [p for p in SEARCH_ROOT.iterdir()
                   if p.is_dir() and not p.name.startswith("_")])
    # Names first, sort desc, THEN parse — so `limit` only json.loads the newest N result
    # files (the dashboard's newest-6) rather than every run's on each call.
    pairs: list[tuple[Path, Path]] = []
    for obj_dir in roots:
        if not obj_dir.is_dir():
            continue
        for run_dir in obj_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "search_result.json").exists():
                pairs.append((obj_dir, run_dir))
    pairs.sort(key=lambda p: p[1].name, reverse=True)
    runs: list[dict[str, Any]] = []
    for obj_dir, run_dir in pairs:
        if limit is not None and len(runs) >= limit:
            break
        try:
            res = json.loads((run_dir / "search_result.json").read_text())
        except json.JSONDecodeError:
            continue
        gate = res.get("gate_result") or {}
        runs.append(_sanitize_nan({
            "objective_id": obj_dir.name, "run_ts": run_dir.name,
            "n_candidates": res.get("n_candidates"), "n_ok": res.get("n_ok"),
            "n_failed": res.get("n_failed"),
            "gate_status": gate.get("status"),
            "champion_id": gate.get("champion_id"),
            "champion_precision": gate.get("champion_precision"),
            "meets_target": gate.get("champion_meets_target"),
        }))
    return runs


def search_run_detail(objective_id: str, run_ts: str) -> dict[str, Any]:
    check_id(objective_id)
    if not re.match(r"^[A-Za-z0-9._-]{1,64}$", run_ts or ""):
        raise ValueError("invalid run id")
    res_f = SEARCH_ROOT / objective_id / run_ts / "search_result.json"
    if not res_f.resolve().is_relative_to(SEARCH_ROOT.resolve()) or not res_f.exists():
        raise FileNotFoundError(f"search run not found: {objective_id}/{run_ts}")
    res = json.loads(res_f.read_text())
    for c in res.get("candidates") or []:
        c.pop("diagnostics", None)                        # bulky; persisted separately
    res["mlflow_url"] = _MLFLOW_URL
    return _sanitize_nan(res)


def catalog_families() -> list[dict[str, Any]]:
    from src.models.catalog import get_catalog
    return [{"name": e.name, "family_key": e.family_key,
             "display_name": getattr(e, "display_name", e.name),
             "family_group": getattr(e, "family_group", ""),
             "compute_class": getattr(e, "compute_class", "cpu"),
             "cost_class": getattr(e, "cost_class", ""),
             "needs_sequence": getattr(e, "needs_sequence", False),
             "feature_view": getattr(e, "feature_view", None),
             "notes": getattr(e, "notes", ""),
             "paradigm": getattr(e, "paradigm", "") or getattr(e, "family_group", "")}
            for e in get_catalog()]


# ── custom test (WebUI board → manual spec + topology overlay) ───────────────

def custom_test_bank(store: str | None = None) -> dict[str, Any]:
    from src.search.custom_test import feature_bank
    return feature_bank(store)


def save_custom_test_view(
    name: str,
    features: list[str],
    *,
    description: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    from src.search.custom_test import list_views, save_custom_view
    rec = save_custom_view(
        name, features, description=description, overwrite=overwrite)
    return {"view": rec, "views": list_views()}


def delete_custom_test_view(name: str) -> dict[str, Any]:
    from src.search.custom_test import delete_custom_view, list_views
    delete_custom_view(name)
    return {"ok": True, "name": name, "views": list_views()}


def custom_test_default(objective_id: str) -> dict[str, Any]:
    from src.search.custom_test import default_board
    detail = get_objective(objective_id)
    parsed = detail.get("parsed") or {}
    if not parsed.get("objective_id"):
        parsed = {**parsed, "objective_id": objective_id}
    return default_board(parsed)


def custom_test_preview(board: dict[str, Any]) -> dict[str, Any]:
    from src.search.custom_test import preview_board
    oid = check_id(str(board.get("objective_id") or ""))
    get_objective(oid)
    board = {**board, "objective_id": oid}
    return preview_board(board)


def custom_test_root(objective_id: str) -> Path:
    from src.common.data_paths import webui_dir
    return webui_dir() / "custom_tests" / check_id(objective_id)


def persist_custom_test(board: dict[str, Any]) -> dict[str, Any]:
    """Compile *board*, write artifacts under ops/webui/custom_tests/, return preview + paths."""
    from datetime import datetime, timezone

    from src.search.custom_test import compile_board, persist_compiled

    oid = check_id(str(board.get("objective_id") or ""))
    get_objective(oid)
    board = {**board, "objective_id": oid}
    compiled = compile_board(board)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = custom_test_root(oid)
    paths = persist_compiled(compiled, dest, run_id)
    return {
        "run_id": run_id,
        "n_candidates": compiled.n_combos,
        "tune_trials": compiled.tune_trials,
        "warnings": compiled.warnings,
        "candidates": compiled.candidates,
        "topology": compiled.topology,
        "walk_forward": compiled.walk_forward,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def build_argv_custom_test(
    objective_id: str,
    manual_spec: str | Path,
    topology_overlay: str | Path | None,
    dry_run: bool,
    *,
    tune_trials: int = 0,
    walk_forward_overlay: str | Path | None = None,
) -> list[str]:
    """Argv for a compiled Custom Test. Specs must live under ops/webui/custom_tests/."""
    oid = check_id(objective_id)
    allowed = custom_test_root(oid).resolve()
    spec = Path(manual_spec).resolve()
    if not spec.is_file() or not spec.is_relative_to(allowed):
        raise ValueError("custom-test spec must be a file under ops/webui/custom_tests/<objective>/")
    argv = [sys.executable, "-m", "src.search.controller",
            "--objective", oid, "--manual-spec", str(spec), "--no-ensemble",
            "--experiment-type", "custom_test"]
    if topology_overlay:
        topo = Path(topology_overlay).resolve()
        if not topo.is_file() or not topo.is_relative_to(allowed):
            raise ValueError("topology overlay must be a file under ops/webui/custom_tests/<objective>/")
        argv += ["--topology-overlay", str(topo)]
    if walk_forward_overlay:
        wf = Path(walk_forward_overlay).resolve()
        if not wf.is_file() or not wf.is_relative_to(allowed):
            raise ValueError("walk-forward overlay must be a file under ops/webui/custom_tests/<objective>/")
        argv += ["--walk-forward-overlay", str(wf)]
    n_trials = int(tune_trials or 0)
    if n_trials > 0:
        argv += ["--tune-trials", str(n_trials)]
    if dry_run:
        argv.append("--dry-run")
    return argv


# ── backtest (staged weights via src.serving — same inference as Live) ───────

BACKTEST_MAX_SESSIONS = 400
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
def _default_feature_root():
    from src.features.stores import parquet_dir_for_store
    return parquet_dir_for_store("5s")


# Tests monkeypatch this to a tmp day-file dir.
_PARQUET_ROOT = None


def _feature_root():
    return _PARQUET_ROOT if _PARQUET_ROOT is not None else _default_feature_root()


def _feature_parquet_span() -> tuple[str | None, str | None, int]:
    """Span of on-disk feature days (what weights backtest can score)."""
    from src.backtest.runner import _available_dates

    roots: list[Path] = []
    if _PARQUET_ROOT is not None:
        roots.append(_PARQUET_ROOT)
    else:
        from src.features.stores import parquet_dir_for_store
        for store in ("5s", "15s", "1m"):
            try:
                roots.append(parquet_dir_for_store(store))
            except Exception:
                continue
    for root in roots:
        if not root.is_dir():
            continue
        dates = _available_dates(root)
        if dates:
            return dates[0], dates[-1], len(dates)
    return None, None, 0


def _training_windows(bundle_path: str) -> dict[str, Any]:
    """Derive chronological train/val date ranges from a bundle's finalize manifest.

    Uses the search run's OOF parquet (min/max ts) plus the finalize refit split:
    ``val`` is the last ``n_val_rows`` chronologically of the full training matrix;
    ``train`` is everything before that back to the earliest feature parquet that
    was available at training time.

    Returns keys (all optional strings YYYY-MM-DD or ints):
        train_start, train_end, train_rows
        val_start,   val_end,   val_rows
        search_run_ts
    Empty dict on any lookup failure.
    """
    try:
        d = Path(bundle_path)
        mp = d / "finalize_manifest.json"
        if not mp.exists():
            return {}
        manifest = json.loads(mp.read_text())
        n_train = int(manifest.get("n_train_rows") or 0)
        n_val = int(manifest.get("n_val_rows") or 0)
        if n_train <= 0 or n_val <= 0:
            return {}
        srd = manifest.get("search_run_dir") or ""
        cid = manifest.get("candidate_id") or ""
        oid = manifest.get("objective_id") or ""
        run_ts = manifest.get("search_run_ts") or ""
        if not cid:
            return {}
        # Prefer the canonical layout (SEARCH_ROOT / objective / run_ts) — the
        # manifest's stored `search_run_dir` may point at a stale relative path
        # from an earlier data-layout revision.
        search_dir_path: Path | None = None
        if oid and run_ts:
            candidate = SEARCH_ROOT / oid / run_ts
            if candidate.is_dir():
                search_dir_path = candidate
        if search_dir_path is None and srd:
            for candidate in (_REPO_ROOT / srd, _DATA_ROOT / srd.removeprefix("data/")):
                if candidate.is_dir():
                    search_dir_path = candidate
                    break
        if search_dir_path is None:
            return {}
        oof_path = search_dir_path / f"oof_{cid}.parquet"
        if not oof_path.exists():
            return {}
        try:
            import pandas as pd  # heavy; imported lazily
        except ImportError:
            return {}
        oof = pd.read_parquet(oof_path, columns=["ts"])
        if oof.empty or "ts" not in oof.columns:
            return {}
        ts = pd.to_datetime(oof["ts"]).sort_values().reset_index(drop=True)
        n_oof = len(ts)
        # val is the last n_val rows of the FULL matrix (chunks 0+1+2+3);
        # OOF represents chunks 1+2+3, so val_start_row_in_OOF = n_oof - n_val.
        val_row_ix = max(0, n_oof - n_val)
        val_start_ts = ts.iloc[val_row_ix]
        val_end_ts = ts.iloc[-1]
        # train_end = one bar before val_start (approximate to date)
        train_end_ts = ts.iloc[max(0, val_row_ix - 1)]
        # train_start: chunk 0 sits BEFORE the OOF window. Use the earliest
        # feature parquet on disk as the corpus lower bound (a good proxy —
        # chunk 0 is fold_size rows == oof_size / 3, chronologically contiguous
        # with OOF start).
        span0, _, _ = _feature_parquet_span()
        train_start = span0  # earliest available feature day
        # Fold_size in rows (for context)
        fold_size = n_oof // 3
        return {
            "train_start": train_start,
            "train_end": train_end_ts.date().isoformat(),
            "train_rows": n_train,
            "val_start": val_start_ts.date().isoformat(),
            "val_end": val_end_ts.date().isoformat(),
            "val_rows": n_val,
            "search_run_ts": manifest.get("search_run_ts"),
            "oof_start": ts.iloc[0].date().isoformat(),
            "oof_end": val_end_ts.date().isoformat(),
            "fold_size": fold_size,
            "n_folds": 3,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("training_windows_lookup_failed path=%s err=%s", bundle_path, exc)
        return {}


def _oof_date_span(bundle_path: str | Path) -> tuple[str | None, str | None]:
    """min/max date of walk-forward ``oof.parquet`` (not the feature hive)."""
    p = Path(bundle_path)
    oof = p / "oof.parquet" if p.is_dir() else p
    if oof.name != "oof.parquet" or not oof.is_file():
        return None, None
    try:
        import pandas as pd
        ts = pd.to_datetime(pd.read_parquet(oof, columns=["ts"])["ts"], utc=True)
        if ts.empty:
            return None, None
        return ts.min().date().isoformat(), ts.max().date().isoformat()
    except Exception:
        return None, None


_OBJ_FIRE_TAU: dict[str, float | None] = {}


def _objective_recommended_tau(objective_id: str) -> float | None:
    """Dual-binary ``labels.fire`` (lg-st 0.40). 3-class report_only → None (keep 80%)."""
    oid = str(objective_id or "").strip()
    if not oid or not ID_RE.match(oid):
        return None
    if oid in _OBJ_FIRE_TAU:
        return _OBJ_FIRE_TAU[oid]
    try:
        from src.evaluation.kpi_criteria import fire_tau
        from src.objectives.loader import load_objective
        t = fire_tau(load_objective(oid), report_only=True)
        val = float(t) if t is not None else None
    except Exception:
        val = None
    _OBJ_FIRE_TAU[oid] = val
    return val


def _inference_policy_fields(pol: dict[str, Any] | None) -> dict[str, Any]:
    """Surface bundle inference policy on the Backtest model dropdown."""
    if not isinstance(pol, dict) or not pol:
        return {}
    out: dict[str, Any] = {"inference_policy": pol}
    tau = pol.get("tau")
    if tau is not None:
        try:
            out["recommended_tau"] = float(tau)
        except (TypeError, ValueError):
            pass
    note = str(pol.get("note") or "").strip()
    if note:
        out["detail"] = note
    return out


def list_backtestable_models(limit: int | None = None) -> list[dict[str, Any]]:
    """Staged/registry bundles plus recent Custom Test lab models."""
    from src.webui.live import list_serving_bundles
    span0, span1, n_days = _feature_parquet_span()
    out: list[dict[str, Any]] = []

    for cascade in _cascade_oof_paths():
        if not cascade.is_file():
            continue
        n_rows = 0
        oof0 = oof1 = None
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(cascade)
            n_rows = int(pf.metadata.num_rows)
            try:
                first = pf.read_row_group(0, columns=["session"]).to_pandas()
                last = pf.read_row_group(pf.num_row_groups - 1,
                                         columns=["session"]).to_pandas()
                oof0 = str(pd.to_datetime(first["session"]).min().date())
                oof1 = str(pd.to_datetime(last["session"]).max().date())
            except Exception:
                pass
        except Exception:
            pass
        label, detail = _cascade_oof_meta(cascade)
        out.append({
            "source_kind": "cascade_oof",
            "source": str(cascade),
            "model": f"mnq_auction_cascade{'' if cascade.name == 'target_scores.parquet' else '_' + cascade.stem.removeprefix('target_scores_')}",
            "objective_id": "",
            "run_ts": "",
            "label": label,
            "run_label": "MNQ Auction Cascade",
            "span_start": oof0 or span0,
            "span_end": oof1 or span1,
            "n_rows": n_rows,
            "backtestable": True,
            "recommended_tau": 0.75,
            "detail": detail,
        })

    for b in list_serving_bundles(None):
        rec = {
            "source_kind": "weights",
            "source": b["path"],
            "model": b.get("candidate_id") or b["name"],
            "objective_id": b.get("objective_id") or "",
            "run_ts": b.get("version") or b["name"],
            "label": (b.get("label")
                      or b.get("run_label")
                      or f"{b.get('candidate_id') or b['name']} · {b.get('objective_id') or '?'}"),
            "run_label": b.get("run_label") or b.get("name"),
            "span_start": span0,
            "span_end": span1,
            "n_rows": n_days,          # feature days available to score
            "backtestable": span0 is not None,
            "detail": (
                "Current staged weights score historical feature parquets "
                "via src.serving (same inference as Live). Not as-of-date weights."
            ),
        }
        rec.update(_training_windows(b["path"]))
        rec.update(_inference_policy_fields(b.get("inference_policy")))
        if rec.get("recommended_tau") is None:
            rt = _objective_recommended_tau(rec["objective_id"])
            if rt is not None:
                rec["recommended_tau"] = rt
        out.append(rec)
    for b in _list_custom_test_bundles(limit=80):
        oof0, oof1 = _oof_date_span(b["path"])
        rec = {
            "source_kind": "custom_test",
            "source": b["path"],
            "model": b.get("candidate_id") or b["name"],
            "objective_id": b.get("objective_id") or "",
            "run_ts": b.get("run_ts") or "",
            "label": b["label"],
            "run_label": b["run_label"],
            "span_start": oof0 or span0,
            "span_end": oof1 or span1,
            "n_rows": n_days,
            "backtestable": (oof0 or span0) is not None,
            "detail": (
                "Last-fold 15s LSTM + 1m/5m XGB asof + meta XGB (ADR-103). "
                "Not promoted to Live."
            ) if str(b.get("candidate_id") or b.get("name") or "").startswith("tf_oof_meta") else (
                "Custom Test walk-forward OOF replay on the bundle's native-TF "
                "hive (as-of fold models). Last-fold weights are not scored on "
                "train dates. Not promoted to Live."
            ),
        }
        rec.update(_training_windows(b["path"]))
        rec.update(_inference_policy_fields(b.get("inference_policy")))
        if rec.get("recommended_tau") is None:
            rt = _objective_recommended_tau(rec["objective_id"])
            if rt is not None:
                rec["recommended_tau"] = rt
        out.append(rec)
    out.sort(key=lambda m: (m["backtestable"], m.get("run_ts") or "", m["model"]),
             reverse=True)
    cascade_rows = [m for m in out if m.get("source_kind") == "cascade_oof"]
    if cascade_rows:
        out = cascade_rows + [m for m in out if m.get("source_kind") != "cascade_oof"]
    return out[:limit] if limit else out


def _list_custom_test_bundles(limit: int = 40) -> list[dict[str, Any]]:
    """Newest research-run serving folders, canonical root first.

    Walks research/runs/ and then the historical roots through the shared
    resolver, so a migrated run is listed once from research/runs/ and an
    unmigrated one is still reachable.
    """
    from src.common.run_layout import run_roots
    from src.serving import is_serving_bundle
    found: list[tuple[float, dict[str, Any]]] = []
    obj_dirs: list[Path] = []
    seen_obj: set[tuple[str, str]] = set()
    for root in run_roots():
        if not root.is_dir():
            continue
        try:
            for p in root.iterdir():
                if p.is_dir() and (root.name, p.name) not in seen_obj:
                    seen_obj.add((root.name, p.name))
                    obj_dirs.append(p)
        except OSError:
            continue
    for obj in obj_dirs:
        try:
            runs = [p for p in obj.iterdir() if p.is_dir()]
        except OSError:
            continue
        for run in runs:
            models = run / "models"
            if not models.is_dir():
                continue
            try:
                cands = list(models.iterdir())
            except OSError:
                continue
            for cand in cands:
                if not cand.is_dir():
                    continue
                serving = is_serving_bundle(cand)
                has_oof = (cand / "oof.parquet").is_file()
                if not serving and not has_oof:
                    continue
                man: dict[str, Any] = {}
                mp = cand / "finalize_manifest.json"
                if mp.exists():
                    try:
                        man = json.loads(mp.read_text())
                    except json.JSONDecodeError:
                        man = {}
                oid = man.get("objective_id") or obj.name
                cid = man.get("candidate_id") or cand.name
                run_ts = man.get("search_run_ts") or run.name
                run_label = f"{oid}/{run_ts}/{cid}"
                ntf = str(man.get("native_tf") or "").strip()
                if not ntf:
                    for tok in ("15s", "5s", "5m", "1m", "1h", "1d"):
                        if tok in str(cid).replace("-", "_").split("_"):
                            ntf = tok
                            break
                tag = f"{ntf} · custom test" if ntf else "custom test"
                mtime = 0.0
                try:
                    mtime = mp.stat().st_mtime if mp.exists() else cand.stat().st_mtime
                except OSError:
                    pass
                pol = man.get("inference_policy")
                if not isinstance(pol, dict) or not pol:
                    ip = cand / "inference_policy.json"
                    if ip.exists():
                        try:
                            loaded = json.loads(ip.read_text())
                        except (OSError, json.JSONDecodeError):
                            loaded = None
                        pol = loaded if isinstance(loaded, dict) else None
                found.append((mtime, {
                    "path": str(cand.resolve()),
                    "name": cand.name,
                    "candidate_id": cid,
                    "objective_id": oid,
                    "run_ts": run_ts,
                    "serving": serving,
                    "run_label": run_label,
                    "label": f"{run_label} · {tag}",
                    "inference_policy": pol,
                }))
    found.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in found[:limit]]


def _cascade_oof_path() -> Path:
    return data_root() / "sources" / "labels" / "target_scores.parquet"


def _cascade_oof_paths() -> list[Path]:
    """All ``target_scores*.parquet`` files under ``sources/labels/``.

    Convention: ``target_scores.parquet`` is the ADR-127 baseline (fixed 0.45 R
    stop); ``target_scores_<slug>.parquet`` are variants — e.g.
    ``target_scores_adaptive.parquet`` for ADR-128's target-adaptive stops.
    """
    root = data_root() / "sources" / "labels"
    if not root.is_dir():
        return []
    files = [p for p in root.glob("target_scores*.parquet") if p.is_file()]
    # Baseline first, then the rest alphabetically — keeps the default at the top.
    files.sort(key=lambda p: (0 if p.name == "target_scores.parquet" else 1,
                              p.name))
    return files


def _cascade_oof_meta(path: Path) -> tuple[str, str]:
    """UI label and detail string for a cascade OOF variant."""
    if path.name == "target_scores.parquet":
        return (
            "MNQ Auction Cascade — Stage 1 + Stage 3 OOF (fixed 0.45 R stop)",
            "TBBO-tape three-stage cascade (ADR-127): Stage 1 move detector "
            "(mv gate) → fixed 0.45 R stop → Stage 3 target selector (tau gate). "
            "OOF payload derived directly from target_scores.parquet — no serving "
            "bundle load. MNQ ($2/pt); default operating cell mv_0.5 / tau_0.75.",
        )
    slug = path.stem.removeprefix("target_scores_") or path.stem
    if slug == "adaptive":
        return (
            "MNQ Auction Cascade — target-adaptive stops (ADR-130)",
            "TBBO-tape cascade with Stage 2 folded into Stage 3 (ADR-130): each row "
            "row uses stop_dist_r = min(0.5 * tgt_dist_r, 0.25) R, and the WebUI "
            "gates on p_move, p_reach, and a minimum target size in R units. "
            "OOF payload derived from target_scores_adaptive.parquet — no serving "
            "bundle load. MNQ ($2/pt).",
        )
    if slug == "context":
        return (
            "MNQ Auction Cascade — context pack + immediate entry",
            "TBBO-tape cascade with the 28-column curated CONTEXT pack "
            "(es_/fs_/st_ prefixes) added to the target selector. Same "
            "target-adaptive stop as ADR-130. Immediate-fill at the decision "
            "price. OOF payload derived from target_scores_context.parquet — "
            "no serving bundle load. MNQ ($2/pt).",
        )
    if slug == "context_rest":
        return (
            "MNQ Auction Cascade — context pack + resting entry",
            "Same context+adaptive-stop model as target_scores_context, but the "
            "order rests at the far edge of the containing bound (ADR-126) and "
            "rows that never fill are dropped. OOF payload derived from "
            "target_scores_context_rest.parquet — no serving bundle load. MNQ ($2/pt).",
        )
    if slug == "v3":
        return (
            "MNQ Auction Cascade — v3 (causal delta/OFI/absorption fix)",
            "Same context+adaptive-stop cascade, rebuilt on move_matrix_v3/"
            "target_matrix_v3 after fixing the floor-labelled 1s aggregation that "
            "leaked up to ~1s of future prints into fs_delta_*/fs_ofi_*/"
            "fs_delta_ratio_*/fs_absorption_*/fs_impact_efficiency_* (on top of "
            "the earlier fs_clv fix). Delta/OFI/volume are now derived causally "
            "per-print via as-of running totals. OOF payload derived from "
            "target_scores_v3.parquet — no serving bundle load. MNQ ($2/pt).",
        )
    if slug == "v4":
        return (
            "MNQ Auction Cascade — v4 (multi-session clustered bounds)",
            "Same context+adaptive-stop cascade as v3, with Stage 1's p_move "
            "now also reading the multi-session KDE-clustered structural bounds "
            "(es_n_bounds/es_inside_bound_*/es_vs_bound_*/es_pos_in_channel/"
            "es_channel_width_r) built from up to 20 prior sessions' volume-"
            "profile levels, the same Bound structure Stage 3 targets against. "
            "Rebuilt on move_matrix_v4/target_matrix_v4. OOF payload derived "
            "from target_scores_v4.parquet — no serving bundle load. MNQ ($2/pt).",
        )
    return (
        f"MNQ Auction Cascade — {slug} variant",
        f"Cascade OOF variant loaded from {path.name}.",
    )


def _backtest_browse_roots() -> list[tuple[str, str, Path]]:
    """Named browse roots: (id, label, resolved path if it exists)."""
    from src.common.data_paths import model_artifact_roots
    roots: list[tuple[str, str, Path]] = []
    seen: set[Path] = set()
    candidates = [
        ("staging", "Staged models", model_staging_dir()),
        ("registry", "Registry", model_registry_dir()),
        ("runs", "Research runs", runs_dir()),
        ("search", "Search runs (historical)", search_dir()),
        ("cascade_oof", "Cascade OOF", data_root() / "sources" / "labels"),
    ]
    for rid, label, raw in candidates:
        try:
            if not raw.exists():
                continue
            p = raw.resolve()
        except OSError:
            continue
        if p in seen:
            continue
        seen.add(p)
        roots.append((rid, label, p))
    for extra in model_artifact_roots():
        try:
            p = extra.resolve()
        except OSError:
            continue
        if p in seen or not p.exists():
            continue
        seen.add(p)
        name = "Registry" if "registry" in p.name else "Staged models"
        roots.append((p.name, name, p))
    return roots


def _browse_pick_meta(path: Path) -> dict[str, Any]:
    """objective_id + run_label for a browse-selectable file or bundle dir."""
    p = Path(path)
    oid = _objective_from_browse_path(p)
    leaf = p.stem[4:] if p.is_file() and p.name.startswith("oof_") else p.name
    label = leaf
    from src.common.run_layout import run_roots
    for root in run_roots():
        try:
            rel = p.resolve().relative_to(root.resolve())
            parts = list(rel.parts)
            if parts and p.is_file():
                parts[-1] = leaf
            if parts:
                label = "/".join(parts)
            break
        except (ValueError, OSError):
            continue
    if label == leaf:
        parent = p.parent.name
        if oid and parent and parent != oid:
            label = f"{oid}/{parent}/{leaf}"
        elif oid:
            label = f"{oid}/{leaf}"
        elif parent:
            label = f"{parent}/{leaf}"
    return {"objective_id": oid or "", "run_label": label, "candidate_id": leaf}


def _browse_root_for(path: Path) -> Path | None:
    for _rid, _label, root in _backtest_browse_roots():
        if path == root or root in path.parents:
            return root
    return None


def browse_backtest_sources(path: str | None = None) -> dict[str, Any]:
    """List a directory under staging / registry / research/search for the Backtest picker."""
    from src.serving import is_serving_bundle

    roots = _backtest_browse_roots()
    if not path:
        return {
            "path": None,
            "parent": None,
            "entries": [
                {"name": label, "path": str(root), "kind": "root",
                 "n_children": sum(1 for _ in root.iterdir()) if root.is_dir() else 0}
                for _rid, label, root in roots
            ],
        }

    raw = Path(path)
    if not raw.is_absolute():
        raw = _REPO_ROOT / raw
    try:
        cur = raw.resolve()
    except OSError as exc:
        raise ValueError(f"invalid path: {exc}") from exc
    root = _browse_root_for(cur)
    if root is None:
        raise ValueError(
            "path must be under model_staging/, model_registry/, "
            "research/runs/, research/custom_test/, or research/search/")
    if not cur.is_dir():
        raise ValueError(f"not a directory: {cur}")

    entries: list[dict[str, Any]] = []
    def _browse_sort(p: Path) -> tuple:
        ts = bool(p.is_dir() and RUN_TS_RE.match(p.name))
        # timestamp dirs newest first (negate by reversing name), then other dirs, then files
        return (0 if ts else (1 if p.is_dir() else 2),
                ("" if not ts else p.name),  # placeholder; see reverse below
                p.name)
    kids = list(cur.iterdir())
    ts_dirs = sorted((p for p in kids if p.is_dir() and RUN_TS_RE.match(p.name)),
                     key=lambda p: p.name, reverse=True)
    other = sorted((p for p in kids if p not in set(ts_dirs)),
                   key=lambda p: (not p.is_dir(), p.name))
    kids = ts_dirs + other
    for child in kids:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            if is_serving_bundle(child):
                meta = _browse_pick_meta(child)
                entries.append({
                    "name": child.name, "path": str(child), "kind": "bundle",
                    "objective_id": meta["objective_id"],
                    "candidate_id": meta["candidate_id"],
                    "run_label": meta["run_label"],
                    "selectable": True,
                })
            elif (child / "oof.parquet").is_file():
                meta = _browse_pick_meta(child)
                entries.append({
                    "name": child.name, "path": str(child), "kind": "oof",
                    "objective_id": meta["objective_id"],
                    "candidate_id": meta["candidate_id"],
                    "run_label": meta["run_label"],
                    "selectable": True,
                })
            else:
                n_oof = 0
                if child.is_dir():
                    n_oof = sum(1 for f in child.glob("oof_*.parquet"))
                    n_oof += sum(1 for f in child.glob("models/*/oof.parquet"))
                entries.append({
                    "name": child.name, "path": str(child), "kind": "dir",
                    "n_oof": n_oof, "selectable": False,
                })
        elif child.is_file() and child.name.startswith("oof_") and child.suffix == ".parquet":
            continue  # root oof_*.parquet tapes are not listed; pick the model folder
        elif child.is_file() and child.suffix == ".parquet":
            from src.backtest.cascade_backtest import is_cascade_oof
            if is_cascade_oof(child):
                entries.append({
                    "name": child.name, "path": str(child), "kind": "cascade_oof",
                    "objective_id": "",
                    "candidate_id": "mnq_auction_cascade",
                    "run_label": "MNQ Auction Cascade",
                    "selectable": True,
                })
    parent = str(cur.parent) if cur != root else None
    if parent and _browse_root_for(Path(parent)) is None:
        parent = None
    return {"path": str(cur), "parent": parent, "root": str(root), "entries": entries}


def classify_backtest_source(source: str) -> tuple[str, Path]:
    """Return ``(bundle|oof, resolved path)`` for a Backtest source.

    Raises ValueError if the path is outside the allowed trees or is not a
    serving bundle / OOF parquet.
    """
    from src.serving import is_serving_bundle

    if not source or not isinstance(source, str):
        raise ValueError("source is required")
    raw = Path(source)
    if not raw.is_absolute():
        raw = _REPO_ROOT / raw
    try:
        p = raw.resolve()
    except OSError as exc:
        raise ValueError(f"invalid source path: {exc}") from exc
    if _browse_root_for(p) is None:
        raise ValueError(
            "source must be under model_staging/, model_registry/, "
            "research/runs/, research/custom_test/, research/search/, "
            "or sources/labels/")
    from src.backtest.cascade_backtest import is_cascade_oof
    if p.is_file() and is_cascade_oof(p):
        return "cascade_oof", p
    if p.is_file() and p.suffix == ".parquet" and (
            p.name.startswith("oof_") or p.name == "oof.parquet"):
        return "oof", p
    if p.is_dir() and is_serving_bundle(p):
        return "bundle", p
    if p.is_dir() and (p / "oof.parquet").is_file():
        return "oof", p / "oof.parquet"
    if p.is_dir():
        oofs = sorted(p.glob("oof_*.parquet"))
        if len(oofs) == 1:
            return "oof", oofs[0]
        if len(oofs) > 1:
            names = ", ".join(f.name for f in oofs)
            raise ValueError(
                f"run dir has multiple OOF frames — pick one file: {names}")
    raise ValueError(
        f"not a serving bundle or oof.parquet: {p}")


def _objective_from_browse_path(p: Path) -> str | None:
    from src.common.run_layout import run_roots
    for root in run_roots():
        try:
            rel = p.relative_to(root.resolve())
            if rel.parts:
                oid = rel.parts[0]
                if ID_RE.match(oid):
                    return oid
        except (ValueError, OSError):
            pass
    parent = p.parent.name if p.is_file() else p.parent.name
    if p.is_dir() and ID_RE.match(p.parent.name or ""):
        return p.parent.name
    if ID_RE.match(parent or ""):
        return parent
    return None


def run_backtest_quicklook(*, source: str, tau: float = 0.8,
                           confidence_gate: bool = True,
                           max_positions: int = 1,
                           start: str | None = None, end: str | None = None,
                           objective_id: str | None = None,
                           sides: list[str] | None = None,
                           mv: float = 0.5,
                           mv_gate: bool = True,
                           p_reach_min: float = 0.75,
                           p_move_min: float = 0.5,
                           min_tgt_r: float = 0.25,
                           live_data: bool = False,
                           tbbo_data: bool = False,
                           mae_adjust: float = 0.0,
                           target_adjust: float = 0.0,
                           price_mode: str = "mid",
                           entry_latency_ms: float = 0.0,
                           exit_latency_ms: float = 0.0,
                           params: dict[str, Any] | None = None,
                           book: dict[str, Any] | None = None,
                           stack: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """In-memory path backtest: score feature parquets with a staged serving bundle.

    Inference is :mod:`src.serving` — identical to Live.

    *tbbo_data* / OCO: gated signals resolved on Databento TBBO with triple-barrier
    first-touch, MID or MKT pricing, optional entry/exit latency (ms).
    Default path uses 5s feature ``_close`` first-touch ("on close").
    """
    from src.external_models.runner import is_external_source, run_backtest as run_external
    if stack or is_external_source(source):
        return _sanitize_nan(run_external(
            source or None, start=start, end=end,
            params=params, book=book, stack=stack))
    if not source or not isinstance(source, str):
        raise ValueError("source is required")
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau (confidence) must be in [0, 1]")
    confidence_gate = bool(confidence_gate)
    max_positions = int(max_positions)
    if not 1 <= max_positions <= 50:
        raise ValueError("open positions must be in [1, 50]")
    for d in (start, end):
        if d and not _DATE_RE.match(d):
            raise ValueError("dates must be YYYY-MM-DD")
    if sides is not None:
        sides = [s for s in sides if s in ("up", "down")] or None
    if objective_id and not ID_RE.match(str(objective_id)):
        # Browse used to send the run timestamp (20260815T202407Z) as objective_id.
        objective_id = None
    elif objective_id:
        check_id(objective_id)
    pm = (price_mode or "mid").lower().strip()
    if pm not in ("mid", "mkt"):
        raise ValueError("price_mode must be 'mid' or 'mkt'")

    kind, path = classify_backtest_source(source)
    oid = objective_id or _objective_from_browse_path(path)

    if kind == "cascade_oof":
        from src.backtest.cascade_backtest import run_cascade_inmemory
        payload = run_cascade_inmemory(
            source=str(path),
            p_reach_min=float(p_reach_min), p_move_min=float(p_move_min),
            min_tgt_r=float(min_tgt_r),
            tau=float(tau), mv=float(mv),
            confidence_gate=bool(confidence_gate), mv_gate=bool(mv_gate),
            max_positions=int(max_positions),
            start=start or None, end=end or None,
            sides=sides, objective_id=oid)
        return _sanitize_nan(payload)

    # Historical Custom Test bundles include walk-forward oof.parquet.
    # Replay that (as-of fold models) instead of last-fold weights on the
    # whole window — last-fold leaks train dates (ADR-100). OCO still needs
    # a serving bundle to rescore.
    if kind == "bundle" and not (tbbo_data or live_data):
        nested = path / "oof.parquet"
        if nested.is_file():
            kind, path = "oof", nested

    if kind == "oof":
        from src.backtest.runner import run_predictions_inmemory
        payload = run_predictions_inmemory(
            source=str(path), tau=tau, confidence_gate=confidence_gate,
            max_positions=max_positions,
            start=start or None, end=end or None,
            objective_id=oid, sides=sides,
            max_sessions=BACKTEST_MAX_SESSIONS)
        cfg = payload.setdefault("config", {})
        cfg["source"] = str(path)
        cfg["model_path"] = str(path)
        cfg["run_label"] = (
            path.parent.name if path.name == "oof.parquet"
            else (f"{path.parent.parent.name}/{path.parent.name}/{path.stem[4:]}"
                  if path.name.startswith("oof_") else path.name))
        cfg["source_kind"] = "oof"
        if tbbo_data or live_data:
            cfg["weights_note"] = (
                "Search OOF replay uses the on-close path (no serving bundle to "
                "rescore for OCO/TBBO). Pick a staged/registry bundle for OCO.")
        return _sanitize_nan(payload)

    from src.backtest.weights_score import run_weights_backtest_inmemory
    payload = run_weights_backtest_inmemory(
        model_path=path, tau=tau, confidence_gate=confidence_gate,
        max_positions=max_positions,
        start=start or None, end=end or None,
        objective_id=oid, sides=sides,
        max_sessions=BACKTEST_MAX_SESSIONS,
        tbbo_data=bool(tbbo_data or live_data),
        mae_adjust=float(mae_adjust),
        target_adjust=float(target_adjust),
        price_mode=pm,
        entry_latency_ms=max(0.0, float(entry_latency_ms or 0)),
        exit_latency_ms=max(0.0, float(exit_latency_ms or 0)))
    return _sanitize_nan(payload)


# ── Research Director: systemd units + funnel/spend ──────────────────────────

def _systemctl(*args: str, sudo: bool = False, timeout: float = 10.0) -> tuple[int, str, str]:
    # NOTE: no extra flags on the sudo path — /etc/sudoers.d/vtech-webui matches the
    # exact command form `systemctl <verb> [--now] <unit>` (see deploy/webui/).
    argv = (["sudo", "-n", "systemctl", *args] if sudo
            else ["systemctl", "--no-pager", *args])
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)  # noqa: S603
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)


_SHOW_PROPS = ("Id", "ActiveState", "SubState", "UnitFileState", "NextElapseUSecRealtime",
               "LastTriggerUSecRealtime", "ExecMainStartTimestamp", "Description")


def _parse_show(block: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in block.strip().splitlines() if "=" in line)


def _unit_status_from_props(unit: str, props: dict[str, str]) -> dict[str, Any]:
    return {
        "unit": unit, "available": True,
        "active_state": props.get("ActiveState"), "sub_state": props.get("SubState"),
        "enabled": props.get("UnitFileState"),
        "next_run": props.get("NextElapseUSecRealtime") or None,
        "last_run": props.get("LastTriggerUSecRealtime") or props.get("ExecMainStartTimestamp") or None,
        "description": props.get("Description"),
    }


def unit_status(unit: str) -> dict[str, Any]:
    rc, out, err = _systemctl("show", unit, "--property=" + ",".join(_SHOW_PROPS))
    if rc != 0:
        return {"unit": unit, "available": False, "error": err.strip() or out.strip()}
    return _unit_status_from_props(unit, _parse_show(out))


def services_status() -> list[dict[str, Any]]:
    """Status of all watched units in ONE ``systemctl show u1 u2 …`` call — blocks come back
    in request order, separated by blank lines. This path is polled by the dashboard every few
    seconds, so batching turns 7 subprocess spawns per poll into 1. Falls back to per-unit calls
    on any parse surprise (missing systemd, block-count mismatch)."""
    rc, out, _ = _systemctl("show", *_SERVICES, "--property=" + ",".join(_SHOW_PROPS))
    blocks = [b for b in out.split("\n\n") if b.strip()] if rc == 0 else []
    if len(blocks) != len(_SERVICES):
        return [unit_status(u) for u in _SERVICES]
    return [_unit_status_from_props(u, _parse_show(b)) for u, b in zip(_SERVICES, blocks)]


def set_timer_enabled(enable: bool) -> dict[str, Any]:
    """Enable/disable the Research Director schedule. Tries plain systemctl (polkit may
    allow it), then ``sudo -n`` (needs the deploy/webui sudoers drop-in)."""
    verb = "enable" if enable else "disable"
    rc, out, err = _systemctl(verb, "--now", RD_TIMER)
    if rc != 0:
        rc, out, err = _systemctl(verb, "--now", RD_TIMER, sudo=True)
    if rc != 0:
        return {"ok": False, "error": (err or out).strip(),
                "hint": "install deploy/webui/vtech-webui-sudoers to grant the console "
                        "user passwordless systemctl for the vtech-research-director units"}
    return {"ok": True, "timer": unit_status(RD_TIMER)}


def _rd_db_snapshot() -> dict[str, Any]:
    """Funnel + spend from the research_director schema (same SQL as rd_dashboard.py)."""
    try:
        from src.common.db import get_pool
        pool = get_pool()
        with pool.connection() as conn:
            funnel = conn.execute(
                "SELECT program_id, state, n, n_7d, last_update "
                "FROM research_director.funnel_v ORDER BY n DESC LIMIT 25").fetchall()
            spend = conn.execute(
                "SELECT category, "
                " round(SUM(usd_debited) FILTER (WHERE ts > now() - interval '7 days')::numeric, 4),"
                " round(SUM(usd_debited) FILTER (WHERE ts > now() - interval '30 days')::numeric, 4) "
                "FROM research_director.gpu_budget_ledger GROUP BY category ORDER BY category").fetchall()
            llm = conn.execute(
                "SELECT agent, count(*), round(COALESCE(SUM(usd_cost),0)::numeric, 4) "
                "FROM research_director.llm_call WHERE ts > now() - interval '7 days' "
                "GROUP BY agent ORDER BY count(*) DESC").fetchall()
            pending = conn.execute(
                "SELECT experiment_id, scope, left(title, 80) FROM research_director.research_journal "
                "WHERE state = 'STAGED_PENDING_PROMOTION' ORDER BY updated_at DESC LIMIT 10").fetchall()
        return {"available": True,
                "funnel": [{"program_id": r[0], "state": r[1], "n": r[2], "n_7d": r[3],
                            "last_update": str(r[4])} for r in funnel],
                "spend": [{"category": r[0], "usd_7d": float(r[1] or 0), "usd_30d": float(r[2] or 0)}
                          for r in spend],
                "llm_calls_7d": [{"agent": r[0], "calls": r[1], "usd": float(r[2] or 0)} for r in llm],
                "awaiting_promotion": [{"experiment_id": r[0], "scope": r[1], "title": r[2]}
                                       for r in pending]}
    except Exception as exc:                              # noqa: BLE001 — DB is optional here
        logger.debug("rd db snapshot unavailable: %s", exc)
        return {"available": False, "error": str(exc)}


def _rd_config_summary() -> dict[str, Any]:
    try:
        from src.common.config_paths import runtime_path as _runtime_path
        cfg = yaml.safe_load(_runtime_path("research_director.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"available": False, "error": str(exc)}
    keep = {}
    for key in ("llm_usd_cap_cycle", "per_agent_usd_cap", "max_spec_cost_usd",
                "warn_frac", "degrade_frac"):
        val = cfg.get(key)
        if val is None and isinstance(cfg.get("cost"), dict):
            val = cfg["cost"].get(key)
        if val is not None:
            keep[key] = val
    panel = cfg.get("panel")
    if isinstance(panel, dict):
        keep["experts"] = panel.get("experts") or panel.get("lenses")
    return {"available": True, **keep}


def director_status() -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    try:
        from src.research_director.rd_enable import list_rd_targets
        targets = list_rd_targets()
    except Exception as exc:  # noqa: BLE001
        logger.debug("rd targets unavailable: %s", exc)
    return {"timer": unit_status(RD_TIMER), "service": unit_status(RD_SERVICE),
            "config": _rd_config_summary(), "db": _rd_db_snapshot(),
            "rd_targets": targets,
            "schedule_hint": "OnCalendar=*-*-* 00,06,12,18:30:00 UTC — one cycle per "
                             "RD-enabled objective with registry LATEST "
                             "(toggle in Objectives tab)"}


# ── overview ──────────────────────────────────────────────────────────────────

def vm_info() -> dict[str, Any]:
    try:
        from src.common.vm_config import get_vm_config
        cfg = get_vm_config()
        info = {"summary": str(cfg)}
        for attr in ("cpu_count", "total_ram_gb", "available_ram_gb", "swap_gb"):
            if hasattr(cfg, attr):
                info[attr] = getattr(cfg, attr)
        try:
            info["workers_default"] = cfg.workers_for()
        except Exception:                                 # noqa: BLE001
            pass
        return info
    except Exception as exc:                              # noqa: BLE001
        return {"error": str(exc)}


def build_argv_scoping(objective_id: str, dry_run: bool, budget: float | None) -> list[str]:
    argv = [sys.executable, "-m", "src.scoping.orchestrator", "--objective", check_id(objective_id)]
    if dry_run:
        argv.append("--dry-run")
    if budget is not None:
        argv += ["--budget", str(float(budget))]
    return argv


def build_argv_search(objective_id: str, families: list[str] | None, dry_run: bool,
                      tune_trials: int, no_ensemble: bool, manual_spec: str | None) -> list[str]:
    argv = [sys.executable, "-m", "src.search.controller", "--objective", check_id(objective_id)]
    if families:
        known = {f["name"] for f in catalog_families()} | {f["family_key"] for f in catalog_families()}
        bad = [f for f in families if f not in known]
        if bad:
            raise ValueError(f"unknown families: {bad}")
        argv += ["--families", ",".join(families)]
    if dry_run:
        argv.append("--dry-run")
    if tune_trials:
        argv += ["--tune-trials", str(int(tune_trials))]
    if no_ensemble:
        argv.append("--no-ensemble")
    if manual_spec:
        from src.common.config_paths import resolve_user_path
        spec = resolve_user_path(manual_spec, kind="specs").resolve()
        cfg_root = (_REPO_ROOT / "config").resolve()
        if not str(spec).startswith(str(cfg_root)) or not spec.exists():
            raise ValueError("manual spec must be an existing file under config/")
        argv += ["--manual-spec", str(spec)]
    return argv


def build_argv_director_cycle(dry_run: bool) -> tuple[list[str], dict[str, str]]:
    argv = [sys.executable, "-m", "src.research_director.run"]
    if dry_run:
        argv.append("--dry-run")
    return argv, {"PYTHONPATH": str(_REPO_ROOT)}
