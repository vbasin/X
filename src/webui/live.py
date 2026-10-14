"""Live streamer + model-prediction snapshot for the web console.

Each snapshot scores the latest 5-second bar with the active objective's
staged or registry serving bundle, and reports feature NaN coverage plus
the prediction. When the live-features heartbeat is fresh the feature frame
is the hive already on disk. A stale runtime extends that hive day;
a full ``build_date`` runs only when the day is missing.

Model resolution (first hit wins):

1. ``VTECH_LIVE_MODEL`` env (explicit path override)
2. Newest bundle under ``data/models/model_registry/<active_objective>/``
3. Newest bundle under ``data/models/model_staging/<active_objective>/``
4. Newest bundle under either tree for any objective (with a warning note)

The webui runs :func:`build_snapshot` on a background cadence and caches the
result in ``_STATE``; ``/api/live/streamer`` returns that cache plus cached
:func:`streamer_health` (refreshed in the background loop). Bundle + objective
labels reload when the resolved path or active objective pin changes.
"""
from __future__ import annotations

from src.common.data_paths import data_root, live_dir, model_registry_dir, model_staging_dir, native_tf_dir, webui_dir

import copy
import contextlib
import io
import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.webui.live")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = data_root()  # re-read each process start; tests patch data_paths/env
_STAGING_ROOT = model_staging_dir()
_REGISTRY_ROOT = model_registry_dir()
# Operator selection for the Live tab (mode=auto|manual + optional path). Survives restarts.
_SELECTION_PATH = Path(os.environ.get(
    "VTECH_LIVE_SELECTION", str(webui_dir() / "live_model.json")))
_SNAPSHOT_CACHE_PATH = Path(os.environ.get(
    "VTECH_LIVE_SNAPSHOT_CACHE", str(webui_dir() / "live_snapshot_cache.json")))
_LIVE_AGE_S = 15.0                       # tbbo fresher than this ⇒ streamer "live"
_DEGRADED_AGE_S = 60.0                   # soft lag threshold for table health
# Health probes MUST be time-bounded. Unbounded ``max(ts_event)`` on the tbbo
# hypertable visits every chunk (including compressed history) and never
# returns — the Live tab then stays ``warming`` forever. bbo_1s has no
# ts_recv index and daily chunks, so even a 2-minute filter seq-scans today.
_HEALTH_STMT_TIMEOUT = "3000"  # ms — connection options take integer milliseconds
_TBBO_FRESH_SQL = """
SELECT max(ts_event),
       count(*) FILTER (WHERE ts_event >= now() - interval '10 seconds'),
       count(*) FILTER (WHERE ts_event >= now() - interval '2 minutes')
FROM databento.tbbo
WHERE ts_event >= now() - interval '1 hour'
"""
_STATS_FRESH_SQL = """
SELECT max(ts_event),
       count(*) FILTER (WHERE ts_event >= now() - interval '10 seconds'),
       count(*) FILTER (WHERE ts_event >= now() - interval '2 minutes')
FROM databento.statistics
WHERE ts_event >= now() - interval '2 hours'
"""
_STREAMER_UNIT = "vtech-live-streamer.service"
_STREAMER_CMDLINE_MARK = "src.acquisition.live_streamer"
_RAW_LIVE_DIR = data_root() / "sources" / "raw" / "live"

# Static universe the live streamer subscribes to (must match live_streamer.subscribe_live_feeds).
_LIVE_SUBSCRIPTIONS: list[dict[str, str]] = [
    {"schema": "tbbo", "symbols": "NQ.FUT", "stype_in": "parent", "target_table": "databento.tbbo"},
    {"schema": "bbo-1s", "symbols": "NQ.FUT", "stype_in": "parent", "target_table": "databento.bbo_1s"},
    {"schema": "definition", "symbols": "NQ_OPTION_PARENTS", "stype_in": "parent",
     "target_table": "databento.definitions"},
    {"schema": "bbo-1s", "symbols": "NQ_OPTION_PARENTS", "stype_in": "parent",
     "target_table": "databento.bbo_1s"},
    {"schema": "statistics", "symbols": "NQ.FUT", "stype_in": "parent",
     "target_table": "databento.statistics"},
    {"schema": "tbbo", "symbols": "ES.FUT", "stype_in": "parent", "target_table": "databento.tbbo"},
]

# cached last-good prediction snapshot (updated by the background builder)
_STATE: dict[str, Any] = {"status": "starting", "prediction": None}
_LOCK = threading.Lock()
# TBBO DATA blotter lives outside _STATE so snapshot cache reloads cannot wipe
# a multi-minute paper replay, and HTTP handlers never wait on that replay.
_TICKS_CACHE: dict[str, Any] = {}
_TICKS_JOB: dict[str, Any] = {"key": None, "gen": 0}
_SNAPSHOT_CACHE_META: dict[str, int | None] = {"mtime_ns": None, "version": None}
_STREAMER_HEALTH: dict[str, Any] = {
    "status": "warming",
    "tables": {"tbbo": None, "bbo_1s": None, "statistics": None},
    "table_detail": {},
    "tbbo_rows_10s": 0,
    "tbbo_age_s": None,
    "live": False,
    "process": None,
    "systemd": None,
    "connections": [],
    "subscriptions": list(_LIVE_SUBSCRIPTIONS),
    "raw": None,
    "disk": None,
    "error": "warming: health probe has not run yet",
    "checked_at": None,
}

# signal exporter watermark (bridge to native broker script — see _export_signals)
_SIG_STATE: dict[str, Any] = {"last_ts": None, "path": None}
_SIG_LOCK = threading.Lock()

# lazily-initialised singletons; invalidated when resolved model/objective changes
_BUNDLE: dict[str, Any] | None = None
_ENGINE = None


# ── active objective + model resolution ───────────────────────────────────────

def active_objective_id() -> str | None:
    """Pinned objective from config.yaml (same source as the console header)."""
    try:
        import yaml
        cfg = yaml.safe_load((_REPO_ROOT / "config.yaml").read_text()) or {}
        oid = cfg.get("objective_id")
        return str(oid) if oid else None
    except Exception:  # noqa: BLE001
        return None


def _parse_horizon_seconds(horizon: str | None, default: int = 600) -> int:
    """``'10m'`` / ``'600s'`` / ``'1h'`` → seconds."""
    if not horizon:
        return default
    s = str(horizon).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", s)
    if not m:
        return default
    n, unit = float(m.group(1)), (m.group(2) or "s")
    mult = {"s": 1, "m": 60, "h": 3600}[unit]
    return max(1, int(n * mult))


def _is_serving_bundle(d: Path) -> bool:
    """Delegate to shared serving contract (Live + Backtest SSOT)."""
    from src.serving import is_serving_bundle
    return is_serving_bundle(d)


def bundle_run_label(path: str | Path) -> str:
    """Identity one folder up from the model files: ``<objective>/<run_dir>``.

    Serving files live in ``…/model_staging/<objective>/<candidate>_<run_ts>/``.
    The leaf directory name is the unique search timestamp operators need.
    """
    p = Path(path)
    parent, name = p.parent.name, p.name
    if not name:
        return str(p)
    if parent in {"model_staging", "model_registry", "models", ""}:
        return name
    return f"{parent}/{name}"


def _bundle_mtime(d: Path) -> float:
    for name in ("finalize_manifest.json", "model.ubj", "feature_schema.json"):
        p = d / name
        if p.exists():
            return p.stat().st_mtime
    return d.stat().st_mtime


def list_serving_bundles(objective_id: str | None = None) -> list[dict[str, Any]]:
    """Discover promotable/staged model dirs, newest first."""
    roots: list[tuple[str, Path]] = []
    for label, root in (("registry", _REGISTRY_ROOT), ("staging", _STAGING_ROOT)):
        if not root.is_dir():
            continue
        if objective_id:
            cand = root / objective_id
            if cand.is_dir():
                roots.append((label, cand))
        else:
            for sub in root.iterdir():
                if sub.is_dir():
                    roots.append((label, sub))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, base in roots:
        # base is either …/<objective_id> or we walk children
        dirs = [base] if _is_serving_bundle(base) else [
            p for p in base.iterdir() if p.is_dir() and _is_serving_bundle(p)
        ]
        for d in dirs:
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            manifest: dict[str, Any] = {}
            mp = d / "finalize_manifest.json"
            if mp.exists():
                try:
                    manifest = json.loads(mp.read_text())
                except Exception:  # noqa: BLE001
                    manifest = {}
            oid = (manifest.get("objective_id")
                   or (d.parent.name if d.parent.name not in ("model_staging", "model_registry")  # basename under models/
                       else None))
            cand = manifest.get("candidate_id") or d.name
            fam = manifest.get("family")
            ver = manifest.get("version") or d.name
            score = manifest.get("search_score") or {}
            exp = score.get("net_expectancy")
            prec = score.get("precision_at_tau")
            run_label = bundle_run_label(d)
            bits = [run_label, source]
            if fam:
                bits.append(str(fam))
            if exp is not None:
                try:
                    bits.append(f"exp {float(exp):.1f}")
                except (TypeError, ValueError):
                    pass
            elif prec is not None:
                try:
                    bits.append(f"P@τ {float(prec):.2f}")
                except (TypeError, ValueError):
                    pass
            pol = manifest.get("inference_policy")
            if not isinstance(pol, dict) or not pol:
                ip = d / "inference_policy.json"
                if ip.exists():
                    try:
                        loaded = json.loads(ip.read_text())
                    except (OSError, json.JSONDecodeError):
                        loaded = None
                    pol = loaded if isinstance(loaded, dict) else None
            if pol:
                tau_p = pol.get("tau")
                win = f"{pol.get('trading_start') or '?' }–{pol.get('trading_end') or '?'}"
                bits.append(win)
                if tau_p is not None:
                    try:
                        bits.append(f"τ {float(tau_p):.2f}")
                    except (TypeError, ValueError):
                        pass
                sides_p = pol.get("sides") or []
                if sides_p:
                    bits.append("/".join(str(s) for s in sides_p))
            out.append({
                "path": str(d.resolve()),
                "name": d.name,
                "source": source,
                "objective_id": oid,
                "candidate_id": cand,
                "family": fam,
                "version": ver,
                "run_label": run_label,
                "mtime": _bundle_mtime(d),
                "search_score": score or None,
                "inference_policy": pol,
                "label": " · ".join(str(x) for x in bits if x),
            })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# ── operator model selection (Live tab dropdown) ──────────────────────────────

def get_selection() -> dict[str, Any]:
    """Persisted Live-tab selection: ``{mode: auto|manual, path?: str}``."""
    try:
        if _SELECTION_PATH.exists():
            d = json.loads(_SELECTION_PATH.read_text())
            if isinstance(d, dict) and d.get("mode") in ("auto", "manual"):
                return {"mode": d["mode"], "path": d.get("path")}
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_selection_read_failed: %s", exc)
    return {"mode": "auto", "path": None}


def _validate_model_path(path: str | Path) -> Path:
    """Resolve *path* and ensure it is a serving bundle under staging/registry."""
    from src.serving import validate_bundle_path
    return validate_bundle_path(path)


def set_selection(*, mode: str = "auto", path: str | Path | None = None) -> dict[str, Any]:
    """Persist Live model selection and invalidate the in-process cache.

    * ``mode='auto'`` — follow active objective's newest staged/registry champion.
    * ``mode='manual'`` — pin a concrete bundle path (must be under staging/registry).
    """
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "manual"):
        raise ValueError("mode must be 'auto' or 'manual'")
    payload: dict[str, Any] = {"mode": mode, "path": None, "updated_at":
                               datetime.now(timezone.utc).isoformat()}
    if mode == "manual":
        if not path:
            raise ValueError("manual mode requires a path")
        p = _validate_model_path(path)
        payload["path"] = str(p)
    _SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SELECTION_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(_SELECTION_PATH)
    invalidate_cache()
    logger.info("live_selection_set mode=%s path=%s", mode, payload.get("path"))
    return get_selection()


def resolve_live_model(
    objective_id: str | None = None,
    *,
    override: str | Path | None = None,
) -> dict[str, Any]:
    """Pick the serving bundle for live scoring.

    Order: env ``VTECH_LIVE_MODEL`` → operator manual selection → auto
    (newest for active objective → any-objective fallback).
    """
    env_override = override or os.environ.get("VTECH_LIVE_MODEL")
    if env_override:
        p = Path(env_override)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if _is_serving_bundle(p):
            # Enrich with manifest objective when possible.
            oid_m = None
            mp = p / "finalize_manifest.json"
            if mp.exists():
                try:
                    oid_m = json.loads(mp.read_text()).get("objective_id")
                except Exception:  # noqa: BLE001
                    pass
            return {
                "path": str(p.resolve()), "name": p.name, "source": "env",
                "objective_id": oid_m or objective_id, "version": p.name,
                "candidate_id": p.name, "label": f"{p.name} · env",
                "selection_mode": "env",
                "note": "VTECH_LIVE_MODEL override",
            }
        raise FileNotFoundError(
            f"VTECH_LIVE_MODEL={env_override} is not a serving bundle "
            f"(need feature_schema.json + model artifact)")

    # Operator pin from the Live tab dropdown.
    sel = get_selection()
    if sel.get("mode") == "manual" and sel.get("path"):
        try:
            p = _validate_model_path(sel["path"])
            # Prefer the catalog row if present (label / scores).
            for b in list_serving_bundles(None):
                if Path(b["path"]).resolve() == p:
                    return {**b, "selection_mode": "manual", "note": None}
            return {
                "path": str(p), "name": p.name, "source": "manual",
                "objective_id": objective_id, "version": p.name,
                "candidate_id": p.name, "label": p.name,
                "selection_mode": "manual", "note": None,
            }
        except ValueError as exc:
            logger.warning("live_manual_selection_invalid (%s) — falling back to auto", exc)

    oid = objective_id if objective_id is not None else active_objective_id()
    if oid:
        # Prefer registry LATEST for this objective (production pin).
        try:
            from src.serving.promote import resolve_latest_bundle
            latest = resolve_latest_bundle(oid)
            if latest is not None:
                for b in list_serving_bundles(oid):
                    if Path(b["path"]).resolve() == latest.resolve():
                        return {**b, "selection_mode": "auto",
                                "note": "registry LATEST", "source": "registry"}
                return {
                    "path": str(latest.resolve()), "name": latest.name,
                    "source": "registry", "objective_id": oid,
                    "version": latest.name, "candidate_id": latest.name,
                    "label": f"{latest.name} · registry LATEST",
                    "selection_mode": "auto", "note": "registry LATEST",
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("live_registry_latest_skip: %s", exc)
        for b in list_serving_bundles(oid):
            return {**b, "selection_mode": "auto", "note": None}
        # No bundle for this objective — fall back with a clear warning.
        any_b = list_serving_bundles(None)
        if any_b:
            fb = any_b[0]
            return {
                **fb,
                "selection_mode": "auto",
                "note": (f"no staged/registry model for objective '{oid}'; "
                         f"using {fb['source']}/{fb['objective_id']}/{fb['name']}"),
            }
        raise FileNotFoundError(
            f"no serving bundle for objective '{oid}' under "
            f"model_staging/ or model_registry/")

    any_b = list_serving_bundles(None)
    if any_b:
        fb = any_b[0]
        return {
            **fb, "selection_mode": "auto",
            "note": "no active objective_id in config.yaml; using newest staged model",
        }
    raise FileNotFoundError("no serving bundles found under model_staging/ or model_registry/")


def objective_trade_params(objective_id: str | None = None) -> dict[str, Any]:
    """Barrier distances + horizon + default τ + session window from the objective doc."""
    oid = objective_id or active_objective_id() or "nq-50pt"
    from src.common.instruments import point_value_from_objective, primary_symbol

    labels: dict[str, Any] = {}
    session: dict[str, Any] = {}
    horizon = "10m"
    primary_tau = 0.8
    obj: dict[str, Any] | None = None
    try:
        from src.objectives.loader import load_objective
        from src.evaluation.kpi_criteria import primary_tau as _ptau
        obj = load_objective(oid)
        labels = obj.get("labels") or {}
        session = obj.get("session") or {}
        horizon = (obj.get("horizon") or {}).get("max_trade_horizon") or horizon
        pt = _ptau(obj)
        if pt is not None:
            primary_tau = float(pt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("objective_trade_params_fallback oid=%s err=%s", oid, exc)
    env_tau = os.environ.get("VTECH_LIVE_TAU")
    tau = float(env_tau) if env_tau else primary_tau
    env_hz = os.environ.get("VTECH_LIVE_HORIZON_S")
    hz = int(env_hz) if env_hz else _parse_horizon_seconds(str(horizon))
    return {
        "objective_id": oid,
        "move_threshold": float(labels.get("move_threshold", 50)),
        "mae_tolerance": float(labels.get("mae_tolerance", 5)),
        "horizon_s": hz,
        "tau": tau,
        "scheme": labels.get("scheme") or "detection_2",
        "trading_start": session.get("trading_start") or "08:30",
        "trading_end": session.get("trading_end") or "14:45",
        "training_start": session.get("training_start") or "08:00",
        "training_end": session.get("training_end") or "14:45",
        "timezone": session.get("timezone") or "America/Chicago",
        "instrument": primary_symbol(obj) or "NQ",
        "point_value": point_value_from_objective(obj, default=20.0),
    }


def _parse_hhmm(s: str) -> tuple[int, int]:
    parts = str(s).strip().split(":")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def session_context(objective_id: str | None = None,
                    *, now: datetime | None = None,
                    roll_at: str = "trading_start") -> dict[str, Any]:
    """Calendar trading-session date + open/closed state in the objective TZ.

    Uses the objective's session window (not UTC midnight). Pre-open (local time
    before *roll_at*) rolls back to the previous calendar day so the overnight
    blotter still shows the day that just closed.

    *roll_at* is ``trading_start`` (08:30, blotter / Live) or ``training_start``
    (08:00, feature parquet date). Using 08:30 for the parquet filename left
    08:00–08:29 CT on yesterday's file and duplicated them onto today's.
    """
    from datetime import date, time as dtime, timedelta
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore

    trade = objective_trade_params(objective_id)
    tz_name = trade["timezone"]
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("America/Chicago")
        tz_name = "America/Chicago"
    now_local = (now or datetime.now(timezone.utc)).astimezone(tz)
    roll_key = roll_at if roll_at in ("trading_start", "training_start") else "trading_start"
    rh, rm = _parse_hhmm(trade[roll_key])
    th, tm = _parse_hhmm(trade["trading_start"])
    eh, em = _parse_hhmm(trade["trading_end"])
    t_roll = dtime(rh, rm)
    t_trade_open = dtime(th, tm)
    t_end = dtime(eh, em)
    local_t = now_local.timetz().replace(tzinfo=None)

    # Before *roll_at* → previous calendar day is still the active file/blotter day.
    if local_t < t_roll:
        session_date = (now_local - timedelta(days=1)).date()
    else:
        session_date = now_local.date()

    # Walk back over weekends/holidays when the rolled date isn't a trading day.
    try:
        from src.common.cme_holidays import is_trading_day
        guard = 0
        while not is_trading_day(session_date) and guard < 10:
            session_date -= timedelta(days=1)
            guard += 1
    except Exception:  # noqa: BLE001
        pass

    in_session = (session_date == now_local.date()
                  and t_trade_open <= local_t <= t_end)
    session_closed = (session_date < now_local.date()
                      or (session_date == now_local.date() and local_t > t_end))
    return {
        "session_date": session_date,
        "timezone": tz_name,
        "trading_start": trade["trading_start"],
        "trading_end": trade["trading_end"],
        "training_start": trade["training_start"],
        "training_end": trade["training_end"],
        "in_session": in_session,
        "session_closed": session_closed,
        "now_local": now_local.isoformat(),
    }


def _parquet_path(session_date) -> Path:
    d = str(session_date)[:10]
    try:
        y, m, _ = d.split("-")
        return native_tf_dir("5s", year=int(y), month=int(m))
    except Exception:
        return native_tf_dir("5s")


def _load_feature_frame(session_date):
    """Load one session day from the 5s monthly parts; empty if missing."""
    import pandas as pd

    from src.features.hive_5s import load_session_day

    p = _parquet_path(session_date)
    # A file path is the test override (production returns the hive directory).
    if isinstance(p, Path) and p.is_file():
        try:
            return pd.read_parquet(p), p
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_parquet_read_failed path=%s err=%s", p, exc)
            return pd.DataFrame(), p
    try:
        df = load_session_day(session_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_parquet_read_failed path=%s err=%s", p, exc)
        return pd.DataFrame(), p
    return df, p


def _previous_trading_day(d):
    from datetime import timedelta
    try:
        from src.common.cme_holidays import is_trading_day
    except Exception:  # noqa: BLE001
        def is_trading_day(x):  # type: ignore
            return x.weekday() < 5
    cur = d - timedelta(days=1)
    for _ in range(14):
        if is_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    return cur


def resolve_feature_day(objective_id: str | None = None, *,
                        force_rebuild: bool = False) -> dict[str, Any]:
    """Pick the feature frame for Live scoring + blotter.

    * **Runtime heartbeat fresh** — read the hive. Do not rebuild.
    * **In session, runtime down** — ``extend_date`` (windowed when the hive
      day already has rows; ``build_date`` only if that day is empty).
      ``force_rebuild`` is the explicit full replace.
    * **After session close** — **do not** rebuild (avoids empty overwrite);
      load the completed day parquet so the blotter stays visible.
    * **Empty day** — walk back up to ~10 trading days for the latest non-empty
      frame (weekend / holiday / missing build).
    """
    import pandas as pd

    ctx = session_context(objective_id)
    session_date = ctx["session_date"]
    source = "disk"
    note = None
    path = _parquet_path(session_date)
    df = pd.DataFrame()

    # The always-on runtime writes the hive, then a fresh heartbeat, then scores.
    # Rebuilding here would load the whole options BBO day again — that fetch is
    # what OOM-kills the web UI while a chart request is in flight.
    fresh = False
    if not force_rebuild:
        try:
            from src.features.live_runtime import is_runtime_fresh
            fresh = is_runtime_fresh()
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_runtime_fresh_check_failed: %s", exc)
    if fresh:
        df, path = _load_feature_frame(session_date)
        if df is not None and not df.empty:
            return {
                "df": df,
                "path": str(path) if path else None,
                "session_date": session_date,
                "source": "live_runtime",
                "note": None,
                "context": ctx,
                "n_rows": int(len(df)),
            }

    # Refresh only while the session is open (or forced). After close, disk is
    # the source of truth — a live rebuild with no new ticks can yield 0 rows.
    # A full ``build_date`` loads the whole primary-filtered options BBO day
    # (~40M rows). That scan is what filled the web UI cgroup and dropped
    # in-flight chart requests. ``extend_date`` recomputes the lag, and
    # bootstraps with ``build_date`` only when the hive day is empty.
    if ctx["in_session"] or force_rebuild:
        try:
            eng = _engine()
            with contextlib.redirect_stdout(io.StringIO()):
                if force_rebuild:
                    built = eng.build_date(session_date)
                else:
                    built = eng.extend_date(session_date)
            df, path = _load_feature_frame(session_date)
            if built:
                path = Path(built)
            if not df.empty:
                source = "live_build" if force_rebuild else "live_extend"
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_build_date_failed date=%s err=%s", session_date, exc)
            note = f"live rebuild failed: {type(exc).__name__}: {exc}"

    if df.empty:
        df, path = _load_feature_frame(session_date)
        if not df.empty:
            source = "disk"
            if ctx["session_closed"]:
                note = (f"session closed ({ctx['trading_end']} {ctx['timezone']}) — "
                        f"showing completed blotter for {session_date}")
            elif not ctx["in_session"]:
                note = f"pre-open / outside hours — showing blotter for {session_date}"

    # Walk back if still empty (no data for this calendar day).
    if df.empty:
        d = session_date
        for _ in range(10):
            d = _previous_trading_day(d)
            df, path = _load_feature_frame(d)
            if not df.empty:
                session_date = d
                source = "disk_fallback"
                note = (f"no features for session day — showing last available "
                        f"trading day {session_date}")
                break

    return {
        "df": df,
        "path": str(path) if path else None,
        "session_date": session_date,
        "source": source,
        "note": note,
        "context": ctx,
        "n_rows": int(len(df)),
    }


# ── model bundle load (shared src.serving — same as Backtest) ────────────────

def _load_bundle(model_dir: Path, *, validate: bool = True) -> dict[str, Any]:
    """Load feature schema + booster + scaler + dir_threshold via src.serving."""
    from src.serving import load_serving_bundle
    return load_serving_bundle(model_dir, validate=validate)


def _bundle() -> dict[str, Any]:
    """Return the loaded serving bundle, reloading when path/selection changes."""
    global _BUNDLE
    pin = active_objective_id()
    resolved = resolve_live_model(pin)
    path = resolved["path"]
    # Barriers follow the *model's* objective when known; else the config pin.
    trade_oid = resolved.get("objective_id") or pin
    sel_mode = resolved.get("selection_mode") or get_selection().get("mode")
    if (_BUNDLE is None
            or _BUNDLE.get("path") != path
            or _BUNDLE.get("active_objective") != pin
            or _BUNDLE.get("trade_objective") != trade_oid
            or _BUNDLE.get("selection_mode") != sel_mode):
        logger.info("live_model_load path=%s pin=%s trade_obj=%s mode=%s source=%s",
                    path, pin, trade_oid, sel_mode, resolved.get("source"))
        # Env override may point outside staging/registry; skip path root check.
        validate = sel_mode != "env"
        loaded = _load_bundle(Path(path), validate=validate)
        loaded["active_objective"] = pin
        loaded["trade_objective"] = trade_oid
        loaded["selection_mode"] = sel_mode
        loaded["resolved"] = resolved
        loaded["trade"] = objective_trade_params(trade_oid)
        _BUNDLE = loaded
    return _BUNDLE


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from src.features.engine import FeatureEngine
        _ENGINE = FeatureEngine()
    return _ENGINE


def invalidate_cache() -> None:
    """Drop the loaded model + snapshot (e.g. after promote / objective pin change)."""
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None
        _TICKS_CACHE.clear()
        _TICKS_JOB["gen"] = int(_TICKS_JOB.get("gen") or 0) + 1
        _TICKS_JOB["key"] = None
    _consume_snapshot_cache()
    _replace_state({"status": "starting", "prediction": None})


def _replace_state(snap: dict[str, Any]) -> None:
    with _LOCK:
        _STATE.clear()
        _STATE.update(snap)


def _consume_snapshot_cache() -> None:
    """Mark the current persisted cache as seen so invalidation doesn't revive stale state."""
    mtime_ns: int | None = None
    cache_version: int | None = None
    try:
        st = _SNAPSHOT_CACHE_PATH.stat()
        mtime_ns = int(st.st_mtime_ns)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("live_snapshot_cache_stat_failed path=%s err=%s", _SNAPSHOT_CACHE_PATH, exc)
    if mtime_ns is not None:
        try:
            raw = json.loads(_SNAPSHOT_CACHE_PATH.read_text())
            if isinstance(raw, dict) and raw.get("version") is not None:
                cache_version = int(raw.get("version"))
        except Exception:
            cache_version = None
    with _LOCK:
        _SNAPSHOT_CACHE_META["mtime_ns"] = mtime_ns
        _SNAPSHOT_CACHE_META["version"] = cache_version


def _write_snapshot_cache(snap: dict[str, Any]) -> None:
    payload = {"version": time.time_ns(), "snapshot": snap}
    try:
        _SNAPSHOT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SNAPSHOT_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(_SNAPSHOT_CACHE_PATH)
        mtime_ns = _SNAPSHOT_CACHE_PATH.stat().st_mtime_ns
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_snapshot_cache_write_failed path=%s err=%s", _SNAPSHOT_CACHE_PATH, exc)
        return
    with _LOCK:
        _SNAPSHOT_CACHE_META["mtime_ns"] = int(mtime_ns)
        _SNAPSHOT_CACHE_META["version"] = int(payload["version"])


def _commit_snapshot(snap: dict[str, Any], *, persist_cache: bool = True) -> dict[str, Any]:
    _replace_state(snap)
    if persist_cache:
        _write_snapshot_cache(snap)
    return snap


def _refresh_snapshot_state_from_cache(*, force: bool = False) -> bool:
    """Load snapshot from disk cache into ``_STATE`` when the file changed.

    Returns True if state was updated (or already matched the file).
    *force*: reload even if this process already consumed the current mtime
    (used after WebUI restart when ``vtech-live-features`` owns scoring and
    will not rewrite the cache until the next bar).
    """
    try:
        st = _SNAPSHOT_CACHE_PATH.stat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("live_snapshot_cache_stat_failed path=%s err=%s", _SNAPSHOT_CACHE_PATH, exc)
        return False

    with _LOCK:
        seen_mtime = _SNAPSHOT_CACHE_META.get("mtime_ns")
        seen_version = _SNAPSHOT_CACHE_META.get("version")
    if not force and seen_mtime == int(st.st_mtime_ns):
        return True

    cache_version: int | None = None
    try:
        raw = json.loads(_SNAPSHOT_CACHE_PATH.read_text())
        if not isinstance(raw, dict):
            raise ValueError("cache payload must be a JSON object")
        snap = raw.get("snapshot", raw)
        if not isinstance(snap, dict):
            raise ValueError("cache snapshot must be a JSON object")
        raw_version = raw.get("version")
        cache_version = int(raw_version) if raw_version is not None else None
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _SNAPSHOT_CACHE_META["mtime_ns"] = int(st.st_mtime_ns)
        logger.warning("live_snapshot_cache_read_failed path=%s err=%s", _SNAPSHOT_CACHE_PATH, exc)
        return False

    if (
        not force
        and seen_version is not None
        and seen_version == cache_version
        and seen_mtime == int(st.st_mtime_ns)
    ):
        return True
    with _LOCK:
        _STATE.clear()
        _STATE.update(snap)
        _SNAPSHOT_CACHE_META["mtime_ns"] = int(st.st_mtime_ns)
        _SNAPSHOT_CACHE_META["version"] = cache_version
    return True


def adopt_external_snapshot_cache() -> bool:
    """Force-load the on-disk snapshot written by ``vtech-live-features``.

    Call this when the always-on runtime is healthy so WebUI does not stay
    stuck in ``status=starting`` after restart (invalidate would otherwise
    mark the current cache file as already-seen and never re-read it).
    """
    ok = _refresh_snapshot_state_from_cache(force=True)
    if ok:
        with _LOCK:
            st = _STATE.get("status")
        logger.info("live_adopt_snapshot_cache status=%s", st)
    return ok


def snapshot_status() -> str | None:
    """Current in-process snapshot status (``ok`` / ``starting`` / ``warming`` / …)."""
    with _LOCK:
        return _STATE.get("status")


# ── streamer-health cache (read per request; refresh in background loop) ───────

def streamer_health() -> dict[str, Any]:
    """Return the in-memory cached streamer-health payload."""
    with _LOCK:
        return copy.deepcopy(_STREAMER_HEALTH)


def _age_s(ts: datetime | None) -> float | None:
    if ts is None:
        return None
    dt = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - dt).total_seconds(), 1)


def _table_status(age: float | None, rows_10s: int, *, kind: str = "tick") -> str:
    """Classify table freshness.

    ``kind="tick"`` uses tight live thresholds (tbbo / bbo_1s).
    ``kind="stats"`` is looser — CME statistics update far less often.
    """
    if age is None:
        return "unknown"
    live_lim = _LIVE_AGE_S if kind == "tick" else 600.0          # 10 min
    deg_lim = _DEGRADED_AGE_S if kind == "tick" else 3600.0      # 60 min
    if age <= live_lim:
        return "healthy"
    if age <= deg_lim:
        return "degraded"
    return "stale"


def _find_streamer_pids() -> list[int]:
    """Return PIDs whose cmdline contains the live streamer module mark."""
    pids: list[int] = []
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except (OSError, PermissionError):
                continue
            cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
            if _STREAMER_CMDLINE_MARK in cmd:
                pids.append(int(entry.name))
    except OSError:
        pass
    return sorted(pids)


def _read_proc_process(pid: int) -> dict[str, Any] | None:
    """Collect process details from /proc for one PID."""
    base = Path(f"/proc/{pid}")
    if not base.exists():
        return None
    try:
        cmdline = (base / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        status_txt = (base / "status").read_text(errors="replace")
        stat_txt = (base / "stat").read_text(errors="replace")
    except (OSError, PermissionError):
        return None

    rss_kb = None
    state = None
    for line in status_txt.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rss_kb = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith("State:"):
            state = line.split(":", 1)[1].strip()

    # uptime via /proc/stat starttime (field 22) and /proc/uptime
    etime_s = None
    try:
        rparen = stat_txt.rfind(")")
        rest = stat_txt[rparen + 2 :].split()
        start_ticks = int(rest[19])  # field 22 of /proc/<pid>/stat
        try:
            clk = os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError, AttributeError):
            clk = 100
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        etime_s = max(0.0, uptime - (start_ticks / float(clk)))
    except (IndexError, ValueError, OSError):
        etime_s = None

    open_files: list[str] = []
    try:
        for fd in (base / "fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if "stream" in target or "raw/live" in target or target.endswith(".dbn") or target.endswith(".dbn.zst"):
                open_files.append(target)
    except (OSError, PermissionError):
        pass

    return {
        "pid": pid,
        "cmdline": cmdline[:240],
        "state": state,
        "rss_mb": round(rss_kb / 1024.0, 1) if rss_kb is not None else None,
        "etime_s": round(etime_s, 0) if etime_s is not None else None,
        "etime": _fmt_etime(etime_s) if etime_s is not None else None,
        "open_raw_files": open_files,
    }


def _fmt_etime(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _probe_systemd_unit(unit: str = _STREAMER_UNIT) -> dict[str, Any]:
    """Best-effort systemctl show for the live streamer unit."""
    import subprocess

    out: dict[str, Any] = {
        "unit": unit,
        "active_state": None,
        "sub_state": None,
        "main_pid": None,
        "memory_current": None,
        "n_restarts": None,
        "result": None,
        "available": False,
    }
    try:
        proc = subprocess.run(
            [
                "systemctl", "show", unit,
                "--property=ActiveState,SubState,MainPID,MemoryCurrent,NRestarts,Result,FragmentPath",
                "--no-page",
            ],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            out["error"] = (proc.stderr or "systemctl show failed").strip()[:200]
            return out
        for line in proc.stdout.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "ActiveState":
                out["active_state"] = v
            elif k == "SubState":
                out["sub_state"] = v
            elif k == "MainPID":
                try:
                    out["main_pid"] = int(v)
                except ValueError:
                    out["main_pid"] = None
            elif k == "MemoryCurrent":
                try:
                    n = int(v)
                    out["memory_current"] = n
                    out["memory_mb"] = round(n / (1024 * 1024), 1) if n >= 0 else None
                except ValueError:
                    pass
            elif k == "NRestarts":
                try:
                    out["n_restarts"] = int(v)
                except ValueError:
                    pass
            elif k == "Result":
                out["result"] = v
            elif k == "FragmentPath":
                out["fragment_path"] = v
        out["available"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _probe_connections(pid: int) -> list[dict[str, Any]]:
    """Parse /proc/net/tcp for established sockets owned by *pid*."""
    inodes: set[str] = set()
    try:
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:["):
                inodes.add(target[8:-1])
    except (OSError, PermissionError):
        return []

    def _parse(path: str, ipv6: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except OSError:
            return rows
        import socket
        import struct

        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            inode = parts[9]
            if inode not in inodes:
                continue
            state = parts[3]
            # 01 = ESTABLISHED
            lip_h, lport_h = parts[1].split(":")
            rip_h, rport_h = parts[2].split(":")
            try:
                if ipv6:
                    def _v6(h: str) -> str:
                        b = bytes.fromhex(h)
                        # kernel stores little-endian 32-bit words
                        words = struct.unpack("!4I", bytes(reversed(b)) if False else b)
                        # simpler: use inet_ntop on reversed word order
                        raw = bytes.fromhex(h)
                        # Linux /proc shows IPv6 as 4 little-endian u32
                        parts32 = [raw[i:i + 4][::-1] for i in range(0, 16, 4)]
                        return socket.inet_ntop(socket.AF_INET6, b"".join(parts32))
                    local = f"[{_v6(lip_h)}]:{int(lport_h, 16)}"
                    remote = f"[{_v6(rip_h)}]:{int(rport_h, 16)}"
                else:
                    def _v4(h: str) -> str:
                        b = bytes.fromhex(h)
                        return socket.inet_ntoa(b[::-1])
                    local = f"{_v4(lip_h)}:{int(lport_h, 16)}"
                    remote = f"{_v4(rip_h)}:{int(rport_h, 16)}"
            except Exception:  # noqa: BLE001
                local, remote = parts[1], parts[2]
            role = "other"
            try:
                rport = int(rport_h, 16)
                rhost = remote.rsplit(":", 1)[0].strip("[]")
                if rport == 5432 or rhost in ("127.0.0.1", "::1"):
                    role = "postgres"
                elif rport == 13000:
                    role = "databento_live"
            except ValueError:
                pass
            rows.append({
                "local": local,
                "remote": remote,
                "state": "ESTAB" if state == "01" else state,
                "role": role,
                "inode": inode,
            })
        return rows

    return _parse("/proc/net/tcp") + _parse("/proc/net/tcp6", ipv6=True)


def _probe_raw_live() -> dict[str, Any]:
    """Summarize today's raw live archive directory."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_dir = _RAW_LIVE_DIR / today
    out: dict[str, Any] = {
        "dir": str(day_dir),
        "exists": day_dir.is_dir(),
        "files": [],
        "total_bytes": 0,
        "total_gb": 0.0,
    }
    if not day_dir.is_dir():
        return out
    files = []
    total = 0
    try:
        for p in sorted(day_dir.iterdir()):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            total += st.st_size
            files.append({
                "name": p.name,
                "path": str(p),
                "bytes": st.st_size,
                "mb": round(st.st_size / (1024 * 1024), 1),
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "compressed": p.name.endswith(".zst") or p.name.endswith(".gz"),
            })
    except OSError as exc:
        out["error"] = str(exc)
    out["files"] = files
    out["total_bytes"] = total
    out["total_gb"] = round(total / 1e9, 2)
    return out


def _probe_disk(path: str = "/opt/vtech/data") -> dict[str, Any]:
    import shutil
    try:
        u = shutil.disk_usage(path)
        return {
            "path": path,
            "total_gb": round(u.total / 1e9, 1),
            "used_gb": round(u.used / 1e9, 1),
            "free_gb": round(u.free / 1e9, 1),
            "used_pct": round(100.0 * u.used / u.total, 1) if u.total else None,
        }
    except OSError as exc:
        return {"path": path, "error": str(exc)}


def _table_detail(
    name: str,
    latest: datetime | None,
    rows_10s: int,
    rows_2m: int,
    *,
    kind: str = "tick",
) -> dict[str, Any]:
    age = _age_s(latest)
    return {
        "name": name,
        "latest": latest.isoformat() if latest else None,
        "age_s": age,
        "rows_10s": int(rows_10s or 0),
        "rows_2m": int(rows_2m or 0),
        "status": _table_status(age, int(rows_10s or 0), kind=kind),
        "kind": kind,
    }


def refresh_streamer_health() -> dict[str, Any]:
    """Refresh cached streamer health: process, systemd, connections, tables, raw."""
    import psycopg
    from src.common.db import get_dsn

    now = datetime.now(timezone.utc)
    out: dict[str, Any] = {
        "status": "ok",
        "tables": {"tbbo": None, "bbo_1s": None, "statistics": None},
        "table_detail": {},
        "tbbo_rows_10s": 0,
        "tbbo_age_s": None,
        "live": False,
        "process": None,
        "processes": [],
        "systemd": None,
        "connections": [],
        "subscriptions": list(_LIVE_SUBSCRIPTIONS),
        "raw": None,
        "disk": None,
        "error": None,
        "checked_at": now.isoformat(),
        "dataset": "GLBX.MDP3",
    }

    # ── process + systemd + raw (always, even if DB down) ────────────────────
    pids = _find_streamer_pids()
    processes = []
    for pid in pids:
        info = _read_proc_process(pid)
        if info:
            processes.append(info)
    out["processes"] = processes
    out["process"] = processes[0] if processes else None

    systemd = _probe_systemd_unit()
    out["systemd"] = systemd

    main_pid = systemd.get("main_pid") if systemd else None
    if main_pid and main_pid > 0:
        out["connections"] = _probe_connections(int(main_pid))
    elif processes:
        out["connections"] = _probe_connections(int(processes[0]["pid"]))

    out["raw"] = _probe_raw_live()
    out["disk"] = _probe_disk()

    # ── DB table freshness (bounded probes only — see _TBBO_FRESH_SQL) ───────
    try:
        with psycopg.connect(
            get_dsn(),
            connect_timeout=4,
            options=f"-c statement_timeout={_HEALTH_STMT_TIMEOUT}",
        ) as c:
            tbbo_row = c.execute(_TBBO_FRESH_SQL).fetchone()
            stats_row = c.execute(_STATS_FRESH_SQL).fetchone()

        latest_tbbo = tbbo_row[0] if tbbo_row else None
        tbbo_10s = int(tbbo_row[1] or 0) if tbbo_row else 0
        tbbo_2m = int(tbbo_row[2] or 0) if tbbo_row else 0
        latest_stats = stats_row[0] if stats_row else None
        stats_10s = int(stats_row[1] or 0) if stats_row else 0
        stats_2m = int(stats_row[2] or 0) if stats_row else 0
        # bbo_1s is not probed here: daily chunks + no ts_recv index ⇒ seq scan.

        out["tables"]["tbbo"] = latest_tbbo.isoformat() if latest_tbbo else None
        out["tables"]["statistics"] = latest_stats.isoformat() if latest_stats else None
        out["tables"]["bbo_1s"] = None
        out["tbbo_rows_10s"] = int(tbbo_10s or 0)

        detail = {
            "tbbo": _table_detail(
                "databento.tbbo", latest_tbbo, int(tbbo_10s or 0), int(tbbo_2m or 0),
                kind="tick",
            ),
            "bbo_1s": _table_detail(
                "databento.bbo_1s", None, 0, 0, kind="tick",
            ),
            # CME statistics are sparse — use loose thresholds (10m healthy / 60m stale).
            "statistics": _table_detail(
                "databento.statistics", latest_stats, int(stats_10s or 0), int(stats_2m or 0),
                kind="stats",
            ),
        }
        out["table_detail"] = detail

        if latest_tbbo is not None:
            age = _age_s(latest_tbbo)
            out["tbbo_age_s"] = age
            out["live"] = bool(age is not None and age <= _LIVE_AGE_S)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["live"] = False

    # ── overall health rollup (``status`` stays ok/error for API compat) ─────
    # UI should prefer ``health``: healthy | degraded | error | warming
    # Overall health tracks the trade path (process, sockets, tbbo). bbo_1s is
    # not SQL-probed (no ts_recv index). Sparse statistics only roll up when
    # *stale*; soft lag stays a table-level note.
    notes: list[str] = []
    if out["status"] == "error":
        health = "error"
    else:
        health = "healthy"
        sd = systemd or {}
        if sd.get("available"):
            if sd.get("active_state") != "active":
                health = "error"
                notes.append(f"systemd active_state={sd.get('active_state')}")
            elif not processes:
                health = "error"
                notes.append("systemd active but streamer process not found")
        elif not processes:
            # No systemd probe (e.g. restricted env): process absence is soft.
            notes.append("streamer process not observed")
            if health == "healthy":
                health = "degraded"

        if not out.get("live"):
            if health == "healthy":
                health = "degraded"
            notes.append(
                f"feed lag tbbo_age_s={out.get('tbbo_age_s')} "
                f"(live threshold {_LIVE_AGE_S}s)"
            )

        disk_free = (out.get("disk") or {}).get("free_gb")
        if disk_free is not None and disk_free < 5.0:
            if health == "healthy":
                health = "degraded"
            notes.append(f"low disk free_gb={disk_free}")

        if processes:
            has_dbento = any(
                c.get("role") == "databento_live" for c in (out.get("connections") or [])
            )
            has_pg = any(c.get("role") == "postgres" for c in (out.get("connections") or []))
            if not has_dbento or not has_pg:
                if health == "healthy":
                    health = "degraded"
                notes.append(f"sockets databento={has_dbento} postgres={has_pg}")

        for key, det in (out.get("table_detail") or {}).items():
            st = det.get("status")
            if st not in ("stale", "degraded"):
                continue
            notes.append(f"{key}={st} age_s={det.get('age_s')}")
            # Tick feeds always affect overall health; stats only when stale.
            if det.get("kind") == "stats" and st != "stale":
                continue
            if health == "healthy":
                health = "degraded"

    out["health"] = health
    out["notes"] = notes
    # Keep ``error`` for hard failures only — soft notes used to be stuffed here
    # and made the Live tab look broken when only statistics lag was soft.
    if health == "error" and not out.get("error") and notes:
        out["error"] = "; ".join(notes[:4])
    out["ok"] = health in ("healthy", "degraded", "warming") or out["status"] == "ok"

    with _LOCK:
        _STREAMER_HEALTH.clear()
        _STREAMER_HEALTH.update(copy.deepcopy(out))
        return copy.deepcopy(_STREAMER_HEALTH)


# ── full-day signal ledger (scored in the heavy snapshot) ───────────────────────

def _build_ledger(
    df, bundle: dict[str, Any], *,
    tgt_pts: float, stop_pts: float, horizon_s: int, tau: float,
) -> list[dict[str, Any]]:
    """Score every 5s bar via src.serving, then first-touch barrier payoff.

    A directional call at bar t is resolved by FIRST-TOUCH on the 5s close path:

      * ``target``  — favourable excursion reaches +``move_threshold`` before stop
      * ``stop``    — adverse excursion reaches −``mae_tolerance`` first
      * ``timeout`` — neither barrier within ``horizon_s`` (vertical barrier)

    Bars whose window extends past the newest data with no touch yet are
    ``pending``. Rows are returned newest-first.

    Each row includes ``exit_ts`` / ``exit_px`` when a barrier (or timeout) has
    resolved on the 5s path. This is **not** trade-print execution — see
    :func:`_build_ledger_live_ticks` for Databento TBBO limit+OCO fills.
    """
    import numpy as np
    from src.serving import predict_canonical

    if "_close" not in df.columns or df.empty:
        return []

    n_missing, feature_complete = _model_feature_coverage(df, bundle["feature_cols"])
    complete_df = df.loc[feature_complete]
    pred = predict_canonical(complete_df, bundle) if not complete_df.empty else None
    pred_by_ts = pred.set_index("ts") if pred is not None else None

    close = df["_close"].astype("float64").ffill().to_numpy()
    ts_ns = df.index.asi8
    horizon_ns = int(horizon_s) * 1_000_000_000
    n = len(df)
    win_end = np.searchsorted(ts_ns, ts_ns + horizon_ns, side="right")

    def _first_touch(i: int, direction: str):
        """Return (outcome, pnl_pts, pending, exit_i | None)."""
        e = close[i]
        if not np.isfinite(e):
            return None, None, False, None
        j_end = int(win_end[i])
        seg = close[i + 1:j_end]
        excur = (seg - e) if direction == "up" else (e - seg)
        hit_t = np.flatnonzero(excur >= tgt_pts)
        hit_s = np.flatnonzero(excur <= -stop_pts)
        t_idx = hit_t[0] if hit_t.size else np.inf
        s_idx = hit_s[0] if hit_s.size else np.inf
        window_complete = ts_ns[i] + horizon_ns <= ts_ns[-1]
        if t_idx < s_idx:
            j = i + 1 + int(hit_t[0])
            return "target", round(float(excur[hit_t[0]]), 2), False, j
        if s_idx < t_idx:
            j = i + 1 + int(hit_s[0])
            return "stop", round(float(excur[hit_s[0]]), 2), False, j
        if not window_complete:
            return None, None, True, None
        drift = excur[-1] if excur.size else 0.0
        # timeout: mark-to-market at last bar inside the horizon window
        j = (j_end - 1) if j_end > i + 1 else None
        return "timeout", round(float(drift), 2), False, j

    rows: list[dict[str, Any]] = []
    for i in range(n):
        complete = bool(feature_complete.iloc[i])
        if complete and pred_by_ts is not None:
            scored = pred_by_ts.loc[df.index[i]]
            direction = str(scored["direction"])
            conf = float(scored["confidence"])
        else:
            # XGBoost was trained with a fully materialized feature matrix. Do
            # not route missing live values through its independent NaN branch.
            direction, conf = "flat", 0.0
        signal = complete and direction != "flat" and conf >= tau

        e = close[i]
        if direction == "flat":
            outcome, pnl, is_pending, exit_i = None, None, False, None
        else:
            outcome, pnl, is_pending, exit_i = _first_touch(i, direction)
        win = {"target": True, "stop": False}.get(outcome)

        exit_ts = exit_px = None
        if exit_i is not None and 0 <= int(exit_i) < n:
            exit_ts = df.index[int(exit_i)].isoformat()
            cx = close[int(exit_i)]
            if np.isfinite(cx):
                exit_px = round(float(cx), 2)

        hold_s = None
        if exit_ts is not None:
            try:
                hold_s = round(
                    (df.index[int(exit_i)] - df.index[i]).total_seconds(), 1)
            except Exception:
                hold_s = None

        rows.append({
            "ts": df.index[i].isoformat(),
            "dir": direction,
            "conf": round(conf, 3),
            "sig": signal,
            "entry": None if not np.isfinite(e) else round(float(e), 2),
            "entry_px": None if not np.isfinite(e) else round(float(e), 2),
            "exit_ts": exit_ts,
            "exit_px": exit_px,
            "hold_s": hold_s,
            "outcome": outcome,
            "pnl": pnl,
            "win": win,
            "pending": is_pending,
            "fill_path": "bar_5s",
            "n_present": len(bundle["feature_cols"]) - int(n_missing.iloc[i]),
            "n_total": len(bundle["feature_cols"]),
            "feature_complete": complete,
        })
    rows.reverse()
    return rows


def _build_ledger_live_ticks(
    *,
    day: str,
    model_path: str | Path,
    objective_id: str | None,
    tau: float,
    entry_style: str = "last",
) -> list[dict[str, Any]]:
    """Trade blotter from Databento TBBO prints + limit entry + OCO (paper engine).

    Uses the same fill rules as ``python -m src.paper.runner --replay-db``.
    Source prints are live-ingested rows in ``databento.tbbo`` (or historical
    live archive for past days) — **not** 5s feature closes.
    """
    from src.paper.engine import PaperEngine
    from src.paper.policy import load_paper_policy
    from src.paper.signals import SignalCursor, load_feature_day, score_feature_frame
    from src.paper.stream import iter_tbbo_from_db

    mpath = Path(model_path)
    policy = load_paper_policy(
        objective_id=objective_id,
        model_path=mpath,
        dir_threshold=float(tau),
        entry_style=entry_style,  # type: ignore[arg-type]
    )
    try:
        df = load_feature_day(str(day))
    except FileNotFoundError:
        logger.warning("live_tick_ledger_no_features day=%s", day)
        return []
    preds = score_feature_frame(df, mpath)
    cursor = SignalCursor(preds)
    engine = PaperEngine(policy=policy)

    n_ticks = 0
    for tick in iter_tbbo_from_db(day):
        n_ticks += 1
        for sig_ts, direction, conf in cursor.signals_through(tick.ts):
            engine.on_signal(direction, conf, ts=sig_ts)
        engine.on_tick(tick)

    if engine.position is not None and engine.last_tick is not None:
        engine._force_close(engine.last_tick, "eod")

    rows: list[dict[str, Any]] = []
    for tr in engine.closed:
        d = "up" if tr.side == "long" else "down"
        oc = tr.exit_reason
        # normalize paper reasons to blotter vocabulary
        if oc == "time":
            oc = "timeout"
        rows.append({
            "ts": tr.entry_ts,
            "exit_ts": tr.exit_ts,
            "dir": d,
            "conf": round(float(tr.confidence), 3),
            "sig": True,
            "entry": round(float(tr.entry_px), 2),
            "entry_px": round(float(tr.entry_px), 2),
            "exit_px": round(float(tr.exit_px), 2),
            "hold_s": round(float(tr.holding_seconds), 1),
            "outcome": oc,
            "pnl": round(float(tr.pnl_points), 2),
            "win": oc == "target",
            "pending": False,
            "fill_path": "tbbo_ticks",
            "side": tr.side,
            "trade_id": tr.trade_id,
        })
    # open / working as pending rows (newest first after reverse)
    if engine.position is not None:
        pos = engine.position
        d = "up" if pos.side == "long" else "down"
        rows.append({
            "ts": pos.entry_ts.isoformat() if hasattr(pos.entry_ts, "isoformat")
            else str(pos.entry_ts),
            "exit_ts": None,
            "dir": d,
            "conf": round(float(pos.confidence), 3),
            "sig": True,
            "entry": round(float(pos.entry_px), 2),
            "entry_px": round(float(pos.entry_px), 2),
            "exit_px": None,
            "hold_s": None,
            "outcome": None,
            "pnl": None,
            "win": None,
            "pending": True,
            "fill_path": "tbbo_ticks",
            "side": pos.side,
            "trade_id": pos.trade_id,
        })
    for od in engine.working:
        if od.kind != "entry_limit" or od.status != "working":
            continue
        d = "up" if od.side == "long" else "down"
        rows.append({
            "ts": engine.last_tick.ts.isoformat() if engine.last_tick else None,
            "exit_ts": None,
            "dir": d,
            "conf": None,
            "sig": True,
            "entry": round(float(od.limit_px), 2),
            "entry_px": round(float(od.limit_px), 2),
            "exit_px": None,
            "hold_s": None,
            "outcome": "working_limit",
            "pnl": None,
            "win": None,
            "pending": True,
            "fill_path": "tbbo_ticks",
            "side": od.side,
            "order_id": od.order_id,
        })

    rows.reverse()  # newest first
    logger.info(
        "live_tick_ledger day=%s ticks=%s trades=%s",
        day, n_ticks, sum(1 for r in rows if not r.get("pending")),
    )
    return rows


def _model_feature_coverage(df, feat_cols: list[str]):
    """Return missing-input counts and a full-feature eligibility mask."""
    import numpy as np
    import pandas as pd

    if df is None or df.empty or not feat_cols:
        raise ValueError("empty frame or feature list")

    values = df.reindex(columns=list(feat_cols)).to_numpy(
        dtype="float64", na_value=np.nan)
    n_missing = pd.Series(
        (~np.isfinite(values)).sum(axis=1), index=df.index, dtype="int64")
    return n_missing, n_missing.eq(0)


def _select_score_bar(
    df,
    feat_cols: list[str],
    *,
    max_nan_frac: float = 0.5,
) -> tuple[Any, int, str | None]:
    """Pick the newest bar with enough model features present.

    Returns ``(bar_index, n_nan, note)``.

    Live feature rebuilds often have a short trailing edge where some modules
    (e.g. order-flow) already wrote values but the model feature set is still
    entirely NaN. Prefer the latest bar with ``n_nan / n_feat <= max_nan_frac``
    (default: at least half of the model's columns non-null). Fall back to the
    fewest-NaN bar in the day if nothing qualifies.
    """
    import numpy as np
    import pandas as pd

    if df is None or df.empty or not feat_cols:
        raise ValueError("empty frame or feature list")

    X = df.reindex(columns=list(feat_cols))
    n_nan = X.isna().sum(axis=1).astype("int64")
    complete = n_nan.eq(0)
    if bool(complete.any()):
        bar_idx = df.index[complete][-1]
        note = None
        edge = df.index[-1]
        if bar_idx != edge:
            try:
                lag_s = (pd.Timestamp(edge) - pd.Timestamp(bar_idx)).total_seconds()
                if lag_s > 0:
                    note = (
                        f"scored bar lags live edge by {lag_s:.0f}s "
                        f"(trailing bars incomplete for model features)"
                    )
            except Exception:
                note = "scored an earlier bar — live edge incomplete for model features"
        return bar_idx, 0, note
    latest_missing = int(n_nan.iloc[-1])
    return (
        None,
        latest_missing,
        f"waiting for all {len(feat_cols)} required model features "
        f"({len(feat_cols) - latest_missing}/{len(feat_cols)} currently usable)",
    )


def _feature_bar_age_seconds(bar_ts: Any, *, now: datetime | None = None) -> float:
    """Age (seconds) of a feature-bar timestamp relative to UTC *now*."""
    if isinstance(bar_ts, datetime):
        dt = bar_ts
    elif hasattr(bar_ts, "to_pydatetime"):
        dt = bar_ts.to_pydatetime()
    else:
        dt = datetime.fromisoformat(str(bar_ts).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    return (now_utc - dt).total_seconds()


# ── file-bridge signal exporter (broker-side tail target) ─────────────────────
#
# On every heavy snapshot we append newly-fired signals (``sig=True`` rows whose
# ts is strictly after the watermark) to ``data/live/signals.jsonl`` and touch
# ``data/live/heartbeat.json``. Broker-side scripts (NinjaScript / cAlgo / etc.)
# tail the JSONL and consult the heartbeat to confirm the feed is alive. Only
# the signal itself is emitted — the OCO/stop legs are left to the broker side
# (uses the policy context we ship alongside).

def _live_dir() -> Path:
    p = live_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _init_signal_watermark(path: Path) -> str | None:
    """Read the last JSON line to seed the watermark on cold start."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and b"\n" not in data[:-1]:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            line = data.strip().split(b"\n")[-1]
        return json.loads(line).get("ts")
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_signal_watermark_read_failed: %s", exc)
        return None


def _is_within_session_window(
    ts: str,
    session: dict[str, Any],
    *,
    trading_start: str,
    trading_end: str,
) -> bool:
    """Whether a timestamp falls within the configured objective session."""
    from datetime import date, time as dtime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore

    session_date = date.fromisoformat(str(session["date"]))
    tz = ZoneInfo(str(session["timezone"]))
    start_h, start_m = _parse_hhmm(trading_start)
    end_h, end_m = _parse_hhmm(trading_end)
    start = dtime(start_h, start_m)
    end = dtime(end_h, end_m)
    signal_at = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(tz)
    signal_time = signal_at.timetz().replace(tzinfo=None)
    return signal_at.date() == session_date and start <= signal_time <= end


def _is_within_signal_export_window(
    ts: str,
    session: dict[str, Any],
    *,
    trading_start: str,
    trading_end: str,
    now: datetime | None = None,
) -> bool:
    """Whether a signal belongs to the active configured export session."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore

    tz = ZoneInfo(str(session["timezone"]))
    now_local = (now or datetime.now(timezone.utc)).astimezone(tz)
    start_h, start_m = _parse_hhmm(trading_start)
    end_h, end_m = _parse_hhmm(trading_end)
    session_date = date.fromisoformat(str(session["date"]))
    now_time = now_local.timetz().replace(tzinfo=None)

    return (
        now_local.date() == session_date
        and (start_h, start_m) <= (now_time.hour, now_time.minute) <= (end_h, end_m)
        and _is_within_session_window(
            ts, session, trading_start=trading_start, trading_end=trading_end)
    )


def read_emitted_signal_rows(
    objective_id: str,
    session: dict[str, Any],
    effective: dict[str, Any],
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Read immutable broker-facing signals for one objective session."""
    session_date = str(session["date"])
    path = _live_dir() / "signals.jsonl"
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    not isinstance(row, dict)
                    or row.get("objective_id") != objective_id
                    or str(row.get("session_date")) != session_date
                    or (model is not None and row.get("model") != model)
                    or not row.get("ts")
                    or not _is_within_session_window(
                        str(row["ts"]),
                        session,
                        trading_start=effective["trading_start"],
                        trading_end=effective["trading_end"],
                    )
                ):
                    continue
                rows.append({**row, "sig": True, "pending": False,
                             "fill_path": "emitted_signal"})
    except OSError as exc:
        logger.warning("live_signal_ledger_read_failed path=%s err=%s", path, exc)
        return []
    return sorted(rows, key=lambda row: str(row["ts"]), reverse=True)


def _export_signals(ledger: list[dict[str, Any]], snap: dict[str, Any], *,
                    now: datetime | None = None) -> None:
    """Append fresh in-session ``sig=True`` rows and refresh the heartbeat file."""
    live = _live_dir()
    sig_path = live / "signals.jsonl"
    hb_path = live / "heartbeat.json"

    with _SIG_LOCK:
        if _SIG_STATE["path"] != str(sig_path):
            _SIG_STATE["last_ts"] = _init_signal_watermark(sig_path)
            _SIG_STATE["path"] = str(sig_path)
        watermark = _SIG_STATE["last_ts"]

        barriers = snap.get("barriers") or {}
        session = snap.get("session") or {}
        objective_id = snap.get("objective_id")
        if not objective_id:
            logger.warning("live_signal_export_skipped: missing objective_id")
            return
        # Imported lazily because regmodel imports this module for its status API.
        # The effective window makes the Registered Model controls authoritative
        # for the broker-facing signal feed without creating an import cycle.
        from src.webui.regmodel import effective_settings
        effective = effective_settings(str(objective_id))
        export_now = now or datetime.now(timezone.utc)
        ctx = {
            "objective_id": objective_id,
            "model": snap.get("model"),
            "candidate_id": snap.get("candidate_id"),
            "model_path": snap.get("model_path"),
            "tau": snap.get("tau"),
            "tgt_pts": barriers.get("move_threshold"),
            "stop_pts": barriers.get("mae_tolerance"),
            "horizon_s": barriers.get("horizon_s"),
            "instrument": snap.get("instrument") or "NQ",
            "point_value": snap.get("point_value"),
            "session_date": session.get("date"),
        }

        # ledger rows are newest-first; walk oldest-first so the file is chronological.
        new_rows: list[dict[str, Any]] = []
        for r in reversed(ledger):
            if not r.get("sig") or r.get("feature_complete") is False:
                continue
            if r.get("pending"):
                pass  # emit anyway — the signal fired at bar-close regardless of outcome
            ts = r.get("ts")
            if not ts or (watermark and ts <= watermark):
                continue
            if not _is_within_signal_export_window(
                str(ts),
                session,
                trading_start=effective["trading_start"],
                trading_end=effective["trading_end"],
                now=export_now,
            ):
                continue
            new_rows.append({
                "ts": ts,
                "dir": r.get("dir"),
                "conf": r.get("conf"),
                "entry_px": r.get("entry_px") or r.get("entry"),
                "n_present": r.get("n_present"),
                "n_total": r.get("n_total"),
                **ctx,
            })

        if new_rows:
            from src.execution.protocol import make_signal_id
            for row in new_rows:
                row["signal_id"] = make_signal_id(row)

        if new_rows:
            with sig_path.open("a") as f:
                for row in new_rows:
                    f.write(json.dumps(row, default=str) + "\n")
            _SIG_STATE["last_ts"] = new_rows[-1]["ts"]
            logger.info("live_signal_export n=%s last_ts=%s", len(new_rows), _SIG_STATE["last_ts"])

        hb = {
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "built_at": snap.get("built_at"),
            "status": snap.get("status"),
            "last_signal_ts": _SIG_STATE["last_ts"],
            "in_session": session.get("in_session"),
            "session_closed": session.get("session_closed"),
            "n_total": snap.get("n_total"),
            "signal_export_window": {
                "trading_start": effective["trading_start"],
                "trading_end": effective["trading_end"],
                "timezone": session.get("timezone"),
                "session_date": session.get("date"),
            },
            **ctx,
        }
        tmp = hb_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(hb, indent=2, default=str))
        tmp.replace(hb_path)


# ── heavy snapshot (background cadence only) ────────────────────────────────────

def build_snapshot() -> dict[str, Any]:
    """Build today's features from live data and score the latest bar."""
    import pandas as pd

    t0 = time.time()
    try:
        b = _bundle()
    except Exception as exc:  # noqa: BLE001
        snap = {
            "status": "error", "prediction": None,
            "note": f"{type(exc).__name__}: {exc}",
            "model": None, "objective_id": active_objective_id(),
            "build_s": round(time.time() - t0, 1),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        return _commit_snapshot(snap)

    from src.serving import predict_canonical

    feat_cols = b["feature_cols"]
    trade = b["trade"]
    resolved = b["resolved"]
    n_total = len(feat_cols)
    tau = float(trade["tau"])
    tgt_pts = float(trade["move_threshold"])
    stop_pts = float(trade["mae_tolerance"])
    horizon_s = int(trade["horizon_s"])

    snap: dict[str, Any] = {
        "model": b["version"],
        "candidate_id": b["candidate_id"],
        "family": b["family"],
        "model_path": b["path"],
        "model_source": resolved.get("source"),
        "model_objective_id": resolved.get("objective_id") or b["manifest"].get("objective_id"),
        "objective_id": trade["objective_id"],   # barriers / KPI objective in force
        "active_objective": b.get("active_objective"),  # config.yaml pin
        "selection_mode": b.get("selection_mode") or resolved.get("selection_mode") or "auto",
        "model_label": resolved.get("label"),
        "model_note": resolved.get("note"),
        "n_total": n_total,
        "tau": tau,
        "instrument": trade.get("instrument") or "NQ",
        "point_value": float(trade.get("point_value") or 20.0),
        "barriers": {
            "move_threshold": tgt_pts,
            "mae_tolerance": stop_pts,
            "horizon_s": horizon_s,
            "horizon_min": round(horizon_s / 60, 1),
            "scheme": trade.get("scheme"),
        },
    }
    try:
        # Session-aware feature day (America/Chicago / objective TZ — NOT UTC date).
        # After the objective session closes, keep serving that day's blotter from
        # disk so Apply-model still shows results outside trading hours.
        trade_oid = trade.get("objective_id") or b.get("active_objective")
        feat = resolve_feature_day(trade_oid)
        ctx = feat["context"]
        snap["session"] = {
            "date": str(feat["session_date"]),
            "timezone": ctx.get("timezone"),
            "trading_start": ctx.get("trading_start"),
            "trading_end": ctx.get("trading_end"),
            "in_session": ctx.get("in_session"),
            "session_closed": ctx.get("session_closed"),
            "feature_source": feat["source"],
            "feature_path": feat["path"],
            "now_local": ctx.get("now_local"),
        }
        df = feat["df"]
        if df is None or df.empty:
            snap.update(
                status="warming", prediction=None, ledger=[],
                note=(feat.get("note")
                      or "no feature parquet for the session day yet "
                         "(insufficient live data / outside market hours)"),
            )
        else:
            # Score the freshest bar with usable *model* features — not merely
            # of_delta present. The live rebuild often leaves a short trailing
            # edge where order-flow columns exist but model cols (rv_/bp_/…) are
            # still all-NaN; XGBoost will still emit a confident class from a
            # fully-missing row, which looked like "0/43 features" + a signal.
            bar_idx, n_nan, edge_note = _select_score_bar(df, feat_cols)
            if bar_idx is None:
                snap.update(
                    status="warming",
                    note=" · ".join(n for n in (feat.get("note"), edge_note) if n),
                    prediction={
                        "bar_ts": df.index[-1].isoformat(),
                        "n_rows": int(len(df)),
                        "n_nan": n_nan,
                        "n_present": n_total - n_nan,
                        "n_total": n_total,
                        "signal": False,
                    },
                    ledger=[],
                )
                snap["horizon_min"] = round(horizon_s / 60, 1)
                snap["build_s"] = round(time.time() - t0, 1)
                snap["built_at"] = datetime.now(timezone.utc).isoformat()
                # Same inference path as Backtest (src.serving.predict_canonical).
                return _commit_snapshot(snap)
            bar_age_s = _feature_bar_age_seconds(bar_idx)
            if ctx.get("in_session") and bar_age_s > _LIVE_AGE_S:
                snap.update(
                    status="warming",
                    note=(
                        f"stale feature bar during active session: age {bar_age_s:.1f}s "
                        f"> {_LIVE_AGE_S:.1f}s freshness threshold"
                    ),
                    prediction={
                        "bar_ts": bar_idx.isoformat(),
                        "n_rows": int(len(df)),
                        "n_nan": n_nan,
                        "n_present": n_total - n_nan,
                        "n_total": n_total,
                        "signal": False,
                    },
                    ledger=[],
                )
                snap["horizon_min"] = round(horizon_s / 60, 1)
                snap["build_s"] = round(time.time() - t0, 1)
                snap["built_at"] = datetime.now(timezone.utc).isoformat()
                return _commit_snapshot(snap)
            one = predict_canonical(df.loc[[bar_idx]], b).iloc[0]
            direction = str(one["direction"])
            conf = float(one["confidence"])
            p_down, p_flat, p_up = (
                float(one["p_down"]), float(one["p_flat"]), float(one["p_up"]))
            notes = [n for n in (feat.get("note"), edge_note) if n]
            # status=ok whenever we have a scorable day, even after the session closed
            snap.update(
                status="ok",
                note=" · ".join(notes) if notes else None,
                prediction={
                    "bar_ts": bar_idx.isoformat(),
                    "n_rows": int(len(df)),
                    "n_nan": n_nan,
                    "n_present": n_total - n_nan,
                    "n_total": n_total,
                    "p_down": p_down, "p_flat": p_flat, "p_up": p_up,
                    "direction": direction, "confidence": conf,
                    "signal": direction != "flat" and conf >= tau,
                })
            try:
                snap["ledger"] = _build_ledger(
                    df, b,
                    tgt_pts=tgt_pts, stop_pts=stop_pts,
                    horizon_s=horizon_s, tau=tau,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("live_ledger_failed: %s", exc)
                snap["ledger"] = []
            try:
                _export_signals(snap.get("ledger") or [], snap)
            except Exception as exc:  # noqa: BLE001
                logger.warning("live_signal_export_failed: %s", exc)
            snap["horizon_min"] = round(horizon_s / 60, 1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_snapshot_failed: %s", exc)
        snap.update(status="error", prediction=None, note=f"{type(exc).__name__}: {exc}")

    snap["build_s"] = round(time.time() - t0, 1)
    snap["built_at"] = datetime.now(timezone.utc).isoformat()
    # Tick blotter cache is keyed by day|model|tau outside _STATE so a
    # snapshot refresh does not restart a multi-minute paper replay.
    return _commit_snapshot(snap)


def snapshot() -> dict[str, Any]:
    """Return the cached prediction snapshot + cached streamer health."""
    _refresh_snapshot_state_from_cache()
    with _LOCK:
        state = dict(_STATE)
    state.pop("ledger", None)
    state["streamer"] = streamer_health()
    state["server_time"] = datetime.now(timezone.utc).isoformat()
    # Always surface the current pin even before the first successful build.
    if "objective_id" not in state or state.get("objective_id") is None:
        state["objective_id"] = active_objective_id()
    return state


def _run_ticks_build(
    gen: int,
    key: str,
    *,
    day: str,
    model_path: str,
    objective_id: str | None,
    tau: float,
) -> None:
    """Full-day paper replay — never call from an HTTP handler thread."""
    t0 = time.time()
    try:
        rows = _build_ledger_live_ticks(
            day=day,
            model_path=model_path,
            objective_id=objective_id,
            tau=tau,
        )
        note = f"tbbo limit+OCO · {len(rows)} rows · {time.time() - t0:.1f}s"
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_tick_ledger_failed: %s", exc)
        rows = []
        note = f"tick blotter failed: {type(exc).__name__}: {exc}"
    built_at = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        if _TICKS_JOB.get("gen") != gen or _TICKS_JOB.get("key") != key:
            return
        _TICKS_CACHE.clear()
        _TICKS_CACHE.update({
            "key": key,
            "rows": rows,
            "built_at": built_at,
            "note": note,
            "building": False,
        })
        if _TICKS_JOB.get("key") == key:
            _TICKS_JOB["key"] = None
    logger.info("live_tick_ledger_ready key=%s n=%s note=%s", key, len(rows), note)


def _schedule_ticks_ledger(
    key: str,
    *,
    day: str,
    model_path: str,
    objective_id: str | None,
    tau: float,
) -> str:
    """Start at most one ticks replay for *key*. HTTP path stays cache-only."""
    with _LOCK:
        if _TICKS_JOB.get("key") == key:
            _TICKS_CACHE["building"] = True
            _TICKS_CACHE.setdefault("key", key)
            return "building"
        _TICKS_JOB["gen"] = int(_TICKS_JOB.get("gen") or 0) + 1
        _TICKS_JOB["key"] = key
        gen = int(_TICKS_JOB["gen"])
        if _TICKS_CACHE.get("key") != key:
            _TICKS_CACHE.clear()
            _TICKS_CACHE["key"] = key
            _TICKS_CACHE["rows"] = []
        _TICKS_CACHE["building"] = True
    threading.Thread(
        target=_run_ticks_build,
        kwargs={
            "gen": gen,
            "key": key,
            "day": day,
            "model_path": model_path,
            "objective_id": objective_id,
            "tau": tau,
        },
        daemon=True,
        name="vtech-live-ticks-ledger",
    ).start()
    return "scheduled"


def ledger(
    mode: str = "bar",
    *,
    tau: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Return the day blotter (newest-first).

    * ``mode=bar`` (default) — every 5s bar, first-touch on feature ``_close``
    * ``mode=ticks`` — only **trades** from Databento ``tbbo`` prints via the
      paper limit+OCO engine (live market data path)
    """
    mode = (mode or "bar").lower().strip()
    if mode in ("live", "tick", "tbbo", "prints"):
        mode = "ticks"
    if mode not in ("bar", "ticks"):
        mode = "bar"

    with _LOCK:
        barriers = _STATE.get("barriers") or {}
        state_tau = float(_STATE.get("tau") or 0.8)
        objective_id = _STATE.get("objective_id")
        session = dict(_STATE.get("session") or {})
        model = _STATE.get("model")
        thr = float(tau) if tau is not None else state_tau
        meta_base = {
            "model": _STATE.get("model"),
            "candidate_id": _STATE.get("candidate_id"),
            "objective_id": _STATE.get("objective_id"),
            "model_objective_id": _STATE.get("model_objective_id"),
            "model_path": _STATE.get("model_path"),
            "status": _STATE.get("status"),
            "note": _STATE.get("note"),
            "built_at": _STATE.get("built_at"),
            "tau": thr,
            "horizon_min": _STATE.get("horizon_min", barriers.get("horizon_min")),
            "barriers": barriers,
            "session": _STATE.get("session"),
            "fill_path": "bar_5s" if mode == "bar" else "tbbo_ticks",
            "mode": mode,
        }
        if mode == "bar":
            # The day parquet is rebuilt during the session and can change old
            # feature rows. The broker-facing JSONL is the immutable record of
            # signals actually emitted at each original 5-second close.
            rows = []
            if objective_id and session.get("date"):
                from src.webui.regmodel import effective_settings
                effective = effective_settings(str(objective_id))
                rows = read_emitted_signal_rows(
                    str(objective_id), session, effective, model=str(model) if model else None)
            meta_base["n"] = len(rows)
            meta_base["source"] = "signals.jsonl"
            meta_base["fill_path"] = "emitted_signal"
            meta_base["note"] = (
                "Immutable signals emitted at their original 5-second close; "
                "not a retroactive rescore of the mutable intraday feature parquet."
            )
            return {"rows": rows, "meta": meta_base}

        # ticks mode — cache is *not* in _STATE (snapshot reloads would wipe it).
        session = _STATE.get("session") or {}
        day = str(session.get("date") or "")
        model_path = _STATE.get("model_path")
        oid = _STATE.get("objective_id")
        snap_status = _STATE.get("status")
        cache_key = f"{day}|{model_path}|{thr:.4f}"
        cache = dict(_TICKS_CACHE) if _TICKS_CACHE.get("key") == cache_key else {}
        inflight = _TICKS_JOB.get("key") == cache_key

    if not day or not model_path or snap_status in (None, "starting", "error"):
        return {
            "rows": [],
            "meta": {
                **meta_base,
                "n": 0,
                "building": False,
                "note": "tick blotter needs a scored session day + model",
            },
        }

    ready = (
        not force
        and not inflight
        and cache.get("key") == cache_key
        and isinstance(cache.get("rows"), list)
        and not cache.get("building")
    )
    if not ready:
        _schedule_ticks_ledger(
            cache_key, day=day, model_path=str(model_path),
            objective_id=str(oid) if oid else None, tau=thr,
        )
        with _LOCK:
            cache = dict(_TICKS_CACHE) if _TICKS_CACHE.get("key") == cache_key else {}
            inflight = _TICKS_JOB.get("key") == cache_key

    rows = list(cache.get("rows") or [])
    building = bool(inflight or cache.get("building"))
    meta_base["n"] = len(rows)
    meta_base["building"] = building
    meta_base["built_at"] = cache.get("built_at") or meta_base["built_at"]
    meta_base["source"] = "tbbo_ticks_cache"
    if building:
        meta_base["note"] = (
            cache.get("note")
            or "Building TBBO DATA blotter in the background (full-day paper replay)…"
        )
    elif cache.get("note"):
        meta_base["note"] = cache.get("note")
    meta_base["n_ticks_note"] = cache.get("note")
    return {"rows": rows, "meta": meta_base}


def blotter_stats(rows: list[dict[str, Any]] | None = None,
                  tau: float | None = None) -> dict[str, Any]:
    """Compact day stats so the UI can prove a model rescore happened."""
    with _LOCK:
        if rows is None:
            rows = list(_STATE.get("ledger") or [])
        if tau is None:
            tau = float(_STATE.get("tau") or 0.8)
        candidate = _STATE.get("candidate_id")
        model_path = _STATE.get("model_path")
        session = _STATE.get("session") or {}
        barriers = _STATE.get("barriers") or {}
        built_at = _STATE.get("built_at")

    n_up = n_down = n_flat = 0
    n_sig = n_win = n_stop = n_to = 0
    net = 0.0
    for r in rows:
        d = r.get("dir")
        if d == "up":
            n_up += 1
        elif d == "down":
            n_down += 1
        else:
            n_flat += 1
        conf = float(r.get("conf") or 0)
        is_sig = d in ("up", "down") and conf >= tau and not r.get("pending")
        if is_sig:
            n_sig += 1
            oc = r.get("outcome")
            if oc == "target":
                n_win += 1
            elif oc == "stop":
                n_stop += 1
            elif oc == "timeout":
                n_to += 1
            if isinstance(r.get("pnl"), (int, float)):
                net += float(r["pnl"])
    return {
        "candidate_id": candidate,
        "model_path": model_path,
        "session_date": session.get("date"),
        "built_at": built_at,
        "tau": tau,
        "n_bars": len(rows),
        "n_up": n_up,
        "n_down": n_down,
        "n_flat": n_flat,
        "n_directional": n_up + n_down,
        "n_signals": n_sig,
        "n_win": n_win,
        "n_stop": n_stop,
        "n_timeout": n_to,
        "net_pnl": round(net, 2),
        "move_threshold": barriers.get("move_threshold"),
        "mae_tolerance": barriers.get("mae_tolerance"),
        "instrument": _STATE.get("instrument") or "NQ",
        "point_value": float(_STATE.get("point_value") or 20.0),
        "net_value": round(net * float(_STATE.get("point_value") or 20.0), 2),
    }
