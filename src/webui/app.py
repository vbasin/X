"""VTech web console — FastAPI backend.

Serves the single-page console (static/) plus a JSON API over the existing workflow
surfaces: trade objectives (ADR-018), the Scoping v2 orchestrator (ADR-025), the
model-selection search controller (Phase 23), and the Research Director schedule
(vtech-research-director.timer).

Trust model: binds 127.0.0.1 by default — same as the MLflow UI on :5000 — and is
reached through an SSH / VS Code port-forward. Isolation is the bind address, not a
Host allow-list: VS Code Ports and tunnels send Host values that are not
``localhost`` (``*.devtunnels.ms``, preview hostnames, the VM hostname). A 400
``Invalid host header`` makes the Ports-tab browser sit on an infinite spinner.
Mutating requests still require ``X-VTech-Console`` (blocks cross-site form/fetch
POSTs, which cannot set custom headers without a CORS preflight this app never grants).

Run::

    python -m src.webui.app [--host 127.0.0.1] [--port 8700]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import multiprocessing as mp
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.webui import datasources as ds
from src.webui import live
from src.webui import live_chart as livechart
from src.webui import mbo_orders_api
from src.webui import regmodel
from src.webui.jobs import JobManager

logger = logging.getLogger("vtech.webui")

_STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8700
_LIVE_LOOP_CADENCE_S = 5.0
_LIVE_LOOP_POLL_S = 0.5
_LIVE_HEALTH_CADENCE_S = 2.0


class _LiveWorkerRefresh:
    """Thread-safe refresh signal consumed by the lifespan worker loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False

    def request(self) -> None:
        with self._lock:
            self._requested = True

    def consume(self) -> bool:
        with self._lock:
            if not self._requested:
                return False
            self._requested = False
            return True


# ── request models ────────────────────────────────────────────────────────────

class ScopingLaunch(BaseModel):
    objective_id: str
    dry_run: bool = False
    budget: float | None = Field(default=None, ge=0, le=500)


class SearchLaunch(BaseModel):
    objective_id: str
    families: list[str] = Field(default_factory=list)
    dry_run: bool = False
    tune_trials: int = Field(default=0, ge=0, le=200)
    no_ensemble: bool = False
    manual_spec: str | None = None


class CustomTestBoard(BaseModel):
    """WebUI Custom Test board (models + dragged features + optional meta)."""
    objective_id: str
    models: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] | None = None
    walk_forward: dict[str, Any] | None = None
    dry_run: bool = False
    schema_version: str | None = "v1"
    tune_trials: int = Field(default=0, ge=0, le=200)
    feature_store: str | None = None
    head: str | None = None


class CustomTestSavedView(BaseModel):
    """Named column snapshot saved from a Custom Test model card (ADR-074)."""
    name: str
    features: list[str] = Field(default_factory=list)
    description: str | None = None
    overwrite: bool = False


class TimerAction(BaseModel):
    action: str                                  # "enable" | "disable"


class DirectorCycle(BaseModel):
    dry_run: bool = False


class ObjectivePayload(BaseModel):
    objective: dict[str, Any]


class RdEnableBody(BaseModel):
    enabled: bool = True


class RegModelSettings(BaseModel):
    """Registered Model page control panel (all fields optional — partial update)."""
    order_type: str | None = None            # "MKT" | "LIMIT"
    execution: str | None = None             # "OCO" | "ONCLOSE"
    tau: float | None = Field(default=None, ge=0, le=1)
    max_positions: int | None = Field(default=None, ge=1, le=50)
    trading_start: str | None = None         # HH:MM — today-only override
    trading_end: str | None = None


class ExternalStackLeg(BaseModel):
    """One engine in an External Models backtest stack."""
    model: str
    hardened: str | None = None
    params: dict[str, Any] | None = None
    timeframe: str | None = None  # 1min|5min|15min|1hour|1day; inherit book TF when omitted


class BacktestRun(BaseModel):
    """Path-backtest a staged serving bundle (same src.serving inference as Live)."""
    source: str = ""                         # staging/registry path, or external:<id>
    tau: float = Field(default=0.8, ge=0, le=1)
    confidence_gate: bool = True                 # False → admit all directional preds
    mv: float = Field(default=0.5, ge=0, le=1)   # Stage 1 gate (cascade OOF only)
    mv_gate: bool = True                          # False → disable Stage 1 gate
    p_reach_min: float = Field(default=0.75, ge=0, le=1)
    p_move_min: float = Field(default=0.5, ge=0, le=1)
    min_tgt_r: float = Field(default=0.25, ge=0, le=10)
    max_positions: int = Field(default=1, ge=1, le=50)
    start: str | None = None
    end: str | None = None
    objective_id: str | None = None
    sides: list[str] | None = None
    # True → OCO (TBBO triple barrier); False → on-close 5s path
    oco: bool = False
    tbbo_data: bool = False                  # legacy alias for oco
    tbbo_dev: bool = False
    live_data: bool = False
    mae_adjust: float = 0.0
    target_adjust: float = 0.0
    price_mode: str = "mid"                  # "mid" | "mkt"
    entry_latency_ms: float = 0.0
    exit_latency_ms: float = 0.0
    params: dict[str, Any] | None = None     # external-model param overlay (legacy mixed)
    stack: list[ExternalStackLeg] | None = None
    book: dict[str, Any] | None = None


class BacktestChart(BaseModel):
    """OHLC + SAR / EMA / ATR / ZigZag series for the Backtest Chart tab (same candles as the run)."""
    source: str = ""
    start: str | None = None
    end: str | None = None
    params: dict[str, Any] | None = None
    book: dict[str, Any] | None = None
    stack: list[ExternalStackLeg] | None = None
    session_only: bool = True


class ExternalDefaults(BaseModel):
    params: dict[str, Any]
    note: str | None = None


class ExternalHardened(BaseModel):
    name: str
    params: dict[str, Any] | None = None
    note: str | None = None
    set_as_defaults: bool = False


class LiveModelSelect(BaseModel):
    """Pin the Live-tab scorer to a staged/registry bundle, or return to auto."""
    mode: str = "auto"                       # "auto" | "manual"
    path: str | None = None                  # required when mode=manual


class ChartModelVarsReq(BaseModel):
    """Chart-Model Design — variable-catalog request (chart request echoed back)."""
    timeframe: str = "1min"
    lookback: str = "session"
    overlays: str | None = None
    profiles: str | None = None
    emas: str | None = None
    stochs: str | None = None
    sar: str | None = None
    vwap_bands: bool = False
    end: str | None = None
    cvd_origin: str = "session"
    vwap_origin: str = "session"
    ofi_origin: str | None = None
    ad_origin: str | None = None
    svp_origin: str | None = None
    vp_start: str | None = None
    vp_end: str | None = None
    eth_aggregate: bool = True


class ChartModelSimReq(BaseModel):
    """Chart-Model Design — simulate a rule tree over the same window the chart shows."""
    chart: ChartModelVarsReq
    model: dict[str, Any]
    print_min: int | None = None
    session_end: str | None = None
    point_value: float | None = None


class ChartModelSpec(BaseModel):
    """Persistable rule spec."""
    id: str
    name: str | None = None
    notes: str | None = None
    rules: list[dict[str, Any]]


class ChartNoteCreate(BaseModel):
    """Live Chart scenario note (brushed range + overlay snapshot)."""
    name: str
    text: str = ""
    symbol: str | None = None
    instrument: str | None = "/NQ"
    range: dict[str, Any]
    chart: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


# ── app factory ───────────────────────────────────────────────────────────────

def _run_live_snapshot_worker() -> None:
    live.build_snapshot()


def _spawn_live_snapshot_process(ctx: mp.context.BaseContext) -> mp.Process:
    proc = ctx.Process(target=_run_live_snapshot_worker, name="vtech-live-snapshot")
    proc.start()
    return proc


def create_app(jobs: JobManager | None = None, *, enable_live_worker: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = jobs or JobManager()
        app.state.live_snapshot_refresh = _LiveWorkerRefresh()
        logger.info("web console ready — %d prior jobs loaded", len(app.state.jobs.list_jobs()))
        # Independent loops: streamer-health cache (cheap, always on cadence)
        # and at most one snapshot worker. Health must not wait on FeatureEngine.
        # When vtech-live-features is healthy it owns per-bar appends + scoring;
        # WebUI adopts that cache and only falls back to a spawned snapshot.
        stop = asyncio.Event()
        tasks: list[asyncio.Task[None]] = []

        async def _streamer_health_loop() -> None:
            while not stop.is_set():
                try:
                    await asyncio.to_thread(live.refresh_streamer_health)
                except Exception as exc:  # noqa: BLE001 — never kill the loop
                    logger.warning("live_streamer_health_error: %s", exc)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=_LIVE_HEALTH_CADENCE_S)

        async def _snapshot_loop() -> None:
            ctx = mp.get_context("spawn")
            proc: mp.Process | None = None
            next_run_at = 0.0
            oom_strikes = 0
            refresh = app.state.live_snapshot_refresh
            try:
                while not stop.is_set():
                    try:
                        now = time.monotonic()
                        refresh_requested = refresh.consume()
                        if proc is not None and not proc.is_alive():
                            code = proc.exitcode
                            await asyncio.to_thread(proc.join, 0.0)
                            logger.info("live_snapshot_worker_exit pid=%s code=%s", proc.pid, code)
                            proc = None
                            if code == -9:
                                # Kernel OOM-killed the snapshot child. Respawning
                                # immediately refills the cgroup and the default
                                # OOM policy then stops this API mid-request.
                                oom_strikes += 1
                                delay = min(300.0, 30.0 * oom_strikes)
                                logger.warning(
                                    "live_snapshot_worker_oom_backoff strikes=%s delay_s=%.0f",
                                    oom_strikes, delay,
                                )
                                next_run_at = now + delay
                            else:
                                oom_strikes = 0
                                next_run_at = now + _LIVE_LOOP_CADENCE_S
                        if refresh_requested:
                            next_run_at = now
                            if proc is not None and proc.is_alive():
                                logger.info("live_snapshot_worker_refresh pid=%s", proc.pid)
                                with contextlib.suppress(Exception):
                                    proc.terminate()
                                with contextlib.suppress(Exception):
                                    await asyncio.to_thread(proc.join, 2.0)
                                proc = None
                            elif proc is not None:
                                with contextlib.suppress(Exception):
                                    await asyncio.to_thread(proc.join, 0.0)
                                proc = None
                        if proc is None and now >= next_run_at:
                            runtime_fresh = False
                            with contextlib.suppress(Exception):
                                from src.features.live_runtime import is_runtime_fresh
                                runtime_fresh = is_runtime_fresh()
                            if runtime_fresh:
                                # Adopt snapshot written by vtech-live-features.
                                await asyncio.to_thread(live.adopt_external_snapshot_cache)
                                # If cache is missing/unreadable, score from disk
                                # (resolve_feature_day is disk-only while runtime is fresh).
                                st = live.snapshot_status()
                                if st in (None, "starting", "error"):
                                    proc = await asyncio.to_thread(
                                        _spawn_live_snapshot_process, ctx
                                    )
                                    logger.info(
                                        "live_snapshot_worker_start pid=%s "
                                        "(runtime fresh, cache empty)",
                                        proc.pid,
                                    )
                                next_run_at = now + _LIVE_LOOP_CADENCE_S
                            else:
                                proc = await asyncio.to_thread(_spawn_live_snapshot_process, ctx)
                                logger.info(
                                    "live_snapshot_worker_start pid=%s (runtime fallback)",
                                    proc.pid,
                                )
                                next_run_at = now + _LIVE_LOOP_CADENCE_S
                    except Exception as exc:  # noqa: BLE001 — never kill the loop
                        logger.warning("live_builder_error: %s", exc)
                        next_run_at = time.monotonic() + _LIVE_LOOP_CADENCE_S
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=_LIVE_LOOP_POLL_S)
            finally:
                if proc is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(proc.join, 0.2)
                    if proc.is_alive():
                        logger.warning("live_snapshot_worker_terminate pid=%s", proc.pid)
                        with contextlib.suppress(Exception):
                            proc.terminate()
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(proc.join, 0.5)

        if enable_live_worker:
            # Prefer always-on feature runtime's snapshot cache when healthy so
            # Latest bar / blotter are not stuck on "starting" after WebUI restart.
            runtime_fresh = False
            with contextlib.suppress(Exception):
                from src.features.live_runtime import is_runtime_fresh as _rt_fresh
                runtime_fresh = _rt_fresh()
            if runtime_fresh and live.adopt_external_snapshot_cache():
                logger.info("live_startup_adopted_external_snapshot_cache")
            else:
                # No external runtime: mark cache consumed until local worker rebuilds.
                live.invalidate_cache()
            tasks.append(asyncio.create_task(_streamer_health_loop()))
            tasks.append(asyncio.create_task(_snapshot_loop()))
        try:
            yield
        finally:
            stop.set()
            for task in tasks:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(task, timeout=1.0)
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    app = FastAPI(title="VTech Console", lifespan=lifespan, docs_url="/api/docs",
                  openapi_url="/api/openapi.json")
    # "*" on purpose: bind is 127.0.0.1, so the Host header is whatever the
    # tunnel wrote. A tight allow-list returns 400 and the VS Code Ports
    # preview never finishes loading. See module docstring.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS") \
                and request.headers.get("x-vtech-console") != "1":
            return JSONResponse({"detail": "missing X-VTech-Console header"}, status_code=403)
        return await call_next(request)

    @app.middleware("http")
    async def nocache_shell(request: Request, call_next):
        """Log at request start (uvicorn access lines only fire after the
        response finishes — a hung GET / is invisible). Do not cache the SPA
        shell. ``X-VTech-Rev`` proves which process served the page."""
        logger.info(
            "req_start %s %s host=%s",
            request.method, request.url.path, request.headers.get("host"),
        )
        response = await call_next(request)
        if request.url.path in ("/", "/index.html"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        response.headers["X-VTech-Rev"] = "20260827-ports"
        # Force the browser to open a fresh TCP for the next request so a
        # webui restart cannot leave the VS Code / SSH forwarded socket wedged
        # on a dead PID (the tunnel doesn't propagate RST).
        response.headers["Connection"] = "close"
        return response

    def _jobs(request: Request) -> JobManager:
        return request.app.state.jobs

    def _schedule_live_snapshot_refresh(request: Request) -> str:
        refresh = getattr(request.app.state, "live_snapshot_refresh", None)
        if refresh is None:
            return "unavailable"
        refresh.request()
        return "scheduled"

    def _live_scoping(request: Request) -> set[str]:
        return {j.objective_id for j in _jobs(request).running() if j.kind == "scoping"}

    # ── overview ──────────────────────────────────────────────────────────────
    @app.get("/api/overview")
    def overview(request: Request) -> dict[str, Any]:
        jm = _jobs(request)
        return {
            "active_objective": ds.active_objective_id(),
            "objectives": ds.list_objectives(),
            "services": ds.services_status(),
            "running_jobs": [j.summary() for j in jm.running()],
            "recent_jobs": [j.summary() for j in jm.list_jobs()[:8]],
            "scoping_runs": ds.list_scoping_runs(live_objectives=_live_scoping(request), limit=6),
            "search_runs": ds.list_search_runs(limit=6),
            "vm": ds.vm_info(),
            "mlflow_url": ds._MLFLOW_URL,
        }

    @app.get("/api/calendar/cme")
    def calendar_cme() -> dict[str, Any]:
        """CME full-close / early-close dates (America/Chicago civil dates)."""
        from src.common.cme_holidays import holiday_iso_dates
        return {
            "timezone": "America/Chicago",
            "holidays": holiday_iso_dates(),
        }

    # ── objectives ────────────────────────────────────────────────────────────
    @app.get("/api/objectives")
    def objectives() -> list[dict[str, Any]]:
        return ds.list_objectives()

    @app.get("/api/objectives/_schema")
    def objectives_schema() -> dict[str, Any]:
        return ds.objective_schema()

    @app.get("/api/objectives/{objective_id}")
    def objective_detail(objective_id: str) -> dict[str, Any]:
        try:
            return ds.get_objective(objective_id)
        except FileNotFoundError:
            raise HTTPException(404, f"objective not found: {objective_id}") from None

    @app.post("/api/objectives/{objective_id}/rd-enable")
    def objective_rd_enable(objective_id: str, body: RdEnableBody) -> dict[str, Any]:
        """Enable/disable Research Director loops for this objective (registry LATEST required)."""
        from src.research_director.rd_enable import set_rd_enabled, status_for_objective
        try:
            ds.check_id(objective_id)
        except Exception as exc:  # noqa: BLE001 — ValueError / AttributeError
            raise HTTPException(400, str(exc)) from exc
        # Require known objective yaml
        try:
            ds.get_objective(objective_id)
        except FileNotFoundError:
            raise HTTPException(404, f"objective not found: {objective_id}") from None
        set_rd_enabled(objective_id, body.enabled)
        return status_for_objective(objective_id)

    # ── registered model (per-objective sub-page) ────────────────────────────
    def _regmodel_oid(objective_id: str) -> str:
        try:
            ds.check_id(objective_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc
        try:
            ds.get_objective(objective_id)
        except FileNotFoundError:
            raise HTTPException(404, f"objective not found: {objective_id}") from None
        return objective_id

    @app.get("/api/objectives/{objective_id}/model")
    def regmodel_status(objective_id: str) -> dict[str, Any]:
        """Registered-model status: resolution, Active flag, session, settings."""
        return regmodel.status(_regmodel_oid(objective_id))

    @app.post("/api/objectives/{objective_id}/model/settings")
    def regmodel_save(objective_id: str, body: RegModelSettings) -> dict[str, Any]:
        oid = _regmodel_oid(objective_id)
        regmodel.save_settings(oid, body.model_dump(exclude_none=True))
        return regmodel.status(oid)

    @app.get("/api/objectives/{objective_id}/model/blotter")
    def regmodel_blotter(objective_id: str, force: bool = False) -> dict[str, Any]:
        """Today's persisted signals emitted by the live serving loop."""
        return regmodel.blotter(_regmodel_oid(objective_id), force=force)

    @app.post("/api/objectives/validate")
    def objective_validate(payload: ObjectivePayload) -> dict[str, Any]:
        errors = ds.validate_objective(payload.objective)
        return {"valid": not errors, "errors": errors}

    @app.post("/api/objectives", status_code=201)
    def objective_create(payload: ObjectivePayload) -> dict[str, Any]:
        path = ds.create_objective(payload.objective)
        return {"created": path.name, "objective_id": payload.objective["objective_id"]}

    # ── scoping ───────────────────────────────────────────────────────────────
    @app.get("/api/scoping/runs")
    def scoping_runs(request: Request, objective_id: str | None = None) -> list[dict[str, Any]]:
        return ds.list_scoping_runs(objective_id, live_objectives=_live_scoping(request))

    @app.get("/api/scoping/runs/{objective_id}/{run_ts}")
    def scoping_detail(request: Request, objective_id: str, run_ts: str) -> dict[str, Any]:
        return ds.scoping_run_detail(objective_id, run_ts,
                                     live_objectives=_live_scoping(request))

    @app.get("/api/scoping/runs/{objective_id}/{run_ts}/artifact/{name}")
    def scoping_artifact(objective_id: str, run_ts: str, name: str) -> dict[str, Any]:
        return ds.read_scoping_artifact(objective_id, run_ts, name)

    @app.post("/api/scoping/launch", status_code=201)
    def scoping_launch(request: Request, body: ScopingLaunch) -> dict[str, Any]:
        ds.get_objective(body.objective_id)                      # must exist + be valid id
        argv = ds.build_argv_scoping(body.objective_id, body.dry_run, body.budget)
        job = _jobs(request).launch("scoping", body.objective_id, argv,
                                    label=f"scoping {body.objective_id}"
                                          f"{' (dry-run)' if body.dry_run else ''}")
        return {"job": job.summary()}

    # ── search / optimization ────────────────────────────────────────────────
    @app.get("/api/search/families")
    def search_families() -> list[dict[str, Any]]:
        return ds.catalog_families()

    @app.get("/api/search/runs")
    def search_runs(objective_id: str | None = None) -> list[dict[str, Any]]:
        return ds.list_search_runs(objective_id)

    @app.get("/api/search/runs/{objective_id}/{run_ts}")
    def search_detail(objective_id: str, run_ts: str) -> dict[str, Any]:
        return ds.search_run_detail(objective_id, run_ts)

    @app.post("/api/search/launch", status_code=201)
    def search_launch(request: Request, body: SearchLaunch) -> dict[str, Any]:
        ds.get_objective(body.objective_id)
        argv = ds.build_argv_search(body.objective_id, body.families, body.dry_run,
                                    body.tune_trials, body.no_ensemble, body.manual_spec)
        job = _jobs(request).launch("search", body.objective_id, argv,
                                    label=f"model search {body.objective_id}")
        return {"job": job.summary()}

    # ── custom test ───────────────────────────────────────────────────────────
    @app.get("/api/custom-test/bank")
    def custom_test_bank(store: str | None = None) -> dict[str, Any]:
        return ds.custom_test_bank(store=store)

    @app.get("/api/custom-test/default/{objective_id}")
    def custom_test_default(objective_id: str) -> dict[str, Any]:
        return ds.custom_test_default(objective_id)

    @app.post("/api/custom-test/preview")
    def custom_test_preview(body: CustomTestBoard) -> dict[str, Any]:
        return ds.custom_test_preview(body.model_dump())

    @app.post("/api/custom-test/launch", status_code=201)
    def custom_test_launch(request: Request, body: CustomTestBoard) -> dict[str, Any]:
        persisted = ds.persist_custom_test(body.model_dump())
        argv = ds.build_argv_custom_test(
            body.objective_id,
            persisted["paths"]["manual_spec"],
            persisted["paths"].get("topology"),
            body.dry_run,
            tune_trials=int(persisted.get("tune_trials") or 0),
            walk_forward_overlay=persisted["paths"].get("walk_forward"),
        )
        job = _jobs(request).launch(
            "custom_test", body.objective_id, argv,
            label=f"custom test {body.objective_id}"
                  f"{' (dry-run)' if body.dry_run else ''}",
        )
        return {"job": job.summary(), "compiled": persisted}

    @app.post("/api/custom-test/views")
    def custom_test_save_view(body: CustomTestSavedView) -> dict[str, Any]:
        return ds.save_custom_test_view(
            body.name,
            body.features,
            description=body.description or "",
            overwrite=body.overwrite,
        )

    @app.delete("/api/custom-test/views/{name}")
    def custom_test_delete_view(name: str) -> dict[str, Any]:
        return ds.delete_custom_test_view(name)

    # ── backtest (staged weights via src.serving — same as Live) ─────────────
    @app.get("/api/backtest/models")
    def backtest_models() -> list[dict[str, Any]]:
        return ds.list_backtestable_models()

    @app.get("/api/backtest/browse")
    def backtest_browse(path: str | None = None) -> dict[str, Any]:
        """Directory listing under staging / registry / research/search."""
        return ds.browse_backtest_sources(path)

    @app.post("/api/backtest/run")
    def backtest_run(body: BacktestRun) -> dict[str, Any]:
        stack = None
        if body.stack:
            stack = [leg.model_dump() for leg in body.stack]
        return ds.run_backtest_quicklook(
            source=body.source,
            tau=body.tau, confidence_gate=body.confidence_gate,
            max_positions=body.max_positions,
            mv=body.mv, mv_gate=body.mv_gate,
            p_reach_min=body.p_reach_min, p_move_min=body.p_move_min,
            min_tgt_r=body.min_tgt_r,
            start=body.start, end=body.end, objective_id=body.objective_id,
            sides=body.sides,
            tbbo_data=body.oco or body.tbbo_data or body.tbbo_dev or body.live_data,
            mae_adjust=body.mae_adjust,
            target_adjust=body.target_adjust,
            price_mode=body.price_mode,
            entry_latency_ms=body.entry_latency_ms,
            exit_latency_ms=body.exit_latency_ms,
            params=body.params,
            book=body.book,
            stack=stack)

    @app.post("/api/backtest/chart")
    def backtest_chart(body: BacktestChart) -> dict[str, Any]:
        from src.external_models.chart import build_chart_payload
        stack = [leg.model_dump() for leg in body.stack] if body.stack else None
        return build_chart_payload(
            source=body.source,
            start=body.start,
            end=body.end,
            params=body.params,
            book=body.book,
            stack=stack,
            session_only=body.session_only,
        )

    # ── external models (Pine replicas; candles from PostgreSQL) ─────────────
    @app.get("/api/external-models")
    def external_models() -> dict[str, Any]:
        from src.external_models.runner import list_for_ui
        return list_for_ui()

    @app.get("/api/external-models/book")
    def external_model_book() -> dict[str, Any]:
        from src.external_models.runner import describe_book
        return describe_book()

    @app.put("/api/external-models/book/defaults")
    def external_model_save_book(body: ExternalDefaults) -> dict[str, Any]:
        from src.external_models.store import save_book_defaults
        return save_book_defaults(body.params, note=body.note)

    @app.post("/api/external-models/book/defaults/reset")
    def external_model_reset_book() -> dict[str, Any]:
        from src.external_models.store import reset_book_defaults
        return reset_book_defaults()

    @app.get("/api/external-models/{model_id}")
    def external_model(model_id: str) -> dict[str, Any]:
        from src.external_models.runner import describe_model
        try:
            return describe_model(model_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.put("/api/external-models/{model_id}/defaults")
    def external_model_save_defaults(model_id: str, body: ExternalDefaults) -> dict[str, Any]:
        from src.external_models.store import save_defaults
        try:
            return save_defaults(model_id, body.params, note=body.note)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/external-models/{model_id}/defaults/reset")
    def external_model_reset_defaults(model_id: str) -> dict[str, Any]:
        from src.external_models.store import reset_defaults
        try:
            return reset_defaults(model_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/external-models/{model_id}/hardened")
    def external_model_hardened(model_id: str) -> list[dict[str, Any]]:
        from src.external_models.store import list_hardened
        from src.external_models.catalog import get_model
        try:
            get_model(model_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return list_hardened(model_id)

    @app.post("/api/external-models/{model_id}/hardened", status_code=201)
    def external_model_save_hardened(model_id: str, body: ExternalHardened) -> dict[str, Any]:
        from src.external_models.store import save_hardened
        try:
            return save_hardened(
                model_id, body.name, body.params,
                note=body.note, set_as_defaults=body.set_as_defaults)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/external-models/{model_id}/hardened/{name}")
    def external_model_delete_hardened(model_id: str, name: str) -> dict[str, Any]:
        from src.external_models.store import delete_hardened
        try:
            delete_hardened(model_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True, "name": name}

    # ── live streamer + model prediction (5s cadence) ─────────────────────────
    @app.get("/api/live/streamer")
    def live_streamer() -> dict[str, Any]:
        return live.snapshot()

    @app.get("/api/live/ledger")
    def live_ledger(
        mode: str = "bar",
        tau: float | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Day blotter. ``mode=bar`` (5s first-touch) or ``mode=ticks`` (TBBO limit+OCO)."""
        return live.ledger(mode=mode, tau=tau, force=force)

    @app.get("/api/live/model")
    def live_model() -> dict[str, Any]:
        """List selectable serving bundles + current Live-tab selection/resolution."""
        oid = live.active_objective_id()
        selection = live.get_selection()
        available = live.list_serving_bundles(None)  # all objectives — the dropdown source
        try:
            resolved = live.resolve_live_model(oid)
            trade_oid = (resolved.get("objective_id") or oid)
            trade = live.objective_trade_params(trade_oid)
            return {
                "active_objective": oid,
                "selection": selection,
                "resolved": resolved,
                "trade": trade,
                "available": available,
            }
        except FileNotFoundError as exc:
            return {
                "active_objective": oid,
                "selection": selection,
                "resolved": None,
                "trade": live.objective_trade_params(oid),
                "available": available,
                "error": str(exc),
            }

    @app.post("/api/live/model")
    def live_model_select(request: Request, body: LiveModelSelect) -> dict[str, Any]:
        """Pin a model and schedule a fresh isolated snapshot worker rebuild."""
        try:
            live.set_selection(mode=body.mode, path=body.path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        refresh_state = _schedule_live_snapshot_refresh(request)
        out = live_model()
        out["snapshot_refresh"] = refresh_state
        out["streamer"] = live.snapshot()
        return out

    @app.post("/api/live/reload")
    def live_reload(request: Request) -> dict[str, Any]:
        """Drop in-process model/snapshot cache and schedule a fresh worker rebuild."""
        live.invalidate_cache()
        refresh_state = _schedule_live_snapshot_refresh(request)
        return {
            "ok": True,
            "active_objective": live.active_objective_id(),
            "selection": live.get_selection(),
            "snapshot_refresh": refresh_state,
            "streamer": live.snapshot(),
        }

    # ── research director ─────────────────────────────────────────────────────
    @app.get("/api/director/status")
    def director_status() -> dict[str, Any]:
        return ds.director_status()

    @app.post("/api/director/timer")
    def director_timer(body: TimerAction) -> dict[str, Any]:
        if body.action not in ("enable", "disable"):
            raise HTTPException(400, "action must be 'enable' or 'disable'")
        result = ds.set_timer_enabled(body.action == "enable")
        if not result.get("ok"):
            raise HTTPException(502, result.get("error", "systemctl failed")
                                + " — " + result.get("hint", ""))
        return result

    @app.post("/api/director/run-cycle", status_code=201)
    def director_run_cycle(request: Request, body: DirectorCycle) -> dict[str, Any]:
        argv, env = ds.build_argv_director_cycle(body.dry_run)
        job = _jobs(request).launch("director_cycle", ds.active_objective_id() or "program",
                                    argv, env_extra=env,
                                    label=f"research-director cycle"
                                          f"{' (dry-run)' if body.dry_run else ''}")
        return {"job": job.summary()}

    # ── Live Chart (TBBO candles + overlays) ──────────────────────────────────
    @app.get("/api/chart/meta")
    def chart_meta() -> dict[str, Any]:
        return livechart.meta()

    @app.get("/api/chart/series")
    def chart_series(
        timeframe: str = "1min",
        lookback: str = "session",
        overlays: str | None = None,
        end: str | None = None,
        cvd_origin: str = "session",
        vwap_origin: str = "session",
        ofi_origin: str | None = None,
        ad_origin: str | None = None,
        svp_origin: str | None = None,
        vp_start: str | None = None,
        vp_end: str | None = None,
        profiles: str | None = None,
        emas: str | None = None,
        stochs: str | None = None,
        sar: str | None = None,
        vwap_bands: str | None = None,
        eth_aggregate: str | None = None,
        asof: str | None = None,
    ) -> dict[str, Any]:
        try:
            return livechart.series(
                timeframe=timeframe, lookback=lookback,
                overlays=overlays, end=end,
                cvd_origin=cvd_origin, vwap_origin=vwap_origin, ofi_origin=ofi_origin,
                ad_origin=ad_origin, svp_origin=svp_origin,
                vp_start=vp_start, vp_end=vp_end, profiles=profiles, emas=emas,
                stochs=stochs,
                sar=sar,
                vwap_bands=livechart.parse_bool(vwap_bands, default=False),
                eth_aggregate=livechart.parse_bool(eth_aggregate, default=True),
                asof=asof,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/chart/prints")
    def chart_prints(
        timeframe: str = "1min",
        lookback: str = "session",
        end: str | None = None,
        min_size: str | None = None,
        eth_aggregate: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        try:
            return livechart.prints(
                timeframe=timeframe, lookback=lookback, end=end,
                min_size=min_size,
                eth_aggregate=livechart.parse_bool(eth_aggregate, default=True),
                since=since,
            )
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc

    # ── Live Chart order map (MBO): large resting orders + reload chains ──────
    @app.get("/api/chart/mbo-meta")
    def chart_mbo_meta() -> dict[str, Any]:
        try:
            return mbo_orders_api.coverage()
        except Exception as exc:  # DB down / table missing: the overlay just reports unavailable
            raise HTTPException(503, f"MBO coverage unavailable: {exc}") from exc

    @app.get("/api/chart/mbo-orders")
    def chart_mbo_orders(
        start: str,
        end: str,
        symbol: str | None = None,
        min_size: int = 20,
        clip_min: int = 5,
        approach: float = 10.0,
        chain_n: int = 3,
        chain_dt: float = 2.0,
        chain_min_secs: float = 10.0,
        chain_min_abs: int = 50,
        break_pts: float = 1.0,
    ) -> dict[str, Any]:
        try:
            return mbo_orders_api.order_map(
                start, end, symbol=symbol, min_size=min_size, clip_min=clip_min,
                approach=approach, chain_n=chain_n, chain_dt=chain_dt,
                chain_min_secs=chain_min_secs, chain_min_abs=chain_min_abs,
                break_pts=break_pts,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except mbo_orders_api.OrderMapBusy as exc:
            raise HTTPException(503, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/chart/mbo-order")
    def chart_mbo_order(order_id: int, start: str, end: str, symbol: str) -> dict[str, Any]:
        """One order's MBO rows, sorted by ts_event then sequence (ADR-152)."""
        try:
            return mbo_orders_api.order_events(order_id, start, end, symbol=symbol)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc

    # ── Live Chart Model Design (ADR-117) ─────────────────────────────────────
    @app.post("/api/chart/model-vars")
    def chart_model_vars(body: ChartModelVarsReq) -> dict[str, Any]:
        """Variable catalog for the given chart request (autocomplete + validation)."""
        from src.webui.chart_models import build_var_catalog
        try:
            payload = livechart.series(
                timeframe=body.timeframe, lookback=body.lookback,
                overlays=body.overlays, end=body.end,
                cvd_origin=body.cvd_origin, vwap_origin=body.vwap_origin,
                ofi_origin=body.ofi_origin,
                ad_origin=body.ad_origin, svp_origin=body.svp_origin,
                vp_start=body.vp_start, vp_end=body.vp_end,
                profiles=body.profiles, emas=body.emas, stochs=body.stochs,
                sar=body.sar,
                vwap_bands=body.vwap_bands,
                eth_aggregate=body.eth_aggregate,
            )
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc
        catalog = build_var_catalog(payload)
        return {
            "catalog": catalog,
            "profiles": [{"n": p.get("n"), "label": p.get("label"),
                          "kind": p.get("kind"), "session_tf": p.get("session_tf")}
                         for p in (payload.get("profiles") or [])],
            "emas": [{"n": e.get("n"), "lookback": e.get("lookback"),
                      "key": e.get("key"), "label": e.get("label")}
                     for e in (payload.get("emas") or [])],
            "stochs": [{"n": s.get("n"), "k_period": s.get("k_period"),
                        "d_period": s.get("d_period"), "key": s.get("key"),
                        "label": s.get("label")}
                       for s in (payload.get("stochs") or [])],
            "sar": payload.get("sar"),
            "vwap_bands": payload.get("vwap_bands"),
            "instrument": payload.get("instrument"),
            "timeframe": payload.get("timeframe"),
        }

    @app.post("/api/chart/simulate-model")
    def chart_simulate_model(body: ChartModelSimReq) -> dict[str, Any]:
        """Load the same chart series + prints, compile the rule tree, simulate it."""
        from src.webui.chart_models import build_var_catalog, compile_model, simulate
        from src.webui.chart_models.schema import SchemaError
        c = body.chart
        try:
            payload = livechart.series(
                timeframe=c.timeframe, lookback=c.lookback,
                overlays=c.overlays, end=c.end,
                cvd_origin=c.cvd_origin, vwap_origin=c.vwap_origin,
                ofi_origin=c.ofi_origin,
                ad_origin=c.ad_origin, svp_origin=c.svp_origin,
                vp_start=c.vp_start, vp_end=c.vp_end,
                profiles=c.profiles, emas=c.emas, stochs=c.stochs, sar=c.sar,
                vwap_bands=c.vwap_bands,
                eth_aggregate=c.eth_aggregate,
            )
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(503, str(exc)) from exc
        # Also fetch prints for the same window so lp_*() has data.
        try:
            prints_payload = livechart.prints(
                timeframe=c.timeframe, lookback=c.lookback, end=c.end,
                min_size=body.print_min, eth_aggregate=c.eth_aggregate,
            )
            payload["prints"] = prints_payload.get("prints") or []
        except (TimeoutError, LookupError):
            payload["prints"] = []
        catalog = build_var_catalog(payload)
        try:
            spec = compile_model(body.model, catalog=catalog)
        except SchemaError as exc:
            raise HTTPException(400, {"error": "schema", "detail": str(exc)}) from exc
        result = simulate(
            payload, spec,
            point_value=body.point_value or 20.0,
            session_end=body.session_end,
        )
        result["catalog"] = catalog
        result["model_name"] = spec.get("name") or spec.get("id")
        result["window"] = {
            "start": payload.get("start"),
            "end": payload.get("end"),
            "timeframe": payload.get("timeframe"),
        }
        return result

    @app.get("/api/chart-models")
    def chart_models_list() -> list[dict[str, Any]]:
        from src.webui.chart_models import list_models
        return list_models()

    @app.get("/api/chart-models/{mid}")
    def chart_models_get(mid: str) -> dict[str, Any]:
        from src.webui.chart_models import load_model
        try:
            return load_model(mid)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.put("/api/chart-models/{mid}")
    def chart_models_put(mid: str, body: ChartModelSpec) -> dict[str, Any]:
        from src.webui.chart_models import save_model
        if body.id != mid:
            raise HTTPException(400, "path id and body id must match")
        try:
            return save_model(body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/chart-models/{mid}")
    def chart_models_delete(mid: str) -> dict[str, Any]:
        from src.webui.chart_models import delete_model
        try:
            delete_model(mid)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "id": mid}

    # ── Live Chart notes (scenario snapshots) ─────────────────────────────────
    @app.get("/api/chart/notes")
    def chart_notes_list() -> list[dict[str, Any]]:
        from src.webui import chart_notes
        return chart_notes.list_notes()

    @app.get("/api/chart/notes/{nid}")
    def chart_notes_get(nid: str) -> dict[str, Any]:
        from src.webui import chart_notes
        try:
            return chart_notes.load_note(nid)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/chart/notes", status_code=201)
    def chart_notes_create(body: ChartNoteCreate) -> dict[str, Any]:
        from src.webui import chart_notes
        try:
            return chart_notes.save_note(body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/chart/notes/{nid}")
    def chart_notes_delete(nid: str) -> dict[str, Any]:
        from src.webui import chart_notes
        try:
            chart_notes.delete_note(nid)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "id": nid}


    # ── jobs ──────────────────────────────────────────────────────────────────
    @app.get("/api/jobs")
    def jobs_list(request: Request) -> list[dict[str, Any]]:
        return [j.summary() for j in _jobs(request).list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def job_detail(request: Request, job_id: str) -> dict[str, Any]:
        job = _jobs(request).get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.summary()

    @app.get("/api/jobs/{job_id}/log")
    def job_log(request: Request, job_id: str, offset: int = 0) -> dict[str, Any]:
        if _jobs(request).get(job_id) is None:
            raise HTTPException(404, "job not found")
        return _jobs(request).read_log(job_id, offset)

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(request: Request, job_id: str) -> dict[str, Any]:
        try:
            return _jobs(request).cancel(job_id).summary()
        except KeyError:
            raise HTTPException(404, "job not found") from None

    # ── error mapping + static SPA ───────────────────────────────────────────
    @app.exception_handler(ValueError)
    async def _val(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(yaml.YAMLError)
    async def _yaml(_, exc: yaml.YAMLError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def _nf(_, exc: FileNotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(FileExistsError)
    async def _ex(_, exc: FileExistsError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(RuntimeError)
    async def _rt(_, exc: RuntimeError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VTech web console")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (127.0.0.1 also opens ::1 on the same port)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("binding %s — the console has NO authentication; "
                       "only expose it through an SSH tunnel or reverse proxy with auth",
                       args.host)
    import uvicorn
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app()
    # VS Code Ports labels the forward "localhost:8700". Browsers and the
    # remote forwarder resolve localhost to ::1 first on many machines;
    # IPv4-only bind then never sees the SYN and the tab spins forever.
    if args.host == "127.0.0.1":
        async def _serve_loopback() -> None:
            kw = dict(app=app, port=args.port, log_level="info",
                      timeout_keep_alive=5, timeout_graceful_shutdown=2)
            s4 = uvicorn.Server(uvicorn.Config(host="127.0.0.1", **kw))
            s6 = uvicorn.Server(uvicorn.Config(host="::1", lifespan="off", **kw))
            s6.install_signal_handlers = False
            await asyncio.gather(s4.serve(), s6.serve())
        asyncio.run(_serve_loopback())
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                    timeout_keep_alive=5, timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
