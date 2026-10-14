"""Job manager for the VTech web console — launches and tracks workflow subprocesses.

Every long-running action the UI can trigger (scoping orchestrator, model-selection
search, one Research Director cycle) runs as a detached subprocess whose stdout/stderr
stream to a log file *owned by the child* (the fd survives a console restart). Metadata
is persisted per job under ``data/ops/webui/jobs/<job_id>/meta.json`` so the manager can
reattach to still-running jobs after the web service restarts (systemd unit uses
``KillMode=process`` for exactly this reason).

Safety properties:
  * argv lists only — never ``shell=True``;
  * single-flight per ``(kind, objective_id)`` — no duplicate concurrent runs;
  * cancellation = SIGTERM to the process group, SIGKILL escalation after a grace period.
"""

from __future__ import annotations

from src.common.data_paths import webui_dir

import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.webui.jobs")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOST_REPO = Path("/opt/vtech")

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
LOST = "lost"                    # was running before a console restart; process is gone
TERMINAL = {SUCCEEDED, FAILED, CANCELLED, LOST}

_CANCEL_GRACE_S = 15.0           # SIGTERM → SIGKILL escalation window
_LOG_CHUNK_MAX = 256 * 1024      # max bytes returned per log read


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_job_cwd() -> Path:
    """Directory for console-launched ``python -m src.*`` jobs.

    ``VTECH_REPO_ROOT`` wins. A console imported from a Grok worktree still
    dispatches against ``/opt/vtech`` so Custom Test / search use the host
    checkout (ADR-036 routing + GPU code-bundle identity).
    """
    env_root = os.environ.get("VTECH_REPO_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "src" / "search" / "controller.py").is_file():
            return p.resolve()
    repo = _REPO_ROOT.resolve()
    host = _HOST_REPO.resolve()
    if (".grok/worktrees" in str(repo)
            and host != repo
            and (host / "src" / "search" / "controller.py").is_file()):
        return host
    return repo


def job_env(cwd: Path, env_extra: dict[str, str] | None = None) -> dict[str, str]:
    """Copy the parent env and put *cwd* first on ``PYTHONPATH``."""
    env = {**os.environ, **(env_extra or {})}
    cwd_s = str(cwd)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if cwd_s not in parts:
        env["PYTHONPATH"] = os.pathsep.join([cwd_s] + parts) if parts else cwd_s
    return env


def _proc_starttime(pid: int) -> int | None:
    """Kernel start time (clock ticks) of *pid* from /proc — guards against PID reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])   # field 22 (1-based), after comm
    except (OSError, ValueError, IndexError):
        return None


@dataclass
class Job:
    job_id:      str
    kind:        str                       # "scoping" | "search" | "director_cycle" | ...
    objective_id: str
    argv:        list[str]
    status:      str = RUNNING
    pid:         int | None = None
    pgid:        int | None = None
    starttime:   int | None = None         # /proc starttime — PID-reuse guard
    created_at:  str = field(default_factory=_utcnow)
    ended_at:    str | None = None
    exit_code:   int | None = None
    reattached:  bool = False
    cancel_requested_at: float | None = None
    label:       str = ""

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cancel_requested_at", None)
        return d


class JobManager:
    """Registry of console-launched subprocesses with on-disk persistence."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (webui_dir() / "jobs")
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._load_existing()
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True,
                                        name="webui-job-reaper")
        self._reaper.start()

    # ── persistence ───────────────────────────────────────────────────────────
    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _persist(self, job: Job) -> None:
        meta = self._dir(job.job_id) / "meta.json"
        tmp = meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.summary(), indent=2))
        tmp.replace(meta)

    def _load_existing(self) -> None:
        for meta in sorted(self.root.glob("*/meta.json")):
            try:
                job = Job(**{k: v for k, v in json.loads(meta.read_text()).items()
                             if k in Job.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping unreadable job meta %s: %s", meta, exc)
                continue
            if job.status == RUNNING:
                alive = (job.pid is not None
                         and _proc_starttime(job.pid) == job.starttime
                         and job.starttime is not None)
                if alive:
                    job.reattached = True
                else:
                    job.status, job.ended_at = LOST, _utcnow()
                self._persist(job)
            self._jobs[job.job_id] = job

    # ── launch ────────────────────────────────────────────────────────────────
    def launch(self, kind: str, objective_id: str, argv: list[str], *,
               env_extra: dict[str, str] | None = None, label: str = "",
               cwd: Path | None = None) -> Job:
        with self._lock:
            for j in self._jobs.values():
                if j.status == RUNNING and j.kind == kind and j.objective_id == objective_id:
                    raise RuntimeError(
                        f"a '{kind}' job for objective '{objective_id}' is already "
                        f"running ({j.job_id}) — cancel it first or wait for it to finish")
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            job_id = f"{ts}-{kind}-{uuid.uuid4().hex[:8]}"
            jdir = self._dir(job_id)
            jdir.mkdir(parents=True)
            log_f = open(jdir / "log.txt", "ab")          # child owns this fd
            job_cwd = Path(cwd) if cwd is not None else default_job_cwd()
            env = job_env(job_cwd, env_extra)
            try:
                proc = subprocess.Popen(                   # noqa: S603 — argv list, no shell
                    argv, stdout=log_f, stderr=subprocess.STDOUT,
                    cwd=str(job_cwd), env=env, start_new_session=True)
            finally:
                log_f.close()                              # parent copy; child keeps its own
            job = Job(job_id=job_id, kind=kind, objective_id=objective_id, argv=argv,
                      pid=proc.pid, pgid=proc.pid, starttime=_proc_starttime(proc.pid),
                      label=label)
            self._jobs[job_id] = job
            self._procs[job_id] = proc
            self._persist(job)
            logger.info("launched job %s cwd=%s: %s", job_id, job_cwd, " ".join(argv))
            return job

    # ── queries ───────────────────────────────────────────────────────────────
    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def running(self) -> list[Job]:
        return [j for j in self.list_jobs() if j.status == RUNNING]

    def read_log(self, job_id: str, offset: int = 0) -> dict[str, Any]:
        """Incremental log tail: returns bytes from *offset* (capped) + next offset."""
        log = self._dir(job_id) / "log.txt"
        if not log.exists():
            return {"content": "", "offset": 0, "size": 0}
        size = log.stat().st_size
        offset = max(0, min(offset, size))
        with open(log, "rb") as f:
            f.seek(offset)
            data = f.read(_LOG_CHUNK_MAX)
        return {"content": data.decode("utf-8", errors="replace"),
                "offset": offset + len(data), "size": size}

    # ── cancel ────────────────────────────────────────────────────────────────
    def cancel(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != RUNNING:
            return job
        if job.pgid is None:
            raise RuntimeError("job has no process group recorded")
        job.cancel_requested_at = time.monotonic()
        try:
            os.killpg(job.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass                                          # reaper will finalize
        logger.info("SIGTERM sent to job %s (pgid %s)", job_id, job.pgid)
        return job

    # ── reaper ────────────────────────────────────────────────────────────────
    def _finalize(self, job: Job, exit_code: int | None) -> None:
        if job.cancel_requested_at is not None:
            job.status = CANCELLED
        elif exit_code == 0:
            job.status = SUCCEEDED
        elif exit_code is None:
            job.status = LOST
        else:
            job.status = FAILED
        job.exit_code, job.ended_at = exit_code, _utcnow()
        self._persist(job)
        self._procs.pop(job.job_id, None)
        logger.info("job %s finished: %s (exit=%s)", job.job_id, job.status, exit_code)

    def _reap_once(self) -> None:
        for job in list(self._jobs.values()):
            if job.status != RUNNING:
                continue
            proc = self._procs.get(job.job_id)
            if proc is not None:
                rc = proc.poll()
                if rc is not None:
                    self._finalize(job, rc)
                    continue
            elif job.pid is None or _proc_starttime(job.pid) != job.starttime:
                self._finalize(job, None)                 # reattached job ended
                continue
            if (job.cancel_requested_at is not None
                    and time.monotonic() - job.cancel_requested_at > _CANCEL_GRACE_S
                    and job.pgid is not None):
                try:
                    os.killpg(job.pgid, signal.SIGKILL)
                    logger.warning("SIGKILL escalation for job %s", job.job_id)
                except ProcessLookupError:
                    pass

    def _reap_loop(self) -> None:
        while True:
            try:
                self._reap_once()
            except Exception:                             # noqa: BLE001 — reaper must survive
                logger.exception("job reaper iteration failed")
            time.sleep(2.0)
