"""GPU-side runner — ADR-022 (OCI Queue + content-addressed code bundles).

Runs on the **GPU VM** (boot systemd service, or manually). It long-polls the OCI Queue jobs
channel with a **resource-aware continuous scheduler**: every claimed job is costed against
live host capacity (CPU cores, memory, GPU slots) by the :class:`ResourceGovernor` and starts
the moment it fits — a long DL job no longer gates queued CPU-family work, and concurrent
jobs get disjoint thread budgets instead of all-cores thrash. For each job it:

  1. fetches + verifies + extracts the content-addressed **code bundle** by digest (no git);
  2. runs the candidate in a **subprocess** whose cwd/PYTHONPATH is the extracted bundle and
     whose env carries the granted resource budget (OMP/TF thread caps; ``VTECH_TREE_DEVICE``
     only when a GPU slot was granted) plus the NVIDIA wheel lib path (``LD_LIBRARY_PATH``)
     so TensorFlow's hermetic CUDA libraries resolve and DL jobs actually use the A10;
  3. uploads the OOF parquet + diagnostics to Object Storage and publishes a small result message
     on the objective's results channel; then acks (deletes) the job message.

After idling it powers the GPU VM off **itself** via the OCI API (``--stop-vm-on-idle``), so the
runner and the billed compute stop together — the CPU never issues STOP (ADR-022 revised). Feature
data (``sources/features/timeframe=*/`` hive plus ``parquet_*`` caches) must be present on the block volume.

Dispatch (on the GPU VM):
    python -m gpu_runner.runner --max-jobs 6 --stop-vm-on-idle  # drain, then self-stop when idle
    python -m gpu_runner.runner --once                          # drain current queue then exit
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/opt/vtech")

from src.gpu_runner.dispatch import (  # noqa: E402
    JOBS_CHANNEL,
    make_plane,
    publish_result,
)

logger = logging.getLogger("vtech.gpu_runner")

_DATA_ROOT = Path(os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))
_REPO_ROOT = os.environ.get("VTECH_REPO_ROOT", "/opt/vtech")
_BUNDLE_CACHE = _DATA_ROOT / "code_cache"

# A claimed job's message stays hidden (visibility timeout) for the whole training run so it is
# not redelivered mid-flight; if the worker dies the queue redelivers it after this window.
# 6h default; OCI Queue's maximum is 12h.
_JOB_VISIBILITY_S = int(os.environ.get("GPU_JOB_VISIBILITY_S", str(6 * 3600)))

_TREE_FAMILIES = {"xgboost", "lightgbm"}      # cuda-capable via VTECH_TREE_DEVICE


class JobExecFailed(RuntimeError):
    """job_exec exited non-zero (or never wrote result.json). Carries *out_dir* so
    partial fold checkpoints / job.log can still be published (ADR-073)."""

    def __init__(self, message: str, *, out_dir: Path) -> None:
        super().__init__(message)
        self.out_dir = Path(out_dir)


# ── Resource-aware admission (semaphore-style continuous scheduling) ─────────

@dataclass
class JobAlloc:
    """Granted resource reservation for one job (drives the subprocess env)."""
    cpus: int = 2
    mem_gb: float = 4.0
    gpu_slots: int = 0
    family: str = ""
    compute_class: str = "cpu"


def _family_profile(family: str) -> tuple[str, str]:
    """(compute_class, cost_class) for *family* — catalog lookup with a safe fallback."""
    try:
        from src.models.catalog import get_entry
        e = get_entry(family)
        return e.compute_class, e.cost_class
    except Exception:  # noqa: BLE001 — unknown family / catalog unavailable
        return ("gpu", "expensive") if family in (
            "attention_lstm", "tcn", "tcn_hazard", "path_tcn", "patch_hazard",
            "transformer", "cnn", "rl") else ("cpu", "moderate")


def _gpu_slot_capacity() -> int:
    """GPU slots this host offers (0 when no NVIDIA GPU). Env-overridable.

    Default 2 slots per A10: with TF memory-growth on, a small DL job and a cuda
    tree job coexist comfortably in 23 GB of VRAM.
    """
    env = os.environ.get("GPU_RUNNER_GPU_SLOTS", "").strip()
    if env:
        return max(0, int(env))
    return 2 if shutil.which("nvidia-smi") else 0


def _mem_total_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return float(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 16.0


class ResourceGovernor:
    """Admission control: fit each claimed job to LIVE host capacity.

    Gauges the hardware itself (``os.cpu_count``, ``/proc/meminfo``, ``nvidia-smi``
    presence) instead of a hardcoded worker count, and costs each job from its
    family's catalog profile (compute_class / cost_class). ``try_admit`` reserves and
    returns a :class:`JobAlloc` (or ``None`` when the job must wait); ``release``
    returns the reservation. Thread-safe.
    """

    def __init__(self) -> None:
        self.total_cpus = os.cpu_count() or 4
        self.cpu_reserve = int(os.environ.get("GPU_RUNNER_CPU_RESERVE", "2"))
        self.mem_budget_gb = _mem_total_gb() * 0.85
        self.gpu_total = _gpu_slot_capacity()
        self.cpu_used = 0
        self.mem_used_gb = 0.0
        self.gpu_used = 0
        self._lock = threading.Lock()

    def _cost(self, spec: dict) -> JobAlloc:
        family = str(spec.get("family") or "")
        if spec.get("smoke"):
            return JobAlloc(cpus=1, mem_gb=0.5, gpu_slots=0, family="__smoke__")
        cc, cost_class = _family_profile(family)
        usable = max(1, self.total_cpus - self.cpu_reserve)
        if cc == "gpu":
            # DL: one GPU slot + a modest CPU share for the input pipeline.
            return JobAlloc(cpus=min(6, max(2, usable // 4)), mem_gb=24.0,
                            gpu_slots=1 if self.gpu_total else 0,
                            family=family, compute_class=cc)
        cpus = int(os.environ.get("GPU_RUNNER_CPU_PER_JOB", "0")) or max(4, usable // 3)
        mem = 16.0 if cost_class == "expensive" else 8.0
        # Tree families opportunistically take a GPU slot (cuda hist) when one is free.
        flex_gpu = 1 if (self.gpu_total and family in _TREE_FAMILIES) else 0
        return JobAlloc(cpus=min(cpus, usable), mem_gb=mem, gpu_slots=flex_gpu,
                        family=family, compute_class=cc)

    def try_admit(self, spec: dict) -> JobAlloc | None:
        """Reserve resources for *spec* if they fit NOW; else None (caller holds the job)."""
        cost = self._cost(spec)
        with self._lock:
            cpu_free = max(1, self.total_cpus - self.cpu_reserve) - self.cpu_used
            mem_free = self.mem_budget_gb - self.mem_used_gb
            gpu_free = self.gpu_total - self.gpu_used
            if cost.cpus > cpu_free and self.cpu_used > 0:
                return None                       # wait (empty ledger always admits — no starvation)
            if cost.mem_gb > mem_free and self.mem_used_gb > 0:
                return None
            granted_gpu = cost.gpu_slots
            if cost.compute_class == "gpu" and granted_gpu > max(gpu_free, 0):
                if self.gpu_total and self.gpu_used > 0:
                    return None                   # DL waits for its GPU slot
                granted_gpu = max(gpu_free, 0)
            else:                                 # flex: trees fall back to CPU, never wait
                granted_gpu = min(granted_gpu, max(gpu_free, 0))
            alloc = JobAlloc(cpus=min(cost.cpus, max(cpu_free, 1)), mem_gb=cost.mem_gb,
                             gpu_slots=granted_gpu, family=cost.family,
                             compute_class=cost.compute_class)
            self.cpu_used += alloc.cpus
            self.mem_used_gb += alloc.mem_gb
            self.gpu_used += alloc.gpu_slots
            return alloc

    def release(self, alloc: JobAlloc | None) -> None:
        if alloc is None:
            return
        with self._lock:
            self.cpu_used = max(0, self.cpu_used - alloc.cpus)
            self.mem_used_gb = max(0.0, self.mem_used_gb - alloc.mem_gb)
            self.gpu_used = max(0, self.gpu_used - alloc.gpu_slots)

    def snapshot(self) -> str:
        return (f"cpu {self.cpu_used}/{max(1, self.total_cpus - self.cpu_reserve)} "
                f"mem {self.mem_used_gb:.0f}/{self.mem_budget_gb:.0f}GB "
                f"gpu {self.gpu_used}/{self.gpu_total}")


def _nvidia_lib_paths() -> str:
    """Colon-joined ``site-packages/nvidia/*/lib`` dirs of THIS interpreter's venv.

    TF >= 2.16 ships its CUDA runtime as ``nvidia-*-cu12`` wheels but does not put their
    ``lib`` dirs on the dynamic loader path — without this, ``dlopen`` fails and TF
    silently falls back to CPU (observed live 2026-07-11: A10 at 0% under an LSTM job).
    """
    try:
        import site
        roots = list(site.getsitepackages())
    except Exception:  # noqa: BLE001
        roots = []
    libs: list[str] = []
    for root in roots:
        libs.extend(sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib"))))
    return ":".join(dict.fromkeys(libs))


def _extract_bundle(store, code_digest: str) -> tuple[Path, str]:
    """Fetch + verify + extract the content-addressed code bundle by digest (cached on the block
    volume). An empty digest falls back to the deployed repo (dev / no-bundle path)."""
    if not code_digest:
        return Path(_REPO_ROOT), ""
    from src.common.code_bundle import fetch_and_extract
    dest = fetch_and_extract(store, code_digest, _BUNDLE_CACHE)
    return dest, code_digest


def _execute_bundle_job(store, spec: dict, *, alloc: JobAlloc | None = None) -> dict:
    """Run one candidate in a subprocess pinned to the dispatched code bundle.

    The subprocess (``src.gpu_runner.job_exec``) runs with cwd/PYTHONPATH = the extracted bundle,
    so it executes exactly the dispatched code version regardless of the runner's own code.
    The env additionally carries the NVIDIA wheel lib path (``LD_LIBRARY_PATH`` — lets TF's
    hermetic CUDA libraries resolve so DL jobs use the GPU) and, when *alloc* is given, the
    job's resource budget: OMP/BLAS/TF thread caps sized to the granted cores, and
    ``VTECH_TREE_DEVICE`` only when a GPU slot was actually granted (prevents VRAM fights).
    Returns ``{"result": <dict from result.json>, "out_dir": Path, "ran_digest": str}``.
    """
    bundle_dir, ran_digest = _extract_bundle(store, str(spec.get("code_digest") or ""))
    out_dir = Path(tempfile.mkdtemp(prefix=f"gpujob_{spec.get('run_id', 'x')}_"))
    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, default=str))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{bundle_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    libs = _nvidia_lib_paths()
    if libs:
        env["LD_LIBRARY_PATH"] = f"{libs}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # share the A10 across slots
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING") or "utf-8"
    run_id = str(spec.get("run_id") or "unknown")
    try:
        from src.common.data_paths import gpu_jobs_dir
        durable = gpu_jobs_dir() / run_id
        durable.mkdir(parents=True, exist_ok=True)
        env["VTECH_JOB_PROGRESS_DIR"] = str(durable)
    except Exception:  # noqa: BLE001 — progress dir is best-effort
        durable = None
    if alloc is not None:
        n = str(max(1, alloc.cpus))
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            env[k] = n
        env["TF_NUM_INTRAOP_THREADS"] = n
        env["TF_NUM_INTEROP_THREADS"] = "2"
        if alloc.family in _TREE_FAMILIES:
            if alloc.gpu_slots > 0:
                env["VTECH_TREE_DEVICE"] = "cuda"
            else:
                env.pop("VTECH_TREE_DEVICE", None)
    log_path = out_dir / "job.log"
    extra_logs = [durable / "job.log"] if durable is not None else []
    rc = _tee_subprocess(
        [sys.executable, "-m", "src.gpu_runner.job_exec", str(spec_path), str(out_dir)],
        cwd=str(bundle_dir), env=env, log_path=log_path, extra_logs=extra_logs,
    )
    result_json = out_dir / "result.json"
    if rc != 0 or not result_json.exists():
        err_txt = ""
        try:
            err_txt = log_path.read_text()[-4000:]
        except OSError:
            err_txt = f"job_exec rc={rc} (no job.log)"
        (out_dir / "job_stderr.log").write_text(err_txt)
        raise JobExecFailed(
            f"job_exec failed (rc={rc}) [full log: {log_path}]: …{err_txt[-1200:]}",
            out_dir=out_dir,
        )
    return {"result": json.loads(result_json.read_text()),
            "out_dir": out_dir, "ran_digest": ran_digest}


def _tee_subprocess(
    cmd: list[str],
    *,
    cwd: str,
    env: dict,
    log_path: Path,
    extra_logs: list[Path] | None = None,
) -> int:
    """Run *cmd*, streaming stdout+stderr live to the journal and *log_path* (ADR-073)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handles: list = []
    try:
        logf = log_path.open("w", encoding="utf-8")
        handles.append(logf)
        for extra in extra_logs or []:
            extra.parent.mkdir(parents=True, exist_ok=True)
            handles.append(extra.open("w", encoding="utf-8"))
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            for fh in handles:
                fh.write(line)
                fh.flush()
        return int(proc.wait())
    finally:
        for fh in handles:
            try:
                fh.close()
            except OSError:
                pass


def _run_and_publish(queue, store, msg, *, alloc: JobAlloc | None = None) -> None:
    """Execute a claimed job, publish artifacts + a result message, then ack. Never raises."""
    body = msg.body
    run_id = body.get("run_id")
    objective_id = body.get("objective_id")
    code_digest = str(body.get("code_digest") or "")
    fail_dir: Path | None = None
    try:
        # Pass alloc only when granted — keeps the two-arg contract for tests/fakes.
        out = (_execute_bundle_job(store, body, alloc=alloc) if alloc is not None
               else _execute_bundle_job(store, body))
        publish_result(queue, store, objective_id=objective_id, run_id=run_id,
                       code_digest=code_digest, ran_digest=out["ran_digest"],
                       result=out["result"], out_dir=out["out_dir"])
        logger.info("job_done run_id=%s status=%s", run_id, out["result"].get("status"))
    except Exception as exc:  # noqa: BLE001 — a failed job must not kill the runner
        logger.error("job_failed run_id=%s err=%s", run_id, exc)
        fail_dir = getattr(exc, "out_dir", None)
        if fail_dir is None:
            fail_dir = Path(tempfile.mkdtemp())
        # Also pick up SSH-visible fold checkpoints if the temp dir is empty.
        try:
            from src.common.data_paths import gpu_jobs_dir
            durable = gpu_jobs_dir() / str(run_id or "unknown")
            folds = durable / "folds"
            dest = Path(fail_dir)
            if folds.is_dir() and not any(dest.glob("folds/*")):
                shutil.copytree(folds, dest / "folds", dirs_exist_ok=True)
            prog = durable / "progress.json"
            if prog.is_file() and not (dest / "progress.json").exists():
                shutil.copy2(prog, dest / "progress.json")
        except Exception:  # noqa: BLE001
            pass
        publish_result(queue, store, objective_id=objective_id, run_id=run_id,
                       code_digest=code_digest, ran_digest="",
                       result={"status": "failed", "family": body.get("family"),
                               # TAIL — the exception (incl. any nested job stderr tail)
                               # ends the formatted traceback.
                               "error": traceback.format_exc()[-1600:], "elapsed_s": 0.0},
                       out_dir=fail_dir)
    finally:
        # Ack: on a crash before this point the visibility timeout redelivers the job.
        queue.delete(msg)


def _self_stop_if_still_idle(queue, store, pool, args) -> bool:
    """Final guard before powering off (closes the enqueue race): one last poll.

    If a job arrived in the meantime, drain it and return ``True`` (the runner stays up).
    Otherwise SOFTSTOP this GPU VM via the OCI API so the runner and the billed compute halt
    together, and return ``False`` (the runner exits as the box powers down).
    """
    final = queue.get(channel=JOBS_CHANNEL, visibility_s=_JOB_VISIBILITY_S,
                      max_msgs=max(1, getattr(args, "max_jobs", 1)), wait_s=2)
    if final:
        logger.info("claimed %d job(s) on final idle check — staying up", len(final))
        for f in [pool.submit(_run_and_publish, queue, store, m) for m in final]:
            f.result()
        return True
    logger.info("stopping GPU VM (SOFTSTOP self via OCI API) — runner + compute halt together")
    try:
        from src.common.gpu_provider import GpuProvider
        GpuProvider().stop_self()
    except Exception as exc:  # noqa: BLE001 — never crash-loop on a stop failure
        logger.error("self-stop failed (%s) — exiting; the idle VM will need a manual/CPU STOP",
                     exc)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="ADR-022 — GPU job runner (OCI Queue + bundles)")
    ap.add_argument("--max-jobs", "--workers", dest="max_jobs", type=int,
                    default=int(os.environ.get("GPU_RUNNER_MAX_JOBS",
                                               os.environ.get("GPU_RUNNER_WORKERS", "6"))),
                    help="Hard cap on concurrent jobs (safety net; real admission is "
                         "resource-based via the ResourceGovernor)")
    ap.add_argument("--poll-secs", type=float, default=20.0,
                    help="Long-poll wait per queue GET when idle (OCI Queue max is 30s)")
    ap.add_argument("--idle-exit-secs", type=float, default=300.0,
                    help="Exit after this long with no work (0 = run forever). With "
                         "--stop-vm-on-idle the runner also powers the GPU VM off at this point.")
    ap.add_argument("--stop-vm-on-idle", action="store_true",
                    help="On idle-exit, SOFTSTOP this GPU VM via the OCI API (instance principal) "
                         "so runner + billed compute halt together. Off for dev/--once runs.")
    ap.add_argument("--once", action="store_true", help="Drain the current queue, then exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    queue, store = make_plane()
    _worker_id = socket.gethostname()
    governor = ResourceGovernor()
    max_jobs = max(1, args.max_jobs)
    logger.info("gpu_runner_start backend=%s max_jobs=%d capacity[%s]",
                queue.backend_kind, max_jobs, governor.snapshot())

    # ── Continuous resource-aware scheduler (semaphore-style) ────────────────
    # running: future -> (msg, alloc); held: claimed msgs waiting for resources
    # (they stay invisible under the 6h visibility window — we own them and will
    # start each one the moment its reservation fits).
    running: dict = {}
    held: list = []
    last_work = time.time()
    with ThreadPoolExecutor(max_workers=max_jobs) as pool:
        while True:
            # 1. Reap finished jobs → release their reservations.
            for fut in [f for f in running if f.done()]:
                _msg, alloc = running.pop(fut)
                governor.release(alloc)
                fut.result()          # _run_and_publish never raises; surfaces bugs in dev
                last_work = time.time()
                logger.info("slot_freed capacity[%s] running=%d held=%d",
                            governor.snapshot(), len(running), len(held))

            # 2. Start held jobs whose resources now fit (FIFO).
            still_held = []
            for msg in held:
                alloc = governor.try_admit(msg.body) if len(running) < max_jobs else None
                if alloc is not None:
                    running[pool.submit(_run_and_publish, queue, store, msg, alloc=alloc)] = (
                        msg, alloc)
                    last_work = time.time()
                    logger.info("job_started family=%s alloc[cpu=%d gpu=%d] capacity[%s]",
                                alloc.family, alloc.cpus, alloc.gpu_slots, governor.snapshot())
                else:
                    still_held.append(msg)
            held = still_held

            # 3. Claim more work whenever there is headroom (short poll while busy,
            #    long poll when fully idle — never blocked behind a running batch).
            headroom = max_jobs - len(running) - len(held)
            if headroom > 0:
                wait_s = min(int(args.poll_secs), 30) if not running else 3
                msgs = queue.get(channel=JOBS_CHANNEL, visibility_s=_JOB_VISIBILITY_S,
                                 max_msgs=min(headroom, 10), wait_s=wait_s)
                for msg in msgs:
                    last_work = time.time()
                    alloc = governor.try_admit(msg.body) if len(running) < max_jobs else None
                    if alloc is not None:
                        running[pool.submit(_run_and_publish, queue, store, msg, alloc=alloc)] = (
                            msg, alloc)
                        logger.info("job_started family=%s alloc[cpu=%d gpu=%d] capacity[%s]",
                                    alloc.family, alloc.cpus, alloc.gpu_slots,
                                    governor.snapshot())
                    else:
                        held.append(msg)
                        logger.info("job_held family=%s (resources busy) capacity[%s]",
                                    msg.body.get("family"), governor.snapshot())
            else:
                time.sleep(1.0)

            if running or held:
                continue
            # ── Fully idle ────────────────────────────────────────────────────
            if args.once:
                break
            if args.idle_exit_secs and (time.time() - last_work) > args.idle_exit_secs:
                logger.info("idle_exit after %.0fs with no work", time.time() - last_work)
                if args.stop_vm_on_idle and _self_stop_if_still_idle(queue, store, pool, args):
                    last_work = time.time()
                    continue   # a job arrived on the final check — stay up and drain it
                break
            time.sleep(min(args.poll_secs, 2.0))   # local backend has no server-side long-poll

    logger.info("gpu_runner_done")


if __name__ == "__main__":
    main()
