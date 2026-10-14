"""Shared GPU→CPU dispatch fallback.

When the GPU VM does not come up after the start command, jobs run in-process on the CPU VM
instead of stranding in the queue until the wave times out (hours). Used by the search
controller (``src/search/controller.py::_dispatch_gpu_jobs``) — which is also the execution
surface the Research Director's auto-validation stage
(``src/research_director/orchestrator.py::_stage_validate``) drives via ``run_search``, so the
RD's validation training inherits the same GPU→CPU fallback.

Design — **confirm RUNNING before enqueue**:

    caller: entry = spec_to_entry(...) / catalog entries
            if gpu_ready_or_cpu(ocid):        # issues START, blocks up to boot timeout
                enqueue_job(...); long-poll results       # GPU path (unchanged)
            else:
                _run_one(...)                              # CPU in-process fallback

This ordering matches the operational intent exactly ("if the VM isn't on after the start
command, re-route to CPU") and is **provably free of double-execution**: nothing is enqueued
unless the VM is confirmed RUNNING, so a late boot can never pick up a job we already ran on the
CPU. There is no queue-drain race to reason about. When the VM is already up, ``wait_running``
returns immediately (no added latency); the wait only occurs on a genuine cold boot — which is
exactly when the caller would wait for the VM anyway.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vtech.gpu_runner.fallback")

# How long to wait for the VM to reach RUNNING after the start command before falling back to CPU.
# A cold A10 boot is typically ~2 min; 300s leaves margin. Override per deploy via env.
GPU_BOOT_TIMEOUT_S = int(os.environ.get("GPU_BOOT_TIMEOUT_S", "300"))


def gpu_ready_or_cpu(
    ocid: str,
    *,
    boot_timeout_s: int | None = None,
    provider: object | None = None,
) -> bool:
    """Issue the GPU start command and block until the VM is RUNNING or the boot timeout elapses.

    Returns:
        ``True``  — the GPU VM is live; the caller should enqueue jobs and long-poll results.
        ``False`` — the VM did not come up after the start command (boot timeout, or the OCI
                    control-plane call failed); the caller should run the job(s) in-process on the
                    CPU VM via ``_run_one``.

    Never raises: any provider/OCI error is treated as "GPU unavailable → CPU fallback", because a
    stranded research cycle is worse than a slow CPU run.
    """
    if boot_timeout_s is None:
        boot_timeout_s = GPU_BOOT_TIMEOUT_S
    prov = provider or _default_provider()
    try:
        prov.ensure_running(ocid)                       # idempotent START; no-op if already up
        prov.wait_running(ocid, timeout_s=boot_timeout_s)  # returns immediately if RUNNING
        return True
    except TimeoutError:
        logger.warning(
            "gpu_boot_timeout: VM %s did not reach RUNNING within %ds after START — "
            "routing job(s) to the CPU VM (in-process)", _short(ocid), boot_timeout_s)
        return False
    except Exception as exc:  # noqa: BLE001 — OCI/control-plane failure ≠ fatal; fall back to CPU
        logger.warning(
            "gpu_start_failed: could not bring up VM %s (%s) — routing job(s) to the CPU VM",
            _short(ocid), exc)
        return False


def _default_provider() -> object:
    from src.common.gpu_provider import GpuProvider  # noqa: PLC0415
    return GpuProvider()


def _short(ocid: str) -> str:
    return ocid[-12:] if len(ocid) > 12 else ocid
