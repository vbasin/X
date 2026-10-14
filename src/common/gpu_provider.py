"""OCI GPU VM lifecycle manager.

Wraps all OCI Compute SDK calls for the Research Director's GPU execution plane.
The GPU VM is a **persistent-stopped** OCI instance: started before a job, stopped
after it completes. Never terminated — only started and stopped.

Auth: CPU side uses user key-based ~/.oci/config (same as sync_to_oci.sh); override via
OCI_CONFIG_FILE. On the GPU VM (no key file) it falls back to **instance principal** auth, which
is what lets the runner stop its own instance when idle (ADR-022 revised).

Required env vars (add to .env):
  GPU_INSTANCE_OCID  — OCID of the persistent GPU VM
  GPU_REGION         — region the GPU VM lives in (e.g. us-phoenix-1); defaults to the
                       ~/.oci/config region. START/STOP are OCI-API control-plane calls,
                       so a cross-region GPU VM needs no network path — only this region.
  GPU_SSH_HOST       — private IP / hostname (ONLY for SSH-push dispatch; not needed under
                       the Object-Storage poller model)
  GPU_SSH_USER       — SSH login user (default: ubuntu)
  GPU_SSH_KEY_PATH   — path to the SSH private key (default: ~/.ssh/id_rsa)

Safety invariants enforced here, not in callers:
  - start()    → instance_action("START")             — never launch_instance
  - stop()     → instance_action("SOFTSTOP") with hard "STOP" fallback
                                                       — never terminate_instance
  - run_job()  → nohup SSH background dispatch; SSH returns immediately
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# OCI lifecycle state strings
_STATE_RUNNING  = "RUNNING"
_STATE_STOPPED  = "STOPPED"
_STATE_STOPPING = "STOPPING"
_STATE_STARTING = "STARTING"

_START_TIMEOUT_S = 300   # 5 min — typical A10 cold-boot is ~2 min
_POLL_INTERVAL_S = 10    # seconds between lifecycle state polls


@dataclass
class JobSpec:
    """Portable description of a GPU training job passed to the runner.

    .. deprecated:: ADR-022
        Legacy SSH-dispatch spec. The current plane dispatches via OCI Queue + content-addressed
        code bundles (``src/gpu_runner/dispatch.py``) and starts the VM with
        :meth:`GpuProvider.ensure_running`. Retained only for ``research_director/conductor.py``
        until that subsystem migrates to the queue plane.
    """
    experiment_id: int
    run_id: str                              # UUID string; keyed in OCI heartbeat objects
    git_sha: str
    model_type: str                          # keras | sklearn | xgboost | custom
    seed: int
    folds: int
    config_diff: dict[str, Any] = field(default_factory=dict)
    implementation_path: str | None = None
    implementation_spec: dict[str, Any] | None = None
    datasets: dict[str, Any] | None = None
    metric: str = "val_precision_at_tau"
    delta_min: float = 0.0
    est_epochs: int = 30
    heartbeat_bucket: str = "vtech_hot"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)


class GpuProviderError(RuntimeError):
    """Unrecoverable GPU provider error — caller should mark journal FAILED_RUN."""


class GpuProvider:
    """OCI GPU VM lifecycle manager.

    All methods are synchronous.  The Conductor wraps long-running calls in a
    thread when it needs non-blocking behaviour.
    """

    def __init__(self, *, config_file: str | None = None, region: str | None = None) -> None:
        config_file = config_file or os.environ.get(
            "OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config")
        )
        import oci  # noqa: PLC0415
        # The GPU VM may live in a different region than the CPU VM's OCI config
        # (e.g. CPU config = us-chicago-1, GPU VM = us-phoenix-1). Target the GPU
        # instance's region for lifecycle calls. START/STOP are control-plane API
        # calls to the OCI endpoint — they do NOT require any network path to the VM.
        region = region or os.environ.get("GPU_REGION")
        if config_file and os.path.exists(config_file):
            # CPU side: user key-based auth via ~/.oci/config.
            cfg = oci.config.from_file(config_file)
            if region:
                cfg = {**cfg, "region": region}
            self._cfg, self._signer = cfg, None
            self._compute = oci.core.ComputeClient(cfg)
        else:
            # GPU side: no ~/.oci/config — instance principal (the VM's own identity), which
            # lets the runner stop its own instance (ADR-022 revised). Requires the GPU's
            # dynamic group to be allowed to manage/use instances in its compartment.
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            cfg = {"region": region} if region else {}
            self._cfg, self._signer = cfg, signer
            self._compute = oci.core.ComputeClient(cfg, signer=signer)
        self._net: Any = None

    # ── instance lifecycle ───────────────────────────────────────────────────

    def start(self, instance_id: str, *, timeout_s: int = _START_TIMEOUT_S) -> None:
        """Power on the GPU VM and block until RUNNING.

        Idempotent: no-op if the instance is already RUNNING.
        Calls instance_action("START") — never launch_instance.
        """
        state = self._state(instance_id)
        if state == _STATE_RUNNING:
            log.info("GpuProvider.start: %s already RUNNING — no-op", _short(instance_id))
            return
        if state not in (_STATE_STOPPED, _STATE_STARTING):
            raise GpuProviderError(
                f"Cannot start instance {_short(instance_id)} in state '{state}'"
            )
        if state == _STATE_STOPPED:
            log.info("GpuProvider.start: sending START to %s", _short(instance_id))
            self._compute.instance_action(instance_id, "START")
        self.wait_running(instance_id, timeout_s=timeout_s)

    def ensure_running(self, instance_id: str, *, stopping_wait_s: int = 180) -> bool:
        """Idempotent trigger (ADR-022): issue START so a queued job has a live consumer.

        This is the sole CPU-side lifecycle action under the autonomous-worker model — the GPU
        runner powers the VM *itself* off after an idle timeout, so the CPU never issues STOP.
        Safe for any number of concurrent callers: it never stops a running VM and never
        double-starts.

        Fire-and-forget: does NOT block for RUNNING (unlike :meth:`start`) — the queue consumer
        boots and drains on its own schedule. Returns ``True`` if a START was issued, ``False``
        if the VM was already RUNNING/STARTING.

        If the VM is **STOPPING** (the runner is mid self-stop after going idle), this briefly
        waits for it to reach STOPPED and then STARTs it — otherwise a job enqueued during that
        power-off window would strand until the next dispatch.
        """
        state = self._state(instance_id)
        if state == _STATE_STOPPING and stopping_wait_s > 0:
            deadline = time.monotonic() + stopping_wait_s
            while time.monotonic() < deadline and state == _STATE_STOPPING:
                log.info("GpuProvider.ensure_running: %s is STOPPING — waiting to re-START",
                         _short(instance_id))
                time.sleep(_POLL_INTERVAL_S)
                state = self._state(instance_id)
        if state == _STATE_STOPPED:
            log.info("GpuProvider.ensure_running: sending START to %s", _short(instance_id))
            self._compute.instance_action(instance_id, "START")
            return True
        log.info("GpuProvider.ensure_running: %s is %s — no START issued",
                 _short(instance_id), state)
        return False

    def stop(self, instance_id: str, *, graceful: bool = True) -> None:
        """Power off the GPU VM.  Uses SOFTSTOP with a hard-STOP fallback.

        Fire-and-forget: does not wait for STOPPED state.
        Never calls terminate_instance — that permanently deletes the VM.
        """
        state = self._state(instance_id)
        if state in (_STATE_STOPPED, _STATE_STOPPING):
            log.info("GpuProvider.stop: %s already stopped/stopping — no-op", _short(instance_id))
            return
        action = "SOFTSTOP" if graceful else "STOP"
        log.info("GpuProvider.stop: sending %s to %s", action, _short(instance_id))
        try:
            self._compute.instance_action(instance_id, action)
        except Exception as exc:
            log.warning("GpuProvider.stop: %s failed (%s) — sending hard STOP", action, exc)
            self._compute.instance_action(instance_id, "STOP")

    def stop_self(self, instance_id: str | None = None, *, graceful: bool = True) -> None:
        """Stop THIS GPU VM via the OCI API (ADR-022 revised).

        Called by the runner when it goes idle so the runner **and** the billed compute halt
        together — instead of an in-guest ``shutdown`` that powers off the OS but leaves the OCI
        instance ``RUNNING`` (still billed). ``instance_id`` defaults to ``$GPU_INSTANCE_OCID``.
        Uses instance-principal auth on the GPU (see :meth:`__init__`).
        """
        instance_id = instance_id or os.environ["GPU_INSTANCE_OCID"]
        self.stop(instance_id, graceful=graceful)

    def health_check(self, instance_id: str) -> bool:
        """True if lifecycle_state is RUNNING."""
        return self._state(instance_id) == _STATE_RUNNING

    def wait_running(self, instance_id: str, *, timeout_s: int = _START_TIMEOUT_S) -> None:
        """Block until lifecycle_state == RUNNING or raise TimeoutError."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._state(instance_id)
            if state == _STATE_RUNNING:
                log.info("GpuProvider.wait_running: %s is RUNNING", _short(instance_id))
                return
            log.debug("wait_running: state=%s — polling in %ds", state, _POLL_INTERVAL_S)
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(
            f"Instance {_short(instance_id)} did not reach RUNNING within {timeout_s}s"
        )

    # ── job dispatch ─────────────────────────────────────────────────────────

    def run_job(self, job: JobSpec) -> None:
        """SSH to the GPU VM and launch the GPU runner as a detached background process.

        .. deprecated:: ADR-022
            Legacy SSH push-dispatch. Superseded by the OCI Queue + code-bundle plane
            (``src/gpu_runner/dispatch.py``) + :meth:`ensure_running`; no inbound SSH is used by
            the search controller or runner anymore. Retained only until
            ``research_director/conductor.py`` migrates to the queue plane.

        The SSH session returns immediately after the nohup launch.  The runner
        communicates progress back via OCI Object Storage heartbeats
        (vtech_hot/heartbeats/{run_id}.json) and Postgres journal updates.

        SSH config (env):
          GPU_SSH_HOST     — private IP / hostname
          GPU_SSH_USER     — default: ubuntu
          GPU_SSH_KEY_PATH — default: ~/.ssh/id_rsa
        """
        host = os.environ.get("GPU_SSH_HOST", "")
        user = os.environ.get("GPU_SSH_USER", "ubuntu")
        key  = os.environ.get("GPU_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa"))

        if not host:
            raise GpuProviderError("GPU_SSH_HOST is not set — cannot dispatch job")

        remote_spec = f"/tmp/job_spec_{job.run_id}.json"
        remote_log  = f"/tmp/runner_{job.run_id}.log"
        venv_python = "/opt/vtech/venv/bin/python"

        runner_cmd = (
            f"cd /opt/vtech && "
            f"git fetch --quiet && git checkout --quiet {job.git_sha} && "
            f"set -a && source .env && set +a && "
            f"nohup {venv_python} -m gpu_runner.runner"
            f" --spec {remote_spec}"
            f" >> {remote_log} 2>&1 &"
        )

        ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30",
            "-i", key,
        ]

        # Step 1: write the spec JSON file onto the GPU VM
        spec_json = job.to_json()
        _ssh_run(host, user, ssh_opts,
                 f"cat > {remote_spec} << 'SPECEOF'\n{spec_json}\nSPECEOF")

        # Step 2: launch the runner detached
        _ssh_run(host, user, ssh_opts, runner_cmd)

        log.info(
            "GpuProvider.run_job: dispatched experiment_id=%d run_id=%s → %s",
            job.experiment_id, job.run_id, host,
        )

    # ── internal ─────────────────────────────────────────────────────────────

    def _state(self, instance_id: str) -> str:
        return self._compute.get_instance(instance_id).data.lifecycle_state

    def public_ip(self, instance_id: str) -> str | None:
        """Resolve the instance's current primary-VNIC **public IP**.

        The IP is ephemeral (changes across stop/start), so callers should re-resolve each time the
        VM comes up. Returns ``None`` if the VM has no public IP (e.g. STOPPED). Used by the CPU-side
        SSH health watchdog (``src/gpu_runner/health_monitor.py``).
        """
        import oci  # noqa: PLC0415
        if self._net is None:
            self._net = (oci.core.VirtualNetworkClient(self._cfg, signer=self._signer)
                         if self._signer else oci.core.VirtualNetworkClient(self._cfg))
        inst = self._compute.get_instance(instance_id).data
        for va in self._compute.list_vnic_attachments(
                compartment_id=inst.compartment_id, instance_id=instance_id).data:
            vnic = self._net.get_vnic(va.vnic_id).data
            if vnic.is_primary and vnic.public_ip:
                return vnic.public_ip
        return None


# ── module-level helpers ─────────────────────────────────────────────────────

def _short(ocid: str) -> str:
    """Last 12 chars of an OCID for log messages."""
    return ocid[-12:] if len(ocid) > 12 else ocid


def _ssh_run(host: str, user: str, opts: list[str], command: str) -> None:
    """Execute a single SSH command; raise GpuProviderError on non-zero exit."""
    result = subprocess.run(
        ["ssh", *opts, f"{user}@{host}", command],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise GpuProviderError(
            f"SSH command failed (rc={result.returncode}): {result.stderr.strip()[:300]}"
        )


def default_provider() -> GpuProvider:
    """Convenience factory using env / ~/.oci/config defaults."""
    return GpuProvider()
