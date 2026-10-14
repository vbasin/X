"""CPU→GPU troubleshooting channel via OCI Run Command (Oracle Cloud Agent) — no SSH / no network.

Drives the compute-instance-agent **Run Command** feature over the same OCI control plane used for
START/STOP (auth via ``~/.oci/config``), so we can inspect processes, logs, the codebase, and job
state on the GPU VM without any inbound network path, bastion, or public IP.

Prereqs:
  * GPU instance RUNNING with the **Run Command** Oracle Cloud Agent plugin enabled
    (Console → the instance → Oracle Cloud Agent → enable "Compute Instance Run Command").
  * The CPU user's IAM group allowed to ``use instance-agent-command-family`` in the GPU's
    compartment (and ``read instances``).

CLI::

    python -m src.common.gpu_exec "cd /opt/vtech && git log --oneline -3"
    python -m src.common.gpu_exec --ensure-running "journalctl -u vtech-gpu-runner -n 40 --no-pager"

Python::

    from src.common.gpu_exec import run_on_gpu
    code, out = run_on_gpu("uptime")
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _config():
    import oci  # noqa: PLC0415
    cfg = oci.config.from_file(
        os.environ.get("OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config")))
    region = os.environ.get("GPU_REGION")
    if region:
        cfg = {**cfg, "region": region}
    return cfg


def run_on_gpu(
    command: str,
    *,
    timeout: int = 300,
    ensure_running: bool = False,
    poll_s: float = 4.0,
) -> tuple[int, str]:
    """Run *command* on the GPU VM and return ``(exit_code, combined_output)``.

    Blocks until the command finishes (or *timeout*). If the instance is not RUNNING and
    ``ensure_running`` is set, it is started first (the agent only receives commands while up).
    """
    import oci  # noqa: PLC0415

    cfg = _config()
    ocid = os.environ["GPU_INSTANCE_OCID"]
    compute = oci.core.ComputeClient(cfg)
    inst = compute.get_instance(ocid).data

    if inst.lifecycle_state != "RUNNING":
        if not ensure_running:
            return 2, f"GPU is {inst.lifecycle_state}; pass ensure_running=True to start it first"
        if inst.lifecycle_state == "STOPPED":
            compute.instance_action(ocid, "START")
        deadline = time.time() + 300
        while time.time() < deadline:
            if compute.get_instance(ocid).data.lifecycle_state == "RUNNING":
                break
            time.sleep(6)
        time.sleep(25)   # let the Oracle Cloud Agent come up after boot

    agent = oci.compute_instance_agent.ComputeInstanceAgentClient(cfg)
    m = oci.compute_instance_agent.models
    details = m.CreateInstanceAgentCommandDetails(
        compartment_id=inst.compartment_id,
        execution_time_out_in_seconds=int(timeout),
        display_name="vtech-gpu-exec",
        target=m.InstanceAgentCommandTarget(instance_id=ocid),
        content=m.InstanceAgentCommandContent(
            source=m.InstanceAgentCommandSourceViaTextDetails(source_type="TEXT", text=command),
            output=m.InstanceAgentCommandOutputViaTextDetails(output_type="TEXT")),
    )
    cmd = agent.create_instance_agent_command(create_instance_agent_command_details=details).data

    deadline = time.time() + int(timeout) + 90
    while time.time() < deadline:
        ex = agent.get_instance_agent_command_execution(
            instance_agent_command_id=cmd.id, instance_id=ocid).data
        if ex.lifecycle_state in ("SUCCEEDED", "FAILED", "TIMED_OUT"):
            c = ex.content
            out = getattr(c, "text", None)
            if out is None:
                out = getattr(c, "message", None) or str(c)
            return int(getattr(c, "exit_code", 1) or 0), out
        time.sleep(poll_s)
    return 124, "timed out waiting for command execution"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a shell command on the GPU VM via OCI Run Command")
    ap.add_argument("command", help="shell command to run on the GPU")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--ensure-running", action="store_true",
                    help="start the GPU VM first if it is STOPPED")
    args = ap.parse_args()
    code, out = run_on_gpu(args.command, timeout=args.timeout, ensure_running=args.ensure_running)
    print(out)
    print(f"[gpu exit={code}]", file=sys.stderr)
    raise SystemExit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
