"""Cross-VM smoke test (ADR-022) — proves CPU↔OCI-Queue↔GPU↔OCI-Queue↔CPU end to end **without
any ML training**.

It publishes the code bundle, enqueues a ``__smoke__`` job, optionally boots the GPU VM
(``--boot`` → ``ensure_running``), and waits for the result to come back on the results channel.
The GPU runner fetches the bundle, runs ``job_exec`` (which short-circuits on ``smoke``), and
publishes a trivial ``ok`` result — exercising the whole plane with zero training and no A10 GPU
work.

Usage (on the CPU VM, after the GPU VM is deployed + running the runner)::

    python -m src.gpu_runner.smoke --boot                 # boot the GPU + wait for round-trip
    python -m src.gpu_runner.smoke --timeout 300          # assume GPU already up
"""

from __future__ import annotations

from src.features.stores import default_parquet_dir

import argparse
import os
import time
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="ADR-022 cross-VM smoke test (no ML)")
    ap.add_argument("--objective", default="smoke", help="objective_id namespace for the results channel")
    ap.add_argument("--boot", action="store_true", help="ensure_running the GPU VM (billable A10 boot)")
    ap.add_argument("--ocid", default="", help="GPU instance OCID (default: env GPU_INSTANCE_OCID)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the round-trip")
    args = ap.parse_args()

    from src.common.code_bundle import build_and_publish
    from src.gpu_runner.dispatch import collect_result, enqueue_job, make_plane, results_channel

    queue, store = make_plane()
    print(f"plane backend: queue={queue.backend_kind}")

    digest = build_and_publish(store, _REPO_ROOT)
    oid = args.objective
    run_id = f"{oid}-smoke-{uuid.uuid4().hex[:8]}"
    enqueue_job(queue, {
        "objective_id": oid, "run_id": run_id, "code_digest": digest,
        "family": "__smoke__", "smoke": True, "spec_params": {},
        "parquet_dir": str(default_parquet_dir()), "tune_trials": 0,
    })
    print(f"enqueued smoke job run_id={run_id} digest={digest[:12]}")

    if args.boot:
        from src.common.gpu_provider import GpuProvider
        ocid = args.ocid or os.environ.get("GPU_INSTANCE_OCID", "")
        if not ocid:
            print("WARNING: --boot given but no GPU_INSTANCE_OCID — skipping boot")
        else:
            started = GpuProvider().ensure_running(ocid)
            print(f"ensure_running({ocid[-12:]}): {'STARTED' if started else 'already running/starting'}")

    rchan = results_channel(oid)
    t0 = time.time()
    print(f"waiting up to {args.timeout:.0f}s for a result on '{rchan}' …")
    while time.time() - t0 < args.timeout:
        for m in queue.get(channel=rchan, visibility_s=30, max_msgs=10, wait_s=10):
            if m.body.get("run_id") != run_id:
                continue
            dt = time.time() - t0
            body = m.body
            with __import__("tempfile").TemporaryDirectory() as tmp:
                collect_result(store, body, tmp)  # exercise the blob download path too
            queue.delete(m)
            print(f"✅ cross-VM round-trip OK in {dt:.0f}s — status={body.get('status')} "
                  f"ran_digest={str(body.get('ran_digest'))[:12]}")
            if body.get("status") != "ok":
                print(f"   ⚠ job error on GPU:\n{str(body.get('error'))[:2000]}")
            return 0
        time.sleep(2)

    print("❌ timeout — no smoke result returned. Check the GPU VM runner service + its .env "
          "(OCI_QUEUE_ID / OCI_QUEUE_MESSAGES_ENDPOINT) and that it booted.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
