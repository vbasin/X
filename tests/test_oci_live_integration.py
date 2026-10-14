"""Live OCI integration tests (ADR-022) — exercise the REAL OCI Queue + Object Storage plane.

Guarded so normal ``pytest`` runs never touch the cloud. Enable explicitly:

    set -a && source /opt/vtech/.env && set +a
    VTECH_LIVE_OCI_TESTS=1 python -m pytest tests/test_oci_live_integration.py -v

Requires the activation-switch env (``OCI_QUEUE_ID`` + ``OCI_QUEUE_MESSAGES_ENDPOINT`` +
``AWS_*`` / ``MLFLOW_S3_ENDPOINT_URL``) and NO ``VTECH_GPU_QUEUE_LOCAL`` override. These tests
never start the GPU VM (no ``ensure_running`` / no A10 boot) and clean up every object/message
they create.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

_LIVE = (
    os.environ.get("VTECH_LIVE_OCI_TESTS") == "1"
    and bool(os.environ.get("OCI_QUEUE_ID"))
    and bool(os.environ.get("OCI_QUEUE_MESSAGES_ENDPOINT"))
    and not os.environ.get("VTECH_GPU_QUEUE_LOCAL")
)
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live OCI tests disabled — set VTECH_LIVE_OCI_TESTS=1 and source .env to enable",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_live_queue_put_get_delete():
    """Real OCI Queue put → long-poll get → delete; not redelivered after ack."""
    from src.common.oci_queue import make_message_queue

    q, kind = make_message_queue()
    assert kind == "oci"
    chan = f"livetest.{uuid.uuid4().hex[:8]}"
    token = uuid.uuid4().hex

    q.put({"ping": token}, channel=chan)
    got = q.get(channel=chan, visibility_s=15, max_msgs=5, wait_s=10)
    mine = [m for m in got if m.body.get("ping") == token]
    assert len(mine) == 1
    q.delete(mine[0])

    leftover = [m for m in q.get(channel=chan, visibility_s=1, wait_s=3)
                if m.body.get("ping") == token]
    assert leftover == []


def test_live_object_store_bundle_roundtrip():
    """Build → publish (real vtech_hot) → fetch+verify+extract → cleanup."""
    from src.common.code_bundle import build_bundle, bundle_key, fetch_and_extract, publish_bundle
    from src.common.object_store import make_object_store

    store, kind = make_object_store()
    assert kind == "s3"
    digest, comp = build_bundle(_REPO_ROOT)
    key = publish_bundle(store, digest, comp)
    try:
        assert store.exists(key)
        with tempfile.TemporaryDirectory() as tmp:
            dest = fetch_and_extract(store, digest, tmp)
            assert (dest / "src" / "common" / "code_bundle.py").exists()
    finally:
        store.delete(bundle_key(digest))


def test_live_full_plane_roundtrip(monkeypatch):
    """Producer→(real queue+store)→consumer(stub exec)→results→collect. No GPU boot."""
    import src.gpu_runner.runner as runner
    from src.gpu_runner.dispatch import (
        JOBS_CHANNEL,
        collect_result,
        enqueue_job,
        make_plane,
        publish_result,
        results_channel,
    )

    queue, store = make_plane()
    oid = f"livetest-{uuid.uuid4().hex[:8]}"
    run_id = f"{oid}-xgboost-1"

    # ── producer: enqueue a job on the real jobs channel ──
    enqueue_job(queue, {
        "objective_id": oid, "run_id": run_id, "code_digest": "live",
        "family": "xgboost", "spec_params": {}, "parquet_dir": "data/parquet", "tune_trials": 0,
    })

    # ── consumer: claim ONLY our job (leave any foreign jobs untouched) ──
    claimed = None
    for _ in range(3):
        for m in queue.get(channel=JOBS_CHANNEL, visibility_s=60, max_msgs=10, wait_s=10):
            if m.body.get("run_id") == run_id:
                claimed = m
        if claimed:
            break
    assert claimed is not None, "enqueued job was not delivered back on the jobs channel"

    # stub execution — no real training, no subprocess, no A10
    out_dir = Path(tempfile.mkdtemp())
    (out_dir / "oof_xgboost.parquet").write_bytes(b"PAR1-live")
    (out_dir / "diagnostics_xgboost.json").write_text('{"ok": true}')
    result = {"status": "ok", "family": "xgboost", "score": {"precision_at_tau": 0.5},
              "meets_target": False, "error": None, "elapsed_s": 0.2}

    publish_result(queue, store, objective_id=oid, run_id=run_id, code_digest="live",
                   ran_digest="live", result=result, out_dir=out_dir)
    queue.delete(claimed)   # ack the job

    # ── producer: collect from the real results channel + download blobs ──
    result_msg = None
    for _ in range(3):
        for m in queue.get(channel=results_channel(oid), visibility_s=30, max_msgs=5, wait_s=10):
            if m.body.get("run_id") == run_id:
                result_msg = m
        if result_msg:
            break
    assert result_msg is not None, "result message was not delivered"
    body = result_msg.body
    assert body["status"] == "ok"
    assert set(body["result_files"]) == {"oof_xgboost.parquet", "diagnostics_xgboost.json"}

    with tempfile.TemporaryDirectory() as tmp:
        oof_path, diag = collect_result(store, body, tmp)
        assert oof_path and Path(oof_path).read_bytes() == b"PAR1-live"
        assert diag == {"ok": True}

    # ── cleanup: ack result + delete result blobs ──
    queue.delete(result_msg)
    for name in body.get("result_files", []):
        store.delete(f"results/{run_id}/{name}")


def test_live_smoke_full_execution(monkeypatch, tmp_path):
    """The closest thing to a cross-VM run without the physical GPU: enqueue a __smoke__ job on the
    REAL queue, publish + fetch the bundle from REAL Object Storage, run the REAL job_exec
    subprocess (no ML), and collect the result over the REAL queue. Only the A10 boot is missing.
    """
    import src.gpu_runner.runner as runner
    from src.common.code_bundle import bundle_key
    from src.gpu_runner.dispatch import (
        JOBS_CHANNEL, enqueue_job, make_plane, results_channel,
    )
    from src.common.code_bundle import build_and_publish

    monkeypatch.setattr(runner, "_BUNDLE_CACHE", tmp_path / "code_cache")
    queue, store = make_plane()
    digest = build_and_publish(store, _REPO_ROOT)          # real upload to vtech_hot
    oid = f"livesmoke-{uuid.uuid4().hex[:8]}"
    run_id = f"{oid}-smoke-1"
    enqueue_job(queue, {"objective_id": oid, "run_id": run_id, "code_digest": digest,
                        "family": "__smoke__", "smoke": True, "spec_params": {},
                        "parquet_dir": "data/parquet", "tune_trials": 0})
    try:
        claimed = None
        for _ in range(3):
            for m in queue.get(channel=JOBS_CHANNEL, visibility_s=120, max_msgs=10, wait_s=10):
                if m.body.get("run_id") == run_id:
                    claimed = m
            if claimed:
                break
        assert claimed is not None, "smoke job not delivered on jobs channel"

        runner._run_and_publish(queue, store, claimed)     # REAL bundle fetch + subprocess + publish

        result = None
        for _ in range(3):
            for m in queue.get(channel=results_channel(oid), visibility_s=30, max_msgs=5, wait_s=10):
                if m.body.get("run_id") == run_id:
                    result = m
            if result:
                break
        assert result is not None and result.body["status"] == "ok"
        assert result.body["ran_digest"] == digest
        queue.delete(result)
    finally:
        store.delete(bundle_key(digest))
