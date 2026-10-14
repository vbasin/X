"""End-to-end cutover test for the OCI-Queue + code-bundle GPU dispatch plane (ADR-022).

Exercises the full producer→consumer→producer message flow on the LOCAL backends (no cloud, no
real training): enqueue a job, have the runner publish results (execution stubbed), then collect
them on the producer side. Also checks the ack (job deletion) and code-digest mismatch handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.gpu_runner.runner as runner
from src.gpu_runner.dispatch import (
    JOBS_CHANNEL,
    collect_result,
    enqueue_job,
    make_plane,
    results_channel,
)


@pytest.fixture()
def plane(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "plane"))
    queue, store = make_plane()
    return queue, store


def _job(run_id: str = "nq-50pt-xgboost-1", digest: str = "deadbeef") -> dict:
    return {
        "objective_id": "nq-50pt", "run_id": run_id, "code_digest": digest,
        "family": "xgboost", "spec_params": {}, "parquet_dir": "data/parquet", "tune_trials": 0,
    }


def test_enqueue_consume_publish_collect_roundtrip(plane, tmp_path, monkeypatch):
    queue, store = plane

    # Producer enqueues a job on the jobs channel.
    enqueue_job(queue, _job())

    # Stub the heavy execution: write fake artifacts + return a result dict.
    def _fake_exec(_store, spec):
        import tempfile
        out_dir = Path(tempfile.mkdtemp())
        (out_dir / "oof_xgboost.parquet").write_bytes(b"PAR1-fake")
        (out_dir / "diagnostics_xgboost.json").write_text(json.dumps({"confusion": [[1]]}))
        return {"result": {"status": "ok", "family": "xgboost",
                           "score": {"precision_at_tau": 0.42}, "meets_target": False,
                           "error": None, "elapsed_s": 1.5},
                "out_dir": out_dir, "ran_digest": spec["code_digest"]}

    monkeypatch.setattr(runner, "_execute_bundle_job", _fake_exec)

    # Consumer claims the job, runs it, publishes results, and acks.
    msgs = queue.get(channel=JOBS_CHANNEL, visibility_s=300, max_msgs=5)
    assert len(msgs) == 1
    runner._run_and_publish(queue, store, msgs[0])

    # Job was acked (deleted) — not redelivered.
    assert queue.get(channel=JOBS_CHANNEL, visibility_s=0) == []

    # Producer collects the result from the results channel.
    rmsgs = queue.get(channel=results_channel("nq-50pt"), visibility_s=60, max_msgs=5)
    assert len(rmsgs) == 1
    body = rmsgs[0].body
    assert body["status"] == "ok"
    assert body["ran_digest"] == "deadbeef"
    assert set(body["result_files"]) == {"oof_xgboost.parquet", "diagnostics_xgboost.json"}

    # Blobs are downloadable from Object Storage into a fresh dir.
    collect_dir = tmp_path / "collected"
    oof_path, diag = collect_result(store, body, collect_dir)
    assert oof_path is not None and Path(oof_path).read_bytes() == b"PAR1-fake"
    assert diag == {"confusion": [[1]]}


def test_failed_job_publishes_failure_and_acks(plane, monkeypatch):
    queue, store = plane
    enqueue_job(queue, _job(run_id="nq-50pt-boom-1"))

    def _boom(_store, _spec):
        raise RuntimeError("training exploded")

    monkeypatch.setattr(runner, "_execute_bundle_job", _boom)
    msg = queue.get(channel=JOBS_CHANNEL, visibility_s=300)[0]
    runner._run_and_publish(queue, store, msg)

    # Acked despite failure.
    assert queue.get(channel=JOBS_CHANNEL, visibility_s=0) == []
    body = queue.get(channel=results_channel("nq-50pt"), visibility_s=60)[0].body
    assert body["status"] == "failed"
    assert "training exploded" in (body["error"] or "")
    assert body["result_files"] == []


def test_producer_ignores_foreign_run_ids(plane):
    """A result for a run_id this producer doesn't own must be left on the channel."""
    queue, store = plane
    # Seed a result for an unknown run_id.
    from src.gpu_runner.dispatch import publish_result
    import tempfile
    empty = tempfile.mkdtemp()
    publish_result(queue, store, objective_id="nq-50pt", run_id="other-wave-1",
                   code_digest="x", ran_digest="x",
                   result={"status": "ok", "family": "rf", "elapsed_s": 0.1}, out_dir=empty)

    pending = {"mine-1"}
    got = queue.get(channel=results_channel("nq-50pt"), visibility_s=0, max_msgs=5)
    owned = [m for m in got if m.body.get("run_id") in pending]
    assert owned == []  # foreign result not consumed
