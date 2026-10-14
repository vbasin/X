"""No-ML smoke execution test (ADR-022).

Runs the **real** ``job_exec`` subprocess from a **real published code bundle** via the runner,
proving the bundle fetch → verify → extract → subprocess → result path end to end — with **no ML
training and no GPU**. This is the local stand-in for the cross-VM smoke test (`src.gpu_runner.smoke`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.gpu_runner.runner as runner
from src.common.code_bundle import build_and_publish
from src.gpu_runner.dispatch import (
    JOBS_CHANNEL,
    collect_result,
    enqueue_job,
    make_plane,
    results_channel,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def plane(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "plane"))
    monkeypatch.setattr(runner, "_BUNDLE_CACHE", tmp_path / "code_cache")
    queue, store = make_plane()
    digest = build_and_publish(store, _REPO_ROOT)   # bundle the real repo (src/ + config/)
    return queue, store, digest


def _smoke_job(digest: str, run_id: str = "smoke-1") -> dict:
    return {"objective_id": "smoke", "run_id": run_id, "code_digest": digest,
            "family": "__smoke__", "smoke": True, "spec_params": {},
            "parquet_dir": "data/parquet", "tune_trials": 0}


def test_execute_bundle_job_smoke_real_subprocess(plane):
    queue, store, digest = plane
    out = runner._execute_bundle_job(store, _smoke_job(digest))
    assert out["result"]["status"] == "ok"
    assert out["result"]["family"] == "__smoke__"
    assert out["ran_digest"] == digest          # ran the exact bundle we published


def test_full_smoke_plane_roundtrip(plane, tmp_path):
    queue, store, digest = plane
    enqueue_job(queue, _smoke_job(digest, run_id="smoke-2"))

    msg = queue.get(channel=JOBS_CHANNEL, visibility_s=300)[0]
    runner._run_and_publish(queue, store, msg)   # REAL subprocess execution inside

    assert queue.get(channel=JOBS_CHANNEL, visibility_s=0) == []       # job acked

    rmsg = queue.get(channel=results_channel("smoke"), visibility_s=60)[0]
    assert rmsg.body["run_id"] == "smoke-2"
    assert rmsg.body["status"] == "ok"
    # smoke produces no OOF/diagnostics; job.log may be teed (ADR-073)
    extra = [n for n in (rmsg.body.get("result_files") or [])
             if n not in ("job.log", "job_stderr.log", "progress.json")]
    assert extra == []

    oof, diag = collect_result(store, rmsg.body, tmp_path / "collected")
    assert oof is None and diag is None


def test_self_stop_when_idle_calls_stop_self(plane, monkeypatch):
    """Idle final-check with an empty queue → SOFTSTOP self, return False (exit)."""
    import types
    from concurrent.futures import ThreadPoolExecutor

    import src.common.gpu_provider as gp

    queue, store, _digest = plane
    calls: list[str] = []

    class _FakeProvider:
        def stop_self(self) -> None:
            calls.append("stop_self")

    monkeypatch.setattr(gp, "GpuProvider", _FakeProvider)
    args = types.SimpleNamespace(workers=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stayed = runner._self_stop_if_still_idle(queue, store, pool, args)
    assert stayed is False
    assert calls == ["stop_self"]


def test_self_stop_drains_final_job_and_stays_up(plane, monkeypatch):
    """A job arriving on the final idle check is drained; the VM is NOT stopped."""
    import types
    from concurrent.futures import ThreadPoolExecutor

    import src.common.gpu_provider as gp

    queue, store, digest = plane
    calls: list[str] = []

    class _FakeProvider:
        def stop_self(self) -> None:
            calls.append("stop_self")

    monkeypatch.setattr(gp, "GpuProvider", _FakeProvider)
    enqueue_job(queue, _smoke_job(digest, run_id="final-1"))
    args = types.SimpleNamespace(workers=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stayed = runner._self_stop_if_still_idle(queue, store, pool, args)
    assert stayed is True                                          # found work → stay up
    assert calls == []                                            # did NOT stop the VM
    assert queue.get(channel=JOBS_CHANNEL, visibility_s=0) == []  # final job acked
    rmsg = queue.get(channel=results_channel("smoke"), visibility_s=60)[0]
    assert rmsg.body["run_id"] == "final-1"
    assert rmsg.body["status"] == "ok"
