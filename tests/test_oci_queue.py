"""Tests for the OCI-Queue message queue (ADR-022) — local backend. No cloud required."""

from __future__ import annotations

from pathlib import Path

from src.common.oci_queue import make_message_queue


def _q(tmp_path: Path):
    q, kind = make_message_queue(local_dir=str(tmp_path / "q"))
    assert kind == "local"
    return q


def test_put_get_delete(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.put({"run_id": "r1", "family": "xgboost"}, channel="jobs.nq-50pt")
    msgs = q.get(channel="jobs.nq-50pt", visibility_s=300, max_msgs=5)
    assert len(msgs) == 1
    assert msgs[0].body["family"] == "xgboost"
    # hidden while in-flight (visibility not expired)
    assert q.get(channel="jobs.nq-50pt", visibility_s=300) == []
    q.delete(msgs[0])
    assert q.get(channel="jobs.nq-50pt", visibility_s=0) == []


def test_visibility_timeout_redelivers(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.put({"run_id": "r1"}, channel="jobs")
    first = q.get(channel="jobs", visibility_s=0)          # expires immediately
    assert len(first) == 1
    # not acked + visibility expired → redelivered
    again = q.get(channel="jobs", visibility_s=300)
    assert len(again) == 1 and again[0].body["run_id"] == "r1"
    # now it is in-flight again
    assert q.get(channel="jobs", visibility_s=300) == []


def test_extend_keeps_message_hidden(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.put({"run_id": "r1"}, channel="jobs")
    m = q.get(channel="jobs", visibility_s=0)[0]
    q.extend(m, visibility_s=300)                          # re-hide despite the 0-timeout claim
    assert q.get(channel="jobs", visibility_s=300) == []


def test_channels_are_isolated(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.put({"n": 1}, channel="jobs.a")
    q.put({"n": 2}, channel="jobs.b")
    a = q.get(channel="jobs.a", visibility_s=300, max_msgs=5)
    assert len(a) == 1 and a[0].body["n"] == 1
    b = q.get(channel="jobs.b", visibility_s=300, max_msgs=5)
    assert len(b) == 1 and b[0].body["n"] == 2


def test_results_roundtrip(tmp_path: Path) -> None:
    q = _q(tmp_path)
    # GPU publishes a small result summary; CPU long-polls the results channel.
    q.put({"run_id": "r1", "status": "ok", "score": {"precision_at_tau": 0.42},
           "oof_ptr": "results/r1/oof_xgboost.parquet"}, channel="results.nq-50pt")
    got = q.get(channel="results.nq-50pt", visibility_s=60, max_msgs=10)
    assert len(got) == 1
    assert got[0].body["status"] == "ok"
    assert got[0].body["score"]["precision_at_tau"] == 0.42
    q.delete(got[0])
