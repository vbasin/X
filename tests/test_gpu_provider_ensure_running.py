"""Tests for GpuProvider.ensure_running (ADR-022) — no OCI SDK / no network.

The provider's __init__ needs ~/.oci/config, so we bypass it via __new__ and inject a fake
compute client. ensure_running itself makes no ``oci`` imports.
"""

from __future__ import annotations

from src.common.gpu_provider import GpuProvider


class _FakeData:
    def __init__(self, state: str) -> None:
        self.lifecycle_state = state


class _FakeInstance:
    def __init__(self, state: str) -> None:
        self.data = _FakeData(state)


class _FakeCompute:
    def __init__(self, state: str) -> None:
        self.state = state
        self.actions: list[str] = []

    def get_instance(self, _ocid: str) -> _FakeInstance:
        return _FakeInstance(self.state)

    def instance_action(self, _ocid: str, action: str) -> None:
        self.actions.append(action)
        if action == "START":
            self.state = "STARTING"


def _provider(state: str) -> tuple[GpuProvider, _FakeCompute]:
    p = GpuProvider.__new__(GpuProvider)      # bypass __init__ (no ~/.oci/config needed)
    fake = _FakeCompute(state)
    p._compute = fake                          # type: ignore[attr-defined]
    return p, fake


def test_starts_only_when_stopped() -> None:
    p, fake = _provider("STOPPED")
    assert p.ensure_running("ocid1.instance.x") is True
    assert fake.actions == ["START"]


def test_noop_when_running() -> None:
    p, fake = _provider("RUNNING")
    assert p.ensure_running("ocid1.instance.x") is False
    assert fake.actions == []


def test_noop_when_starting() -> None:
    p, fake = _provider("STARTING")
    assert p.ensure_running("ocid1.instance.x") is False
    assert fake.actions == []


def test_stopping_no_wait_is_noop() -> None:
    # With stopping_wait_s=0 we don't wait out a self-stop — no START issued.
    p, fake = _provider("STOPPING")
    assert p.ensure_running("ocid1.instance.x", stopping_wait_s=0) is False
    assert fake.actions == []


def test_stopping_waits_then_starts(monkeypatch) -> None:
    # A job enqueued while the runner is self-stopping: ensure_running waits for STOPPED, then
    # STARTs it, so the job is not stranded.
    import src.common.gpu_provider as gp
    monkeypatch.setattr(gp.time, "sleep", lambda *_a, **_k: None)
    p, fake = _provider("STOPPING")
    seq = ["STOPPING", "STOPPED"]

    def _get_instance(_ocid: str) -> _FakeInstance:
        if seq:
            fake.state = seq.pop(0)
        return _FakeInstance(fake.state)

    fake.get_instance = _get_instance  # type: ignore[assignment]
    assert p.ensure_running("ocid1.instance.x", stopping_wait_s=60) is True
    assert fake.actions == ["START"]


def test_stop_self_softstops(monkeypatch) -> None:
    monkeypatch.setenv("GPU_INSTANCE_OCID", "ocid1.instance.self")
    p, fake = _provider("RUNNING")
    p.stop_self()
    assert fake.actions == ["SOFTSTOP"]


def test_concurrent_callers_start_at_most_once() -> None:
    p, fake = _provider("STOPPED")
    results = [p.ensure_running("ocid1.instance.x") for _ in range(5)]
    # first flips STOPPED→STARTING; the rest are no-ops
    assert results == [True, False, False, False, False]
    assert fake.actions == ["START"]
