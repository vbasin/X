"""Bounded HTTP deployment orchestration tests without fitting a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import urllib.error

import pandas as pd
import pytest

from src.serving import workflow_http


pytestmark = pytest.mark.ml_workflow


class FakeProcess:
    pid = 987654321
    returncode = None

    def __init__(self, *, needs_kill: bool = False):
        self.terminated = False
        self.killed = False
        self.needs_kill = needs_kill

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.needs_kill and not self.killed:
            raise subprocess.TimeoutExpired("fake-server", timeout)
        self.returncode = -9 if self.killed else -15
        return self.returncode


@pytest.fixture
def fixture_model(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "MLmodel").write_text("flavors: {}\n")
    inputs = pd.DataFrame({"ts": ["2026-01-01T00:00:00+00:00"], "signal": [0.5]})
    expected = pd.DataFrame({
        "ts": inputs.ts, "direction": ["up"], "confidence": [0.8],
        "p_down": [0.1], "p_flat": [0.1], "p_up": [0.8],
    })
    return model, inputs, expected, tmp_path / "evidence"


def install_fake(monkeypatch, expected, *, invalid_status=400, needs_kill=False):
    process = FakeProcess(needs_kill=needs_kill)
    launch = {}

    def popen(command, **kwargs):
        launch.update(command=command, **kwargs)
        kwargs["stdout"].write("retained server startup log\n")
        kwargs["stdout"].flush()
        return process

    def request(url, *, payload=None, timeout):
        assert 0 < timeout <= 10
        assert url.startswith("http://127.0.0.1:")
        if url.endswith("/health"):
            return 200, b"ok"
        columns = payload["dataframe_split"]["columns"]
        if "signal" not in columns:
            return invalid_status, b'{"error_code":"BAD_REQUEST"}'
        return 200, json.dumps({"predictions": json.loads(expected.to_json(orient="records", date_format="iso"))}).encode()

    monkeypatch.setattr(workflow_http.subprocess, "Popen", popen)
    monkeypatch.setattr(workflow_http, "_request", request)
    return process, launch


def test_http_contract_parity_schema_rejection_and_cleanup(monkeypatch, fixture_model):
    model, inputs, expected, evidence = fixture_model
    process, launch = install_fake(monkeypatch, expected)
    result = workflow_http.verify_http_deployment(model, inputs, expected, evidence)
    assert result["verified"] and result["server_stopped"]
    assert result["invalid_feature_status"] == 400
    assert process.terminated and not process.killed
    command = launch["command"]
    assert command[:5] == [workflow_http.sys.executable, "-m", "mlflow", "models", "serve"]
    assert command[command.index("--env-manager") + 1] == "local"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert launch["start_new_session"] is False
    assert launch["env"]["PATH"].split(os.pathsep)[0] == str(Path(workflow_http.sys.executable).parent)
    assert (evidence / "http_server.log").read_text() == "retained server startup log\n"


def test_readiness_deadline_stops_child_and_retains_log(monkeypatch, fixture_model):
    model, inputs, expected, evidence = fixture_model
    process, _ = install_fake(monkeypatch, expected)
    clock = [0.0]
    monkeypatch.setattr(workflow_http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(workflow_http.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("not ready")

    monkeypatch.setattr(workflow_http, "_request", unavailable)
    with pytest.raises(TimeoutError, match="readiness timed out"):
        workflow_http.verify_http_deployment(model, inputs, expected, evidence, timeout_seconds=0.25)
    assert clock[0] <= 0.35
    assert process.terminated
    assert (evidence / "http_server.log").exists()


def test_probability_mismatch_still_terminates_child(monkeypatch, fixture_model):
    model, inputs, expected, evidence = fixture_model
    wrong = expected.copy()
    wrong["p_up"] = 0.6
    process, _ = install_fake(monkeypatch, wrong)
    with pytest.raises(AssertionError):
        workflow_http.verify_http_deployment(model, inputs, expected, evidence)
    assert process.terminated


def test_schema_acceptance_is_a_failure_and_child_is_killed_if_needed(monkeypatch, fixture_model):
    model, inputs, expected, evidence = fixture_model
    process, _ = install_fake(monkeypatch, expected, invalid_status=200, needs_kill=True)
    with pytest.raises(RuntimeError, match="schema validation expected"):
        workflow_http.verify_http_deployment(model, inputs, expected, evidence)
    assert process.terminated and process.killed


def test_early_exit_is_not_reported_as_ready(monkeypatch, fixture_model):
    model, inputs, expected, evidence = fixture_model
    process, _ = install_fake(monkeypatch, expected)
    process.returncode = 3
    with pytest.raises(RuntimeError, match="server exited with 3"):
        workflow_http.verify_http_deployment(model, inputs, expected, evidence)
    assert not process.terminated


def test_nonlocal_model_and_invalid_expected_are_rejected_before_launch(monkeypatch, fixture_model):
    model, inputs, expected, evidence = fixture_model

    def unexpected_launch(*args, **kwargs):
        pytest.fail("invalid input must not launch a server")

    monkeypatch.setattr(workflow_http.subprocess, "Popen", unexpected_launch)
    with pytest.raises(ValueError, match="local MLflow"):
        workflow_http.verify_http_deployment("https://external.example/model", inputs, expected, evidence)
    with pytest.raises(ValueError, match="aligned canonical"):
        workflow_http.verify_http_deployment(model, inputs, expected.drop(columns="direction"), evidence)
    with pytest.raises(ValueError, match="loopback"):
        workflow_http._request("https://external.example/health", timeout=1)
