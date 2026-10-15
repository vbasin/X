"""Bounded loopback REST acceptance check for an exported MLflow workflow model."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


def _request(url: str, *, payload: dict | None = None, timeout: float) -> tuple[int, bytes]:
    """Loopback-only request, explicitly bypassing environment HTTP proxies."""
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("deployment verification only permits loopback URLs")
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"} if body else {},
        method="GET" if body is None else "POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(4 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(4 * 1024 * 1024)


def _stop_process(process: subprocess.Popen) -> None:
    """Stop only our child; its group belongs to the outer workflow supervisor.

    MLflow's Linux local-serving backend gives its server child a SIGTERM
    parent-death signal. The CLI deliberately inherits the supervisor's process
    group, so the outer hard deadline can also kill every descendant together.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def verify_http_deployment(
    model_path: str | Path,
    input_example: Any,
    expected: Any,
    output_dir: str | Path,
    timeout_seconds: float = 45,
) -> dict[str, Any]:
    """Serve a local model, compare REST inference, and require schema rejection.

    ``input_example`` and ``expected`` are DataFrames from local pyfunc inference.
    The server is temporary, binds only 127.0.0.1, and uses the current Python
    environment without dependency installation. Its complete output is retained
    as ``http_server.log`` even when readiness or inference fails. The outer
    workflow supervisor remains responsible for the total process-group deadline.
    """
    import numpy as np
    import pandas as pd

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    model = Path(model_path).resolve()
    if not model.is_dir() or not (model / "MLmodel").is_file():
        raise ValueError("HTTP deployment requires an existing local MLflow model directory")
    if not isinstance(input_example, pd.DataFrame) or not isinstance(expected, pd.DataFrame):
        raise ValueError("HTTP deployment requires input and expected DataFrames")
    feature_columns = [column for column in input_example.columns if column != "ts"]
    probability_columns = ["p_down", "p_flat", "p_up"]
    if not feature_columns or len(input_example) == 0:
        raise ValueError("HTTP deployment needs a nonempty input with feature columns")
    required = [*probability_columns, "direction", "confidence", "ts"]
    if any(column not in expected for column in required) or len(expected) != len(input_example):
        raise ValueError("expected inference must contain aligned canonical predictions")
    if not np.isfinite(expected[probability_columns + ["confidence"]].to_numpy(dtype=float)).all():
        raise ValueError("expected inference must be finite")
    # Bind-to-zero reserves an available local port until just before launch.
    # If another process wins the small release/bind race, readiness fails rather
    # than interacting with a live platform service at a known fixed port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "http_server.log"
    command = [sys.executable, "-m", "mlflow", "models", "serve", "--model-uri", str(model),
               "--env-manager", "local", "--host", "127.0.0.1", "--port", str(port)]
    # MLflow starts its serving executable (uvicorn in 3.x) by name. Selecting
    # the venv's Python alone does not put sibling entry points on PATH when
    # this CLI was launched without shell activation.
    executable_directory = str(Path(sys.executable).parent)
    environment = {
        **os.environ, "DO_NOT_TRACK": "true", "MLFLOW_ENABLE_TELEMETRY": "false",
        "PATH": executable_directory + os.pathsep + os.environ.get("PATH", os.defpath),
    }
    started = time.monotonic()
    deadline = started + timeout_seconds
    endpoint = f"http://127.0.0.1:{port}"
    process = None
    with log_path.open("x") as log:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                start_new_session=False,
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"MLflow HTTP readiness timed out; inspect {log_path}")
                if process.poll() is not None:
                    raise RuntimeError(f"MLflow HTTP server exited with {process.returncode}; inspect {log_path}")
                try:
                    status, _ = _request(endpoint + "/health", timeout=min(0.5, remaining))
                    if status == 200:
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            ready_seconds = time.monotonic() - started

            def request_timeout() -> float:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("MLflow HTTP deployment verification exceeded its deadline")
                return min(10.0, remaining)

            payload = {"dataframe_split": json.loads(input_example.to_json(orient="split", date_format="iso"))}
            inference_started = time.monotonic()
            status, response = _request(endpoint + "/invocations", payload=payload, timeout=request_timeout())
            inference_seconds = time.monotonic() - inference_started
            if status != 200:
                raise RuntimeError(f"MLflow HTTP inference failed with status {status}: {response[:500]!r}")
            parsed = json.loads(response)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("predictions"), list):
                raise RuntimeError("MLflow HTTP inference did not return prediction records")
            actual = pd.DataFrame(parsed["predictions"])
            if len(actual) != len(expected) or any(column not in actual for column in required):
                raise RuntimeError("MLflow HTTP output violates the canonical prediction shape")
            np.testing.assert_allclose(
                actual[probability_columns + ["confidence"]].to_numpy(dtype=float),
                expected[probability_columns + ["confidence"]].to_numpy(dtype=float),
                rtol=0, atol=1e-6,
            )
            np.testing.assert_array_equal(actual.direction.to_numpy(), expected.direction.to_numpy())
            np.testing.assert_array_equal(
                pd.to_datetime(actual.ts, utc=True).array.asi8,
                pd.to_datetime(expected.ts, utc=True).array.asi8,
            )
            invalid = input_example.drop(columns=[feature_columns[0]])
            invalid_payload = {"dataframe_split": json.loads(invalid.to_json(orient="split", date_format="iso"))}
            invalid_status, _ = _request(endpoint + "/invocations", payload=invalid_payload, timeout=request_timeout())
            if invalid_status not in (400, 422):
                raise RuntimeError(f"MLflow HTTP schema validation expected 400/422, got {invalid_status}")
            if process.poll() is not None:
                raise RuntimeError("MLflow HTTP server exited during inference verification")
            result = {
                "verified": True, "endpoint": endpoint, "input_rows": len(input_example),
                "startup_seconds": ready_seconds, "inference_seconds": inference_seconds,
                "probability_parity": True, "direction_parity": True, "timestamp_parity": True,
                "invalid_feature_status": invalid_status, "log_path": str(log_path),
                "server_pid": process.pid, "local_only": True,
            }
        finally:
            if process is not None:
                _stop_process(process)
    result.update(server_stopped=True, elapsed_seconds=time.monotonic() - started)
    return result
