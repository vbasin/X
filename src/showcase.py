"""Prepare and launch the portable recorded Transformer showcase.

Run from a source checkout. Runtime files belong to the explicit output root;
the packaged example and the operator's original run are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]


def _free_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"Port {port} is in use; choose a different --port or --mlflow-port") from exc


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def serve(output: Path, port: int = 8700, mlflow_port: int = 5000) -> None:
    """Own both localhost services; stop MLflow when the console exits."""
    from src.showcase_data import relocate_prepared, tracking_uri
    from src.webui.showcase import create_showcase_app

    if port == mlflow_port:
        raise ValueError("Console and MLflow need different ports")
    _free_port(port)
    _free_port(mlflow_port)
    relocate_prepared(output)
    mlflow_url = f"http://127.0.0.1:{mlflow_port}"
    app = create_showcase_app(output, mlflow_url=mlflow_url)
    log_path = output / "mlflow-server.log"
    env = os.environ.copy()
    env.update(MLFLOW_ENABLE_TELEMETRY="false", DO_NOT_TRACK="true")
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server", "--backend-store-uri",
             tracking_uri(output), "--host", "127.0.0.1", "--port", str(mlflow_port),
             "--workers", "1"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=(os.name == "posix"),
        )
        def interrupted(signum, frame):
            raise KeyboardInterrupt

        previous_handlers = {
            sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"MLflow exited. See {log_path}")
                try:
                    with urlopen(f"{mlflow_url}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"MLflow did not become ready within 60 seconds. See {log_path}")
                time.sleep(0.2)
            print(f"Showcase: http://127.0.0.1:{port}\nMLflow: {mlflow_url}\nCtrl+C stops both services.", flush=True)
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=port)
        finally:
            _stop(process)
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)


def main(argv: list[str] | None = None) -> int:
    """CLI for immutable fixture preparation and local viewing."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Verify and copy the example; import its recorded MLflow runs")
    prepare.add_argument("--example", choices=["transformer-regime"], default="transformer-regime")
    prepare.add_argument("--output", type=Path, required=True, help="New output directory (must not exist)")
    prepare.add_argument("--run-dir", type=Path, help="Export a completed fresh Transformer run instead of the bundled example")
    launch = sub.add_parser("serve", help="Launch the read-only console and local MLflow")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--port", type=int, default=8700)
    launch.add_argument("--mlflow-port", type=int, default=5000)
    verify = sub.add_parser("verify", help="Verify all retained fixture hashes")
    verify.add_argument("--example-dir", type=Path, default=REPO / "examples/transformer-regime")
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        if args.command == "prepare":
            from src.showcase_data import prepare_example
            example = REPO / "examples" / args.example
            if args.run_dir:
                from scripts.showcase.export_transformer import export_example
                # Export beside the new output; no existing output or evidence is rewritten.
                example = args.output.with_name(args.output.name + "-export").resolve()
                if args.output.exists():
                    raise FileExistsError(f"Output already exists: {args.output}")
                export_example(args.run_dir.resolve(), example)
            print("████░░░░░░░░░░░░░░░░ Verifying artifacts and importing MLflow", flush=True)
            result = prepare_example(example, args.output.resolve())
            print(json.dumps(result, indent=2))
            print(f"████████████████████ Ready. ✅ Done in {time.monotonic()-started:.1f}s", flush=True)
        elif args.command == "verify":
            from src.showcase_data import verify_example
            verify_example(args.example_dir)
            print(f"Verified {args.example_dir}")
        else:
            serve(args.output.resolve(), args.port, args.mlflow_port)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Showcase: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
