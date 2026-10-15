"""Read-only, artifact-backed profile of the VTech console.

No operational WebUI module is imported here: opening a published example must
never start live workers, inspect cloud credentials, or connect to PostgreSQL.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from src.showcase_data import example_root, load_example

_STATIC = Path(__file__).resolve().parent / "static"


def create_showcase_app(
    output: Path, mlflow_url: str = "http://127.0.0.1:5000",
) -> FastAPI:
    """Serve a verified prepared example without operational dependencies."""
    parsed = urlparse(mlflow_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MLflow URL must be an HTTP(S) URL")
    output = Path(output).resolve()
    root = example_root(output)
    payload = load_example(output)
    experiment_id = payload.get("mlflow", {}).get("experiment_id")
    mlflow_link = mlflow_url.rstrip("/")
    if experiment_id is not None:
        mlflow_link += "/#/experiments/" + quote(str(experiment_id), safe="") + "/runs"
    payload["viewer"] = {
        "mode": "recorded_example", "read_only": True,
        "mlflow_url": mlflow_link,
        "unsupported": ["live_feeds", "order_book", "cloud_jobs", "trading"],
    }
    artifacts = {row["path"]: row for row in payload["artifacts"]}
    app = FastAPI(title="VTech Transformer showcase", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def read_only(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            return JSONResponse(
                {"detail": "This recorded showcase is read-only. Use the training CLI for a fresh run."},
                status_code=405,
                headers={"Allow": "GET, HEAD, OPTIONS"},
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    async def index():
        return FileResponse(_STATIC / "showcase.html")

    @app.get("/api/showcase/health")
    async def health():
        return {"status": "ready", "mode": "recorded_example", "read_only": True}

    @app.get("/api/showcase")
    async def example():
        return payload

    @app.get("/api/showcase/artifact/{artifact_path:path}")
    async def artifact(artifact_path: str):
        row = artifacts.get(artifact_path)
        if row is None:
            raise HTTPException(status_code=404, detail="Artifact is not part of this example")
        path = (root / artifact_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact unavailable")
        if (path.stat().st_size != row["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise HTTPException(status_code=409, detail="Artifact integrity check failed")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.get("/{asset}")
    async def static_asset(asset: str):
        if asset not in {"showcase.js", "showcase.css", "style.css"}:
            raise HTTPException(status_code=404)
        return FileResponse(_STATIC / asset)

    return app
