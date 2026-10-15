"""The published viewer must never depend on or mutate operational state."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.webui.showcase import create_showcase_app

pytestmark = pytest.mark.showcase
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "transformer-regime"


@pytest.fixture
def client():
    with TestClient(create_showcase_app(FIXTURE)) as instance:
        yield instance


def test_recorded_metrics_and_populations_are_real(client):
    payload = client.get("/api/showcase").json()
    assert len(payload["data"]["rows"]) == 3000
    assert len(payload["data"]["features"]) == 5
    assert set(payload["data"]["features"]).isdisjoint(payload["data"]["outcomes"])
    assert len(payload["models"]) == 2
    assert all(len(rows) == 435 for rows in payload["predictions"]["validation"].values())
    assert len(payload["predictions"]["holdout"]) == 289
    gate = payload["evaluation"]["gates"]["holdout"]
    assert len(gate["checks"]) == 8 and all(gate["checks"].values())
    assert gate["metrics"]["accuracy"] == pytest.approx(0.9342560553633218)
    assert gate["metrics"]["log_loss"] == pytest.approx(0.242029646786449)
    assert all(not row["passed"] for row in payload["evaluation"]["negative_controls"].values())
    selected = next(model for model in payload["models"] if model["selected"])
    ledger = payload["backtest"][selected["design_id"]]
    assert len(ledger["trades"]) == 70
    assert sum(row["net_dollars"] for row in ledger["trades"]) == pytest.approx(ledger["economics"]["net_dollars"])
    assert payload["viewer"]["mode"] == "recorded_example"
    assert payload["summary"]["trading_activation"] is False


def test_navigation_assets_and_security_headers(client):
    index = client.get("/")
    assert index.status_code == 200
    for tab in ("dashboard", "data", "models", "backtest", "evaluation", "artifacts", "live"):
        assert f'data-nav="{tab}"' in index.text
    for asset in ("style.css", "showcase.css", "showcase.js"):
        response = client.get("/" + asset)
        assert response.status_code == 200
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "script-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/app.js").status_code == 404  # Operational JS is never served.
    assert client.get("/api/showcase/health").json()["read_only"] is True


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_all_mutations_rejected_server_side(client, method):
    for route in ("/api/jobs", "/api/backtest/run", "/api/showcase", "/anything-new"):
        response = getattr(client, method)(route)
        assert response.status_code == 405
        assert "read-only" in response.json()["detail"]


def test_downloads_are_manifest_allowlisted(client):
    payload = client.get("/api/showcase").json()
    artifact = next(row for row in payload["artifacts"] if row["path"] == "synthetic.parquet")
    response = client.get("/api/showcase/artifact/synthetic.parquet")
    assert response.status_code == 200
    assert len(response.content) == artifact["size"]
    assert hashlib.sha256(response.content).hexdigest() == artifact["sha256"]
    for path in ("tracking.sqlite", ".env", "../README.md", "%2e%2e%2fREADME.md", "showcase_manifest.json"):
        assert client.get("/api/showcase/artifact/" + path).status_code == 404


def test_download_checks_for_changes_after_startup(tmp_path):
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    with TestClient(create_showcase_app(fixture)) as client:
        target = fixture / "synthetic.parquet"
        content = target.read_bytes()
        target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        assert client.get("/api/showcase/artifact/synthetic.parquet").status_code == 409


def test_viewer_starts_without_operational_imports_or_network():
    script = '''
import builtins, pathlib, socket, sys
original_import = builtins.__import__
blocked = ('src.webui.app', 'src.webui.datasources', 'src.webui.live',
           'src.webui.jobs', 'psycopg', 'psycopg2', 'sqlalchemy', 'oci',
           'databento', 'tensorflow', 'mlflow')
def isolated_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked):
        raise AssertionError('Forbidden operational import: ' + name)
    return original_import(name, *args, **kwargs)
def no_connect(*args, **kwargs):
    raise AssertionError('Viewer attempted network access')
builtins.__import__ = isolated_import
socket.socket.connect = no_connect
socket.create_connection = no_connect
from src.webui.showcase import create_showcase_app
from fastapi.testclient import TestClient
with TestClient(create_showcase_app(pathlib.Path(sys.argv[1]))) as client:
    assert client.get('/').status_code == 200
    assert len(client.get('/api/showcase').json()['data']['rows']) == 3000
print('isolated viewer ready')
'''
    result = subprocess.run([sys.executable, "-c", script, str(FIXTURE)], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "isolated viewer ready" in result.stdout


def test_invalid_mlflow_link_rejected():
    with pytest.raises(ValueError, match="HTTP"):
        create_showcase_app(FIXTURE, "javascript:alert(1)")


def test_prepared_relative_pointer_loads_with_local_lineage(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "example")
    mapping = {"old-parent": "new-parent"}
    (tmp_path / "showcase.json").write_text(json.dumps({
        "example_root": "example",
        "mlflow": {"run_id_map": mapping, "evidence_kind": "recorded_example", "experiment_id": "1"},
    }))
    with TestClient(create_showcase_app(tmp_path)) as client:
        payload = client.get("/api/showcase").json()
        assert payload["mlflow"]["run_id_map"] == mapping
        assert payload["viewer"]["mlflow_url"] == "http://127.0.0.1:5000/#/experiments/1/runs"
        assert client.get("/api/showcase/artifact/synthetic.parquet").status_code == 200
