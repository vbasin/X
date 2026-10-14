"""Web console (src/webui) — API surface, job manager, and security guards.

Uses FastAPI's TestClient (httpx) with a JobManager rooted in tmp_path so no real
workflow subprocesses are launched (jobs run /bin/echo etc.). Filesystem fixtures
mimic the real scoping/search artifact layouts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.webui import datasources as ds
from src.webui.app import create_app
from src.webui.jobs import JobManager

REPO = Path(__file__).resolve().parents[1]
H = {"X-VTech-Console": "1"}                    # mutating-request header


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """App wired to a tmp JobManager + tmp scoping/search roots (real objectives dir)."""
    monkeypatch.setattr(ds, "SCOPING_ROOT", tmp_path / "scoping")
    monkeypatch.setattr(ds, "SEARCH_ROOT", tmp_path / "search")
    monkeypatch.setattr(
        "src.search.custom_test.saved_views_dir",
        lambda: tmp_path / "custom_feature_views",
    )
    jm = JobManager(root=tmp_path / "jobs")
    with TestClient(create_app(jobs=jm, enable_live_worker=False), base_url="http://127.0.0.1") as c:
        yield c


@pytest.fixture(autouse=True)
def isolate_live_snapshot_cache(tmp_path, monkeypatch):
    from src.webui import live as live_mod

    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_PATH", tmp_path / "webui" / "live_snapshot_cache.json")
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_META", {"mtime_ns": None, "version": None})
    monkeypatch.setattr(live_mod, "_STATE", {"status": "starting", "prediction": None})
    monkeypatch.setattr(live_mod, "_BUNDLE", None)
    monkeypatch.setattr(live_mod, "_TICKS_CACHE", {})
    monkeypatch.setattr(live_mod, "_TICKS_JOB", {"key": None, "gen": 0})
    monkeypatch.setattr(live_mod, "_SIG_STATE", {"last_ts": None, "path": None})
    monkeypatch.setattr(live_mod, "_STREAMER_HEALTH", {
        "status": "warming",
        "tables": {"tbbo": None, "bbo_1s": None, "statistics": None},
        "tbbo_rows_10s": 0,
        "tbbo_age_s": None,
        "live": False,
        "error": "warming: health probe has not run yet",
    })


def _mk_scoping_run(root: Path, obj: str, ts: str, done: int, failed: bool = False) -> Path:
    d = root / obj / ts
    d.mkdir(parents=True)
    stages = {}
    order = list(ds.SCOPING_STAGES)
    for i, s in enumerate(order[:done]):
        stages[s] = {"status": "done", "at": "2026-07-14T00:00:00+00:00", "elapsed_s": 1.0}
    if failed and done < len(order):
        stages[order[done]] = {"status": "failed", "error": "boom"}
    (d / "state.json").write_text(json.dumps(
        {"objective_id": obj, "run_ts": ts, "dry_run": True,
         "stages": stages, "elapsed_s": float(done)}))
    (d / "final_research_report.md").write_text("# Report\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    (d / "scoping_plan.json").write_text('{"model_plan": []}')
    return d


def _mk_search_run(root: Path, obj: str, ts: str) -> Path:
    d = root / obj / ts
    d.mkdir(parents=True)
    (d / "search_result.json").write_text(json.dumps({
        "objective_id": obj, "run_ts": ts, "n_candidates": 2, "n_ok": 2, "n_failed": 0,
        "champion": None,
        "gate_result": {"status": "NO_CHAMPION", "champion_id": None,
                        "champion_precision": float("nan"), "champion_meets_target": False,
                        "passed_ids": [], "target_ids": [], "failed_ids": [], "reasons": []},
        "candidates": [
            {"candidate_id": "xgb", "family": "xgboost", "spec_params": {}, "status": "ok",
             "score": {"precision_at_tau": 0.62, "directional_f1": 0.21, "n_signals": 100},
             "elapsed_s": 12.0, "meets_target": False, "diagnostics": {"big": "blob"}},
            {"candidate_id": "lstm", "family": "attention_lstm", "spec_params": {}, "status": "ok",
             "score": {"precision_at_tau": float("nan"), "directional_f1": 0.0, "n_signals": 5},
             "elapsed_s": 90.0, "meets_target": False},
        ]}, allow_nan=True))
    return d


def _mk_oof(root: Path, obj: str, ts: str, *, model: str = "xgb",
            start: str = "2026-05-01 13:30", rows: int = 6) -> Path:
    import pandas as pd
    d = root / obj / ts
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(start, periods=rows, freq="5s", tz="UTC")
    half = rows // 2
    pd.DataFrame({
        "ts": idx,
        "direction":      (["up", "flat"] * half)[:rows],
        "confidence":     ([0.95, 0.10] * half)[:rows],
        "direction_true": (["up", "flat"] * half)[:rows],
    }).to_parquet(d / f"oof_{model}.parquet", index=False)
    return d / f"oof_{model}.parquet"


def _mk_features(pdir: Path, date: str, *, start: str, rows: int, base: float = 100.0,
                 step: float = 2.0) -> None:
    import numpy as np
    import pandas as pd
    pdir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(start, periods=rows, freq="5s", tz="UTC")
    pd.DataFrame({"_close": base + step * np.arange(rows, dtype="float64")},
                 index=idx).to_parquet(pdir / f"features_{date}.parquet")


# ── objectives ────────────────────────────────────────────────────────────────

def test_objectives_list_and_detail(client):
    objs = client.get("/api/objectives").json()
    ids = {o["objective_id"] for o in objs}
    assert {"nq-50pt", "nq-50pt-tbm"} <= ids
    tbm = client.get("/api/objectives/nq-50pt-tbm").json()
    assert tbm["parsed"]["labels"]["scheme"] == "triple_barrier"
    assert "yaml" in tbm
    opt = next(o for o in objs if o["objective_id"] == "nq-50pt-tbm-opt")
    assert "error" not in opt
    assert opt["symbol"] == "NQ"
    assert opt["scheme"] == "triple_barrier"
    assert opt["move_threshold"] == 50
    assert opt["mae_tolerance"] == 6
    assert opt["kpi_primary"] == "directional_f1"
    detail = client.get("/api/objectives/nq-50pt-tbm-opt")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["parse_error"] is None
    assert body["parsed"]["labels"]["scheme"] == "triple_barrier"
    assert body["parsed"]["traded products"][0]["symbol"] == "NQ"


def test_objective_detail_bad_yaml_is_not_500(client, tmp_path, monkeypatch):
    """A ScannerError in one objective YAML must not 500 the detail page."""
    monkeypatch.setattr(ds, "OBJECTIVES_DIR", tmp_path)
    (tmp_path / "broken-opt.yaml").write_text(
        "objective_id: broken-opt\ncustom_optimization: false\n  labels: [down, flat, up]\n"
    )
    listed = client.get("/api/objectives").json()
    row = next(o for o in listed if o["objective_id"] == "broken-opt")
    assert "error" in row
    r = client.get("/api/objectives/broken-opt")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parse_error"]
    assert "mapping values" in body["parse_error"]
    assert body["parsed"] is None
    assert "custom_optimization: false" in body["yaml"]


def test_objective_schema_served(client):
    schema = client.get("/api/objectives/_schema").json()
    assert schema["title"] == "TradeObjective"


def test_objective_validate_bad_and_good(client):
    bad = client.post("/api/objectives/validate", headers=H,
                      json={"objective": {"objective_id": "x"}}).json()
    assert not bad["valid"] and bad["errors"]
    good_src = client.get("/api/objectives/nq-50pt-tbm").json()["parsed"]
    good = client.post("/api/objectives/validate", headers=H,
                       json={"objective": good_src}).json()
    assert good["valid"], good["errors"]


def test_objective_create_roundtrip(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "OBJECTIVES_DIR", tmp_path / "objectives")
    ds.OBJECTIVES_DIR.mkdir()
    (ds.OBJECTIVES_DIR / "objective.schema.json").write_text(
        (REPO / "config/objectives/objective.schema.json").read_text())
    obj = {"objective_id": "test-10pt", "description": "Test objective for the console.",
           "traded products": [{"symbol": "NQ", "role": "primary"}],
           "labels": {"move_threshold": 10, "mae_tolerance": 3, "unit": "points",
                      "scheme": "detection_2"},
           "kpi": {"primary": "precision_at_tau",
                   "requirement": {"any": [{"metric": "precision_at_tau", "tau": 0.8, "min": 0.8}]},
                   "target": {"all": [{"metric": "precision_at_tau", "tau": 0.9, "min": 0.9}]}}}
    r = client.post("/api/objectives", headers=H, json={"objective": obj})
    assert r.status_code == 201, r.text
    assert (ds.OBJECTIVES_DIR / "test-10pt.yaml").exists()
    # overwrite refused
    assert client.post("/api/objectives", headers=H, json={"objective": obj}).status_code == 409
    # invalid id refused
    obj2 = {**obj, "objective_id": "Bad/../Id"}
    assert client.post("/api/objectives", headers=H, json={"objective": obj2}).status_code == 400


# ── scoping runs ──────────────────────────────────────────────────────────────

def test_scoping_runs_listing_and_detail(client, tmp_path):
    _mk_scoping_run(ds.SCOPING_ROOT, "nq-50pt-tbm", "20260714T000000.000000Z", done=5)
    _mk_scoping_run(ds.SCOPING_ROOT, "nq-50pt-tbm", "20260714T010000.000000Z", done=2, failed=True)
    runs = client.get("/api/scoping/runs").json()
    assert [r["status"] for r in runs] == ["failed", "complete"]
    det = client.get("/api/scoping/runs/nq-50pt-tbm/20260714T000000.000000Z").json()
    assert det["stages_done"] == 5 and det["status"] == "complete"
    assert any(a["name"] == "final_research_report.md" for a in det["artifacts"])


def test_list_limit_reads_only_newest(client):
    for i in range(5):
        _mk_scoping_run(ds.SCOPING_ROOT, "nq-50pt", f"2026071{i}T000000.000000Z", done=5)
    all_runs = ds.list_scoping_runs()
    assert len(all_runs) == 5 and all_runs[0]["run_ts"] == "20260714T000000.000000Z"
    top = ds.list_scoping_runs(limit=2)
    assert [r["run_ts"] for r in top] == [r["run_ts"] for r in all_runs[:2]]  # newest-2, same order
    assert ds.list_scoping_runs(limit=0) == []                                 # limit honoured at 0


def test_scoping_artifact_read_and_traversal_block(client):
    _mk_scoping_run(ds.SCOPING_ROOT, "nq-50pt-tbm", "20260714T000000.000000Z", done=5)
    art = client.get("/api/scoping/runs/nq-50pt-tbm/20260714T000000.000000Z/"
                     "artifact/final_research_report.md").json()
    assert art["content"].startswith("# Report")
    for evil in ("..%2F..%2Fetc%2Fpasswd", "state..json/../../x"):
        r = client.get(f"/api/scoping/runs/nq-50pt-tbm/20260714T000000.000000Z/artifact/{evil}")
        assert r.status_code in (400, 404), evil
    # bad objective id + bad ts rejected (400 from validation, or 404 route mismatch)
    assert client.get("/api/scoping/runs/NQ%2F..%2F/20260714T000000.000000Z").status_code in (400, 404)
    assert client.get("/api/scoping/runs/nq-50pt-tbm/not-a-ts").status_code == 400


# ── search runs ───────────────────────────────────────────────────────────────

def test_search_runs_and_nan_sanitized(client):
    _mk_search_run(ds.SEARCH_ROOT, "nq-50pt-tbm", "20260714T000000Z")
    runs = client.get("/api/search/runs").json()
    assert runs[0]["gate_status"] == "NO_CHAMPION"
    assert runs[0]["champion_precision"] is None          # NaN → null
    det = client.get("/api/search/runs/nq-50pt-tbm/20260714T000000Z").json()
    cands = {c["candidate_id"]: c for c in det["candidates"]}
    assert cands["lstm"]["score"]["precision_at_tau"] is None
    assert "diagnostics" not in cands["xgb"]              # bulky blob stripped


def test_search_families(client):
    fams = client.get("/api/search/families").json()
    assert any(f["name"] == "xgboost" for f in fams)


# ── launches → jobs ───────────────────────────────────────────────────────────

def test_scoping_launch_creates_job_and_single_flight(client, monkeypatch):
    monkeypatch.setattr(ds, "build_argv_scoping",
                        lambda oid, dry, budget: ["/bin/sleep", "5"])
    r = client.post("/api/scoping/launch", headers=H,
                    json={"objective_id": "nq-50pt-tbm", "dry_run": True})
    assert r.status_code == 201, r.text
    job = r.json()["job"]
    assert job["status"] == "running" and job["kind"] == "scoping"
    # duplicate for the same objective is rejected while running
    dup = client.post("/api/scoping/launch", headers=H,
                      json={"objective_id": "nq-50pt-tbm", "dry_run": True})
    assert dup.status_code == 409
    # cancel it
    c = client.post(f"/api/jobs/{job['job_id']}/cancel", headers=H)
    assert c.status_code == 200
    for _ in range(40):
        cur = client.get(f"/api/jobs/{job['job_id']}").json()
        if cur["status"] != "running":
            break
        time.sleep(0.25)
    assert cur["status"] == "cancelled"


def test_job_log_tail_and_completion(client, monkeypatch):
    monkeypatch.setattr(ds, "build_argv_scoping",
                        lambda oid, dry, budget: ["/bin/sh", "-c", "echo hello-console; exit 0"])
    job = client.post("/api/scoping/launch", headers=H,
                      json={"objective_id": "nq-50pt", "dry_run": True}).json()["job"]
    for _ in range(40):
        cur = client.get(f"/api/jobs/{job['job_id']}").json()
        if cur["status"] != "running":
            break
        time.sleep(0.25)
    assert cur["status"] == "succeeded" and cur["exit_code"] == 0
    log = client.get(f"/api/jobs/{job['job_id']}/log").json()
    assert "hello-console" in log["content"]
    tail = client.get(f"/api/jobs/{job['job_id']}/log?offset={log['offset']}").json()
    assert tail["content"] == ""                          # nothing new past EOF


def test_search_launch_validates_families(client):
    r = client.post("/api/search/launch", headers=H,
                    json={"objective_id": "nq-50pt-tbm", "families": ["not_a_family"]})
    assert r.status_code == 400 and "unknown families" in r.json()["detail"]


def test_argv_builders():
    argv = ds.build_argv_scoping("nq-50pt-tbm", True, 25.0)
    assert argv[1:] == ["-m", "src.scoping.orchestrator", "--objective", "nq-50pt-tbm",
                        "--dry-run", "--budget", "25.0"]
    argv = ds.build_argv_search("nq-50pt", ["xgboost"], False, 10, True, None)
    assert "--families" in argv and "--tune-trials" in argv and "--no-ensemble" in argv
    argv, env = ds.build_argv_director_cycle(True)
    assert argv[1:] == ["-m", "src.research_director.run", "--dry-run"]
    # Repo root on PYTHONPATH (not …/src) so `from src.*` resolves.
    assert env["PYTHONPATH"].endswith("src") is False
    assert Path(env["PYTHONPATH"]).name in ("opt-vtech", "2026-08-05-21516cb0") or (
        Path(env["PYTHONPATH"]) / "src").is_dir()
    with pytest.raises(ValueError):
        ds.build_argv_scoping("Bad Id!", False, None)
    with pytest.raises(ValueError):
        ds.build_argv_search("nq-50pt", None, False, 0, False, "../../etc/passwd")


# ── backtest (staged weights via src.serving — same as Live) ───────────────────

class _AlwaysProba:
    """Module-level so joblib can pickle it for serving-bundle fixtures."""

    def __init__(self, side: str = "up"):
        self.side = side

    def predict_proba(self, X):
        import numpy as np
        n = len(X)
        out = np.full((n, 3), 0.05, dtype="float64")
        out[:, {"down": 0, "flat": 1, "up": 2}[self.side]] = 0.9
        return out


def _mk_scorable_bundle(root: Path, objective_id: str, name: str, *,
                        candidate_id: str | None = None,
                        always: str = "up") -> Path:
    """Serving bundle with a tiny joblib model that always predicts *always* at high conf."""
    import joblib

    d = root / objective_id / name
    d.mkdir(parents=True)
    (d / "feature_schema.json").write_text(json.dumps({"feature_cols": ["f1"]}))
    joblib.dump(_AlwaysProba(always), d / "model.pkl")
    (d / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": objective_id,
        "candidate_id": candidate_id or name,
        "family": "sklearn",
        "version": name,
        "spec_params": {"dir_threshold": 0.5},
    }))
    return d


def test_backtest_models_surface_inference_policy(tmp_path, monkeypatch, client):
    from src.webui import live as live_mod

    staging = tmp_path / "model_staging"
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", staging)
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)

    d = _mk_scorable_bundle(
        staging, "nq-50pt-xgb-unique", "xgb_causal_topgroups_20260821T110058Z",
        candidate_id="xgb_causal_topgroups")
    pol = {
        "tau": 0.65, "trading_start": "08:30", "trading_end": "10:00",
        "sides": ["up"], "cooldown_bars": 120, "note": "causal long timer",
    }
    man = json.loads((d / "finalize_manifest.json").read_text())
    man["inference_policy"] = pol
    (d / "finalize_manifest.json").write_text(json.dumps(man))
    (d / "inference_policy.json").write_text(json.dumps(pol))
    _mk_features(tmp_path / "parquet", "2026-05-01", start="2026-05-01 13:30", rows=10)

    models = client.get("/api/backtest/models").json()
    m = next((x for x in models if x["model"] == "xgb_causal_topgroups"), None)
    assert m is not None
    assert m["recommended_tau"] == pytest.approx(0.65)
    assert m["inference_policy"]["trading_end"] == "10:00"
    assert "causal long timer" in m["detail"]
    assert "08:30" in (m["label"] or "")


def test_bundle_run_label_uses_parent_and_timestamp_folder():
    from src.webui.live import bundle_run_label
    assert bundle_run_label(
        "/opt/vtech/data/models/model_staging/nq-50pt-econ/"
        "xgb_tbm_topgroups_20260715T234308Z"
    ) == "nq-50pt-econ/xgb_tbm_topgroups_20260715T234308Z"
    assert bundle_run_label("model_staging/orphan_bundle") == "orphan_bundle"


def test_backtest_models_listing(tmp_path, monkeypatch, client):
    from src.webui import live as live_mod

    staging = tmp_path / "model_staging"
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", staging)
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)

    _mk_scorable_bundle(staging, "nq-50pt-tbm", "demo_weights", candidate_id="xgb_demo")
    _mk_features(tmp_path / "parquet", "2026-05-01", start="2026-05-01 13:30", rows=10)

    models = client.get("/api/backtest/models").json()
    m = next((m for m in models if m["model"] == "xgb_demo"), None)
    assert m is not None
    assert m["objective_id"] == "nq-50pt-tbm"
    assert m["source_kind"] == "weights"
    assert "demo_weights" in m["source"]
    assert m["backtestable"] is True
    assert m["run_label"] == "nq-50pt-tbm/demo_weights"
    assert "LIVE WEIGHTS" not in m["label"]
    assert "nq-50pt-tbm/demo_weights" in m["label"]

    run = tmp_path / "research" / "search" / "nq-50pt-econ" / "20260815T202407Z"
    run.mkdir(parents=True)
    (run / "search_result.json").write_text("{}")
    (run / "oof_xgboost.parquet").write_bytes(b"oof")
    _mk_scorable_bundle(run.parent, "20260815T202407Z", "xgboost", candidate_id="xgboost")
    models2 = client.get("/api/backtest/models").json()
    assert not any(x.get("source_kind") == "oof" for x in models2)
    assert not any("20260815T202407Z" in str(x.get("run_label") or "") for x in models2)


def test_backtest_run_rejects_source_outside_staging(client):
    for bad in ("../../etc/passwd", "nope/missing", "/tmp/evil"):
        r = client.post("/api/backtest/run", headers=H, json={"source": bad})
        assert r.status_code == 400, bad


def test_backtest_browse_search_oof(tmp_path, monkeypatch, client):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    (tmp_path / "models" / "model_staging").mkdir(parents=True)
    run = tmp_path / "research" / "search" / "nq-50pt-econ" / "20260815T202407Z"
    run.mkdir(parents=True)
    (run / "oof_xgboost.parquet").write_bytes(b"not-a-real-parquet")
    (run / "search_result.json").write_text("{}")

    roots = client.get("/api/backtest/browse").json()["entries"]
    assert any(e["name"] == "Search runs" for e in roots)
    search_root = next(e["path"] for e in roots if e["name"] == "Search runs")
    objs = client.get("/api/backtest/browse", params={"path": search_root}).json()
    assert any(e["name"] == "nq-50pt-econ" for e in objs["entries"])
    listing = client.get("/api/backtest/browse", params={"path": str(run)}).json()
    assert not any(e.get("kind") == "oof" for e in listing["entries"])
    assert client.get("/api/backtest/browse", params={"path": "/etc"}).status_code == 400

    _mk_scorable_bundle(run, "", "xgboost", candidate_id="xgboost")
    listing2 = client.get("/api/backtest/browse", params={"path": str(run)}).json()
    bun = next(e for e in listing2["entries"] if e["kind"] == "bundle")
    assert bun["name"] == "xgboost"
    assert bun["objective_id"] == "nq-50pt-econ"
    assert bun["run_label"] == "nq-50pt-econ/20260815T202407Z/xgboost"

    monkeypatch.setattr(
        "src.backtest.weights_score.run_weights_backtest_inmemory",
        lambda **kw: {"config": {"objective_id": kw.get("objective_id"),
                                 "source": str(kw.get("model_path"))},
                      "summary": {"n_trades": 0}, "equity_curve": [], "trades": []},
    )
    r_bad = client.post("/api/backtest/run", headers=H, json={
        "source": str(run / "xgboost"),
        "objective_id": "20260815T202407Z",
        "tau": 0.8,
    })
    assert r_bad.status_code == 200, r_bad.text
    assert r_bad.json()["config"]["objective_id"] == "nq-50pt-econ"

    r = client.post("/api/backtest/run", headers=H, json={
        "source": str(run / "xgboost"),
        "objective_id": "nq-50pt-econ",
        "tau": 0.8,
    })
    assert r.status_code == 200, r.text
    assert r.json()["config"]["objective_id"] == "nq-50pt-econ"


def test_backtest_lists_custom_test_bundle(tmp_path, monkeypatch, client):
    from src.webui import live as live_mod

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "models" / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "models" / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)
    (tmp_path / "models" / "model_staging").mkdir(parents=True)
    (tmp_path / "models" / "model_registry").mkdir(parents=True)
    run = tmp_path / "research" / "custom_test" / "nq-50pt-econ" / "20260820T010203Z"
    bun = _mk_scorable_bundle(run, "models", "xgboost", candidate_id="xgboost")
    man = json.loads((bun / "finalize_manifest.json").read_text())
    man["objective_id"] = "nq-50pt-econ"
    man["search_run_ts"] = "20260820T010203Z"
    (bun / "finalize_manifest.json").write_text(json.dumps(man))

    models = client.get("/api/backtest/models").json()
    hit = next((m for m in models if m.get("source_kind") == "custom_test"), None)
    assert hit is not None
    assert "custom test" in hit["label"]
    assert hit["objective_id"] == "nq-50pt-econ"
    assert hit["model"] == "xgboost"

    roots = client.get("/api/backtest/browse").json()["entries"]
    assert any(e["name"] == "Custom tests" for e in roots)
    listing = client.get("/api/backtest/browse", params={"path": str(run / "models")}).json()
    bun_e = next(e for e in listing["entries"] if e["kind"] == "bundle")
    assert bun_e["name"] == "xgboost"

    monkeypatch.setattr(
        "src.backtest.weights_score.run_weights_backtest_inmemory",
        lambda **kw: {"config": {"objective_id": kw.get("objective_id"),
                                 "source": str(kw.get("model_path"))},
                      "summary": {"n_trades": 0}, "equity_curve": [], "trades": []},
    )
    r = client.post("/api/backtest/run", headers=H, json={
        "source": str(bun), "objective_id": "nq-50pt-econ", "tau": 0.8,
    })
    assert r.status_code == 200, r.text


def test_backtest_lists_oof_only_custom_test_dir(tmp_path, monkeypatch, client):
    from src.webui import live as live_mod
    from src.webui.datasources import classify_backtest_source

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "models" / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "models" / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)
    (tmp_path / "models" / "model_staging").mkdir(parents=True)
    (tmp_path / "models" / "model_registry").mkdir(parents=True)
    _mk_features(tmp_path / "parquet", "2026-05-01", start="2026-05-01 13:30", rows=10)
    import pandas as pd
    oof_df = pd.DataFrame({
        "ts": pd.date_range("2025-09-02", "2026-06-30", periods=4, tz="UTC"),
        "direction": ["up", "flat", "down", "up"],
        "confidence": [0.5, 0.2, 0.5, 0.5],
        "p_long": [0.5, 0.2, 0.1, 0.5],
        "p_short": [0.1, 0.1, 0.5, 0.1],
    })
    run = tmp_path / "research" / "custom_test" / "nq-50pt-econ-lg-st" / "20260829T182514Z"
    meta = run / "models" / "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m"
    meta.mkdir(parents=True)
    oof_df.to_parquet(meta / "oof.parquet", index=False)
    (meta / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": "nq-50pt-econ-lg-st",
        "candidate_id": "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m",
        "search_run_ts": "20260829T182514Z",
        "native_tf": "15s",
    }))
    lstm = run / "models" / "lstm_15s"
    lstm.mkdir(parents=True)
    oof_df.to_parquet(lstm / "oof.parquet", index=False)
    (lstm / "feature_schema.json").write_text(json.dumps({"feature_cols": ["cs_close"]}))
    (lstm / "model.keras").write_bytes(b"K")
    (lstm / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": "nq-50pt-econ-lg-st",
        "candidate_id": "lstm_15s",
        "search_run_ts": "20260829T182514Z",
        "native_tf": "15s",
    }))

    models = client.get("/api/backtest/models").json()
    oof_hit = next((m for m in models if m.get("model") == "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m"), None)
    assert oof_hit is not None
    assert oof_hit["source_kind"] == "custom_test"
    assert oof_hit["backtestable"] is True
    assert "15s" in oof_hit["label"]
    lstm_hit = next((m for m in models if m.get("model") == "lstm_15s"), None)
    assert lstm_hit is not None
    assert "15s" in lstm_hit["label"]
    assert lstm_hit.get("recommended_tau") == 0.4
    assert lstm_hit.get("span_start") == "2025-09-02"
    assert lstm_hit.get("span_end") == "2026-06-30"

    listing = client.get("/api/backtest/browse", params={"path": str(run / "models")}).json()
    kinds = {e["name"]: e["kind"] for e in listing["entries"]}
    assert kinds["lstm_15s"] == "bundle"
    assert kinds["tf_oof_meta__lstm_15s_xgb_1m_xgb_5m"] == "oof"
    assert all(e["selectable"] for e in listing["entries"] if e["name"] in kinds)

    kind, path = classify_backtest_source(str(meta))
    assert kind == "oof"
    assert path == meta / "oof.parquet"
    kind2, path2 = classify_backtest_source(str(lstm))
    assert kind2 == "bundle"
    assert path2 == lstm


def test_backtest_run_tau_out_of_range_422(client):
    r = client.post("/api/backtest/run", headers=H,
                    json={"source": "/opt/vtech/data/model_staging/x", "tau": 1.5})
    assert r.status_code == 422                              # pydantic ge/le bound


def test_backtest_run_passes_confidence_gate_flag(tmp_path, monkeypatch, client):
    from src.webui import live as live_mod

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)
    bundle = _mk_scorable_bundle(
        tmp_path / "model_staging", "nq-50pt-tbm", "gate_flag",
        candidate_id="gate_flag")
    captured = {}

    def _capture(**kw):
        captured.update(kw)
        return {"config": {"tau": kw.get("tau"),
                           "confidence_gate": kw.get("confidence_gate")},
                "summary": {"n_trades": 0}, "equity_curve": [], "trades": []}

    monkeypatch.setattr(
        "src.backtest.weights_score.run_weights_backtest_inmemory", _capture)

    r = client.post("/api/backtest/run", headers=H, json={
        "source": str(bundle), "objective_id": "nq-50pt-tbm",
        "tau": 0.05, "confidence_gate": False,
    })
    assert r.status_code == 200, r.text
    assert captured["tau"] == 0.05
    assert captured["confidence_gate"] is False
    assert r.json()["config"]["confidence_gate"] is False

    captured.clear()
    r2 = client.post("/api/backtest/run", headers=H, json={
        "source": str(bundle), "objective_id": "nq-50pt-tbm", "tau": 0.8,
    })
    assert r2.status_code == 200, r2.text
    assert captured["confidence_gate"] is True


def test_backtest_run_accepts_tbbo_data_flag(client, tmp_path, monkeypatch):
    """tbbo_data=True routes to signal-first TBBO path; empty prints → timeouts."""
    from src.webui import live as live_mod

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)

    bundle = _mk_scorable_bundle(
        tmp_path / "model_staging", "nq-50pt-tbm", "tick_flag",
        candidate_id="tick_flag", always="up")
    import numpy as np
    import pandas as pd
    pdir = tmp_path / "parquet"
    pdir.mkdir(parents=True)
    n_bars = 300
    idx = pd.date_range("2026-05-01 13:30", periods=n_bars, freq="5s", tz="UTC")
    feat = pd.DataFrame({
        "_close": 100.0 + np.arange(n_bars, dtype="float64"),
        "f1": np.ones(n_bars, dtype="float64"),
    }, index=idx)
    feat.to_parquet(pdir / "features_2026-05-01.parquet")
    from src.features.hive_5s import persist_session_day
    persist_session_day(feat)

    # Empty TBBO window → still emits signal rows as timeouts (0 pnl).
    def _empty_tbbo(**_kw):
        z = np.empty(0, dtype="float64")
        return np.empty(0, dtype="int64"), z, z.copy(), z.copy()

    monkeypatch.setattr(
        "src.backtest.weights_score._load_tbbo_window", _empty_tbbo)

    r = client.post("/api/backtest/run", headers=H, json={
        "source": str(bundle), "objective_id": "nq-50pt-tbm", "tau": 0.5,
        "start": "2026-05-01", "end": "2026-05-01", "tbbo_data": True,
        "mae_adjust": 1.0, "target_adjust": 5.0,
    })
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["config"]["tbbo_data"] is True or p["config"].get("oco") is True
    assert p["config"]["fill_path"] in ("oco", "tbbo_data")
    assert p["config"]["mae_adjust"] == 1.0
    assert p["config"]["target_adjust"] == 5.0
    # target 50+5, stop 6+1
    assert p["config"]["target_points"] == 55.0
    assert p["config"]["stop_points"] == 7.0
    assert p["summary"]["n_trades"] >= 1


def test_backtest_run_end_to_end(client, tmp_path, monkeypatch):
    """HTTP API: staged weights + feature parquet → src.serving → path engine."""
    import numpy as np
    import pandas as pd
    from src.webui import live as live_mod

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)

    bundle = _mk_scorable_bundle(
        tmp_path / "model_staging", "nq-50pt-tbm", "e2e_weights",
        candidate_id="e2e_weights", always="up")

    # Feature day: long enough for TBM label horizon (10m = 120 bars) + path target.
    # Rising path so long target (+50) hits; include model feature f1.
    pdir = tmp_path / "parquet"
    pdir.mkdir(parents=True)
    n_bars = 300
    idx = pd.date_range("2026-05-01 13:30", periods=n_bars, freq="5s", tz="UTC")
    # +2/bar for 40 bars then flat — target hit early; labels scorable before overhang
    close = np.concatenate([
        100.0 + 2.0 * np.arange(40, dtype="float64"),
        np.full(n_bars - 40, 100.0 + 2.0 * 39, dtype="float64"),
    ])
    feat = pd.DataFrame({
        "_close": close,
        "f1": np.ones(n_bars, dtype="float64"),
    }, index=idx)
    feat.to_parquet(pdir / "features_2026-05-01.parquet")
    from src.features.hive_5s import persist_session_day
    persist_session_day(feat)

    r = client.post("/api/backtest/run", headers=H,
                    json={"source": str(bundle), "objective_id": "nq-50pt-tbm", "tau": 0.8,
                          "max_positions": 1, "start": "2026-05-01", "end": "2026-05-01"})
    assert r.status_code == 200, r.text
    p = r.json()
    assert set(p) >= {"config", "summary", "equity_curve", "confusion_matrix", "trades"}
    assert p["config"]["tau"] == 0.8
    assert p["config"]["prediction_mode"] == "weights"
    assert p["config"]["candidate_id"] == "e2e_weights"
    assert p["config"]["target_points"] == 50 and p["config"]["stop_points"] == 6
    assert p["summary"]["n_trades"] >= 1
    t0 = p["trades"][0]
    assert t0["side"] == "long" and t0["exit_reason"] == "target"
    assert t0["pnl_points"] > 0 and p["summary"]["net_pnl"] > 0
    assert p["confusion_matrix"]["labels"] == ["down", "flat", "up"]
    # on_close / OCO signal paths may leave CM empty (no bar entry_idx / OOF labels)
    assert p["config"].get("fill_path") in ("on_close", "oco", "bar_5s", "tbbo_data")


def test_backtest_run_confidence_gate_off_admits_below_tau(client, tmp_path, monkeypatch):
    """AlwaysProba conf=0.9 is blocked at τ=0.95 unless the gate is off."""
    import numpy as np
    import pandas as pd
    from src.webui import live as live_mod

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "_PARQUET_ROOT", tmp_path / "parquet")
    monkeypatch.setattr(ds, "_DATA_ROOT", tmp_path)

    bundle = _mk_scorable_bundle(
        tmp_path / "model_staging", "nq-50pt-tbm", "gate_off",
        candidate_id="gate_off", always="up")
    pdir = tmp_path / "parquet"
    pdir.mkdir(parents=True)
    n_bars = 300
    idx = pd.date_range("2026-05-01 13:30", periods=n_bars, freq="5s", tz="UTC")
    close = np.concatenate([
        100.0 + 2.0 * np.arange(40, dtype="float64"),
        np.full(n_bars - 40, 100.0 + 2.0 * 39, dtype="float64"),
    ])
    feat = pd.DataFrame({
        "_close": close,
        "f1": np.ones(n_bars, dtype="float64"),
    }, index=idx)
    feat.to_parquet(pdir / "features_2026-05-01.parquet")
    from src.features.hive_5s import persist_session_day
    persist_session_day(feat)

    body = {"source": str(bundle), "objective_id": "nq-50pt-tbm",
            "tau": 0.95, "start": "2026-05-01", "end": "2026-05-01"}
    gated = client.post("/api/backtest/run", headers=H,
                        json={**body, "confidence_gate": True})
    assert gated.status_code == 200, gated.text
    assert gated.json()["summary"]["n_trades"] == 0
    assert gated.json()["config"]["confidence_gate"] is True

    ungated = client.post("/api/backtest/run", headers=H,
                          json={**body, "confidence_gate": False})
    assert ungated.status_code == 200, ungated.text
    assert ungated.json()["summary"]["n_trades"] >= 1
    assert ungated.json()["config"]["confidence_gate"] is False
    assert ungated.json()["config"]["tau"] == 0.95


# ── security guards ───────────────────────────────────────────────────────────

def test_mutating_requires_console_header(client):
    r = client.post("/api/objectives/validate", json={"objective": {}})
    assert r.status_code == 403
    r = client.post("/api/director/timer", json={"action": "enable"})
    assert r.status_code == 403


def test_host_header_allowlist(client):
    # Bind is 127.0.0.1; VS Code Ports send non-localhost Host values.
    # A 400 Invalid host is what made the forwarded tab spin forever.
    r = client.get("/api/overview", headers={"Host": "evil.example.com"})
    assert r.status_code == 200


def test_spa_shell_is_not_cached(client):
    r = client.get("/")
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "").lower()
    assert "no-store" in cc
    assert r.headers.get("x-vtech-rev") == "20260827-ports"


# ── overview + director status (read-only, environment-tolerant) ─────────────

def test_cme_calendar_holidays(client):
    r = client.get("/api/calendar/cme")
    assert r.status_code == 200
    body = r.json()
    assert body["timezone"] == "America/Chicago"
    days = body["holidays"]
    assert "2026-01-01" in days
    assert "2026-07-02" not in days
    assert "2026-07-03" in days
    assert "2026-04-03" in days
    assert all(isinstance(d, str) and len(d) == 10 for d in days)
    assert days == sorted(days)


def test_overview_shape(client):
    o = client.get("/api/overview").json()
    assert {"active_objective", "services", "running_jobs", "scoping_runs",
            "search_runs", "vm", "mlflow_url"} <= set(o)


def test_director_status_shape(client):
    d = client.get("/api/director/status").json()
    assert {"timer", "service", "config", "db"} <= set(d)
    assert d["timer"]["unit"] == "vtech-research-director.timer"


# ── live model resolution (no longer hardcoded to a TBM bundle) ───────────────

def _mk_serving_bundle(root: Path, objective_id: str, name: str, *,
                       candidate_id: str | None = None) -> Path:
    d = root / objective_id / name
    d.mkdir(parents=True)
    (d / "feature_schema.json").write_text(json.dumps({"feature_cols": ["of_delta", "rv_vol"]}))
    (d / "model.ubj").write_bytes(b"dummy")          # presence-only; not loaded in resolve tests
    (d / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": objective_id,
        "candidate_id": candidate_id or name,
        "family": "xgboost",
        "version": name,
        "spec_params": {"dir_threshold": 0.5},
    }))
    return d


def test_resolve_live_model_prefers_active_objective(tmp_path, monkeypatch):
    from src.webui import live as live_mod
    staging = tmp_path / "model_staging"
    registry = tmp_path / "model_registry"
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", staging)
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", registry)
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(live_mod, "_SELECTION_PATH", tmp_path / "webui" / "live_model.json")
    monkeypatch.delenv("VTECH_LIVE_MODEL", raising=False)

    old = _mk_serving_bundle(staging, "nq-50pt-tbm", "xgboost_20260711T033354Z")
    new = _mk_serving_bundle(staging, "nq-50pt-econ", "xgb_tbm_topgroups_20260715T234308Z",
                             candidate_id="xgb_tbm_topgroups")
    # Make the econ bundle newer
    import os, time
    now = time.time()
    os.utime(new / "finalize_manifest.json", (now, now))
    os.utime(old / "finalize_manifest.json", (now - 1000, now - 1000))

    r = live_mod.resolve_live_model("nq-50pt-econ")
    assert r["name"] == "xgb_tbm_topgroups_20260715T234308Z"
    assert r["objective_id"] == "nq-50pt-econ"
    assert "033354Z" not in r["path"]
    assert r.get("note") is None

    # Wrong objective → falls back with a warning note
    r2 = live_mod.resolve_live_model("does-not-exist")
    assert r2["note"] and "no staged" in r2["note"]


def test_resolve_live_model_env_override(tmp_path, monkeypatch):
    from src.webui import live as live_mod
    staging = tmp_path / "model_staging"
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", staging)
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_SELECTION_PATH", tmp_path / "webui" / "live_model.json")
    d = _mk_serving_bundle(staging, "nq-50pt-tbm", "forced_model")
    monkeypatch.setenv("VTECH_LIVE_MODEL", str(d))
    r = live_mod.resolve_live_model("nq-50pt-econ")
    assert r["source"] == "env" and r["name"] == "forced_model"


def test_objective_trade_params_reads_econ():
    from src.webui import live as live_mod
    t = live_mod.objective_trade_params("nq-50pt-econ")
    assert t["objective_id"] == "nq-50pt-econ"
    assert t["move_threshold"] == 50.0 and t["mae_tolerance"] == 6.0
    assert t["horizon_s"] == 600
    assert t["scheme"] == "triple_barrier"
    assert t["instrument"] == "NQ"
    assert t["point_value"] == 20.0          # CME E-mini NQ $ / point


def test_live_model_api(client):
    r = client.get("/api/live/model").json()
    assert "active_objective" in r and "trade" in r
    assert "selection" in r and "available" in r
    assert r["selection"]["mode"] in ("auto", "manual")
    # When the real staging dir has the econ champion, resolution must point at it
    # (unless the operator pinned something else).
    if (r.get("resolved") and r["active_objective"] == "nq-50pt-econ"
            and r["selection"]["mode"] == "auto"):
        assert "nq-50pt-econ" in r["resolved"]["path"]
        assert "033354Z" not in r["resolved"]["path"]


def test_live_model_select_manual_and_auto(tmp_path, monkeypatch, client):
    from src.webui import live as live_mod
    staging = tmp_path / "model_staging"
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", staging)
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_SELECTION_PATH", tmp_path / "webui" / "live_model.json")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.delenv("VTECH_LIVE_MODEL", raising=False)
    a = _mk_serving_bundle(staging, "nq-50pt-econ", "econ_a", candidate_id="econ_a")
    b = _mk_serving_bundle(staging, "nq-50pt-tbm", "tbm_b", candidate_id="tbm_b")

    # Manual pin via API
    r = client.post("/api/live/model", headers=H,
                    json={"mode": "manual", "path": str(b)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["selection"]["mode"] == "manual"
    assert Path(body["resolved"]["path"]).resolve() == b.resolve()
    assert body["resolved"]["selection_mode"] == "manual"

    # Back to auto — should prefer active objective's bundle when present
    monkeypatch.setattr(live_mod, "active_objective_id", lambda: "nq-50pt-econ")
    r2 = client.post("/api/live/model", headers=H, json={"mode": "auto"})
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["selection"]["mode"] == "auto"
    assert "nq-50pt-econ" in body2["resolved"]["path"]

    # Reject path outside staging/registry
    bad = client.post("/api/live/model", headers=H,
                      json={"mode": "manual", "path": str(tmp_path / "evil")})
    assert bad.status_code == 400


def test_live_reload_api(client):
    r = client.post("/api/live/reload", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["snapshot_refresh"] == "scheduled"
    assert body["streamer"]["status"] == "starting"


def test_live_model_select_does_not_build_snapshot_synchronously(monkeypatch, client):
    from src.webui import live as live_mod

    def _raise():
        raise AssertionError("build_snapshot should not run in request thread")

    monkeypatch.setattr(live_mod, "build_snapshot", _raise)
    monkeypatch.setattr(
        live_mod,
        "set_selection",
        lambda *, mode="auto", path=None: {"mode": mode, "path": path},
    )

    r = client.post("/api/live/model", headers=H, json={"mode": "auto"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["snapshot_refresh"] == "scheduled"
    assert body["streamer"]["status"] == "starting"


def test_live_reload_does_not_build_snapshot_synchronously(monkeypatch, client):
    from src.webui import live as live_mod

    def _raise():
        raise AssertionError("build_snapshot should not run in request thread")

    monkeypatch.setattr(live_mod, "build_snapshot", _raise)

    r = client.post("/api/live/reload", headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["snapshot_refresh"] == "scheduled"
    assert body["streamer"]["status"] == "starting"


def test_session_context_uses_objective_tz_not_utc():
    """After CT close but before UTC midnight roll, session day stays the CT calendar day."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.webui import live as live_mod

    # 2026-07-15 19:53 CT == 2026-07-16 00:53 UTC (past 14:45 session end).
    now = datetime(2026, 7, 16, 0, 53, tzinfo=ZoneInfo("UTC"))
    ctx = live_mod.session_context("nq-50pt-econ", now=now)
    assert str(ctx["session_date"]) == "2026-07-15"
    assert ctx["session_closed"] is True
    assert ctx["in_session"] is False
    assert ctx["timezone"] == "America/Chicago"


def test_session_context_pre_open_rolls_back():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.webui import live as live_mod

    # 2026-07-16 07:00 CT — before 08:00 open → blotter day is previous trading day.
    now = datetime(2026, 7, 16, 12, 0, tzinfo=ZoneInfo("UTC"))  # 07:00 CT
    ctx = live_mod.session_context("nq-50pt-econ", now=now)
    assert ctx["in_session"] is False
    # 2026-07-15 is a Wednesday (trading day)
    assert str(ctx["session_date"]) == "2026-07-15"


def test_session_context_feature_date_rolls_at_training_start():
    """08:00–08:29 CT is still yesterday on the blotter, today for feature files."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.webui import live as live_mod

    # 2026-08-26 08:15 CT
    now = datetime(2026, 8, 26, 13, 15, tzinfo=ZoneInfo("UTC"))
    blotter = live_mod.session_context("nq-50pt-econ", now=now)
    assert str(blotter["session_date"]) == "2026-08-25"
    assert blotter["in_session"] is False
    feat = live_mod.session_context("nq-50pt-econ", now=now, roll_at="training_start")
    assert str(feat["session_date"]) == "2026-08-26"
    assert feat["in_session"] is False


def test_signal_export_window_enforces_registered_model_session_boundaries():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.webui import live as live_mod

    session = {"date": "2026-08-06", "timezone": "America/Chicago"}
    at_start = datetime(2026, 8, 6, 13, 30, tzinfo=ZoneInfo("UTC"))
    at_end = datetime(2026, 8, 6, 19, 45, tzinfo=ZoneInfo("UTC"))

    assert not live_mod._is_within_signal_export_window(
        "2026-08-06T13:29:55+00:00", session,
        trading_start="08:30", trading_end="14:45", now=at_start)
    assert live_mod._is_within_signal_export_window(
        "2026-08-06T13:30:00+00:00", session,
        trading_start="08:30", trading_end="14:45", now=at_start)
    assert live_mod._is_within_signal_export_window(
        "2026-08-06T19:45:00+00:00", session,
        trading_start="08:30", trading_end="14:45", now=at_end)
    assert not live_mod._is_within_signal_export_window(
        "2026-08-06T19:45:05+00:00", session,
        trading_start="08:30", trading_end="14:45", now=at_end)


def test_signal_export_uses_registered_model_session_override(rm_env, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.webui import live as live_mod
    from src.webui import regmodel

    monkeypatch.setattr(regmodel, "effective_settings", lambda _: {
        "trading_start": "08:30", "trading_end": "14:45"})
    monkeypatch.setattr(live_mod, "_SIG_STATE", {"last_ts": None, "path": None})
    snap = {
        "objective_id": "nq-50pt-tbm",
        "session": {"date": "2026-08-06", "timezone": "America/Chicago"},
        "barriers": {},
    }
    ledger = [
        {"ts": "2026-08-06T13:30:00+00:00", "sig": True, "dir": "up", "conf": 0.9},
        {"ts": "2026-08-06T13:29:55+00:00", "sig": True, "dir": "up", "conf": 0.9},
    ]
    live_mod._export_signals(
        ledger, snap, now=datetime(2026, 8, 6, 13, 31, tzinfo=ZoneInfo("UTC")))

    records = [json.loads(line) for line in (rm_env / "live" / "signals.jsonl").read_text().splitlines()]
    assert [record["ts"] for record in records] == ["2026-08-06T13:30:00+00:00"]
    heartbeat = json.loads((rm_env / "live" / "heartbeat.json").read_text())
    assert heartbeat["signal_export_window"]["trading_start"] == "08:30"


# ── registered model (per-objective sub-page) ─────────────────────────────────

@pytest.fixture()
def rm_env(tmp_path, monkeypatch):
    """Isolated data root + no DB: streamer health stubbed, tick loader stubbed."""
    from src.webui import live as live_mod
    from src.webui import regmodel

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_mod, "_STAGING_ROOT", tmp_path / "model_staging")
    monkeypatch.setattr(live_mod, "_REGISTRY_ROOT", tmp_path / "model_registry")
    monkeypatch.setattr(live_mod, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(live_mod, "streamer_health", lambda: {"live": False})
    monkeypatch.setattr(regmodel, "_load_ticks", lambda day: None)
    regmodel.invalidate_cache()
    yield tmp_path
    regmodel.invalidate_cache()


def test_regmodel_status_and_settings_roundtrip(client, rm_env):
    base = "/api/objectives/nq-50pt-tbm/model"

    st = client.get(base).json()
    assert st["objective_id"] == "nq-50pt-tbm"
    assert st["has_model"] is False and st["active"] is False
    assert set(st["active_checks"]) == {"registered_model", "trading_day",
                                        "in_session", "feed_live"}
    assert "market_closed_today" in st and "session" in st
    eff = st["effective"]
    assert eff["order_type"] == "MKT" and eff["execution"] == "OCO"
    assert eff["tau_is_default"] is True
    assert eff["trading_start"] == eff["default_trading_start"]

    # save panel settings
    r = client.post(f"{base}/settings", headers=H,
                    json={"order_type": "LIMIT", "execution": "onclose",
                          "tau": 0.7, "max_positions": 3})
    assert r.status_code == 200, r.text
    eff = r.json()["effective"]
    assert eff["order_type"] == "LIMIT" and eff["execution"] == "ONCLOSE"
    assert eff["tau"] == 0.7 and eff["tau_is_default"] is False
    assert eff["max_positions"] == 3
    # persisted on disk under webui_dir()/regmodel/
    from src.common.data_paths import webui_dir
    assert (webui_dir() / "regmodel" / "nq-50pt-tbm.json").exists()

    # today-only session override, then reset to defaults clears it
    r = client.post(f"{base}/settings", headers=H, json={"trading_start": "09:15"})
    eff = r.json()["effective"]
    assert eff["session_override_active"] is True and eff["trading_start"] == "09:15"
    r = client.post(f"{base}/settings", headers=H,
                    json={"trading_start": eff["default_trading_start"],
                          "trading_end": eff["default_trading_end"]})
    eff = r.json()["effective"]
    assert eff["session_override_active"] is False

    # validation
    assert client.post(f"{base}/settings", headers=H,
                       json={"order_type": "FOO"}).status_code == 400
    assert client.post(f"{base}/settings", headers=H,
                       json={"tau": 1.5}).status_code == 422       # pydantic bound
    assert client.post(f"{base}/settings", headers=H,
                       json={"trading_start": "25:99"}).status_code == 400
    assert client.post(f"{base}/settings", headers=H,
                       json={"trading_start": "14:00",
                             "trading_end": "09:00"}).status_code == 400
    # CSRF header required on the mutating route
    assert client.post(f"{base}/settings", json={"tau": 0.6}).status_code == 403
    # unknown objective
    assert client.get("/api/objectives/no-such/model").status_code == 404


def test_regmodel_settings_override_expires_next_day(rm_env):
    from src.common.data_paths import webui_dir
    from src.webui import regmodel
    regmodel.save_settings("nq-50pt-tbm", {"trading_start": "09:15"})
    s = regmodel.get_settings("nq-50pt-tbm")
    s["session_override"]["date"] = "2020-01-02"               # stale session date
    p = webui_dir() / "regmodel" / "nq-50pt-tbm.json"
    p.write_text(json.dumps(s))
    eff = regmodel.effective_settings("nq-50pt-tbm")
    assert eff["session_override_active"] is False
    assert eff["trading_start"] == eff["default_trading_start"]


def test_regmodel_blotter_no_model(client, rm_env):
    b = client.get("/api/objectives/nq-50pt-tbm/model/blotter").json()
    assert b["rows"] == []
    assert b["stats"]["n_signals"] == 0
    assert "No emitted signals" in (b["meta"].get("note") or "")
    assert b["status"]["has_model"] is False


def test_regmodel_blotter_reads_emitted_signals_only(client, rm_env):
    """The ledger exposes only actual records emitted to signals.jsonl."""
    from src.webui import live as live_mod
    ctx = live_mod.session_context("nq-50pt-tbm")
    day = str(ctx["session_date"])
    signals = rm_env / "live" / "signals.jsonl"
    signals.parent.mkdir(parents=True)
    # 13:30 UTC = 08:30 CT, inside the default 08:00–15:00 CT window.
    ts0 = f"{day}T13:30:00+00:00"
    ts1 = f"{day}T13:30:05+00:00"
    records = [
        {"ts": ts0, "dir": "up", "conf": 0.81, "entry_px": 22000.0, "tgt_pts": 50.0, "stop_pts": 6.0, "horizon_s": 600, "candidate_id": "winner", "objective_id": "nq-50pt-tbm", "session_date": day},
        {"ts": ts1, "dir": "down", "conf": 0.84, "entry_px": 21999.0, "tgt_pts": 50.0, "stop_pts": 6.0, "horizon_s": 600, "candidate_id": "winner", "objective_id": "nq-50pt-tbm", "session_date": day},
        {"ts": f"{day}T13:29:55+00:00", "dir": "up", "objective_id": "nq-50pt-tbm", "session_date": day},
        {"ts": f"{day}T12:00:10+00:00", "dir": "up", "objective_id": "other-objective", "session_date": day},
        {"ts": f"{day}T12:00:15+00:00", "dir": "up", "objective_id": "nq-50pt-tbm", "session_date": "2020-01-01"},
    ]
    signals.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    b = client.get("/api/objectives/nq-50pt-tbm/model/blotter").json()
    st, rows, meta = b["stats"], b["rows"], b["meta"]
    assert meta["source"] == "signals.jsonl"
    assert st == {"n_signals": 2, "n_up": 1, "n_down": 1}
    assert [row["ts"] for row in rows] == [ts1, ts0]
    assert rows[0]["entry_px"] == 21999.0 and rows[0]["candidate_id"] == "winner"
    assert all("pnl_pts" not in row and "exit_reason" not in row for row in rows)


# ── job manager reattach ──────────────────────────────────────────────────────

def test_jobmanager_reattach_and_lost(tmp_path):
    jm = JobManager(root=tmp_path / "jobs")
    job = jm.launch("scoping", "obj-a", ["/bin/sleep", "3"])
    jm2 = JobManager(root=tmp_path / "jobs")               # simulates console restart
    re_job = jm2.get(job.job_id)
    assert re_job is not None and re_job.status == "running" and re_job.reattached
    jm.cancel(job.job_id)
    for _ in range(40):
        if jm.get(job.job_id).status != "running":
            break
        time.sleep(0.25)
    jm3 = JobManager(root=tmp_path / "jobs")
    dead = jm3.launch("scoping", "obj-b", ["/bin/true"])
    time.sleep(0.1)
    meta = json.loads((tmp_path / "jobs" / dead.job_id / "meta.json").read_text())
    meta["status"] = "running"                             # forge a stale 'running' record
    meta["pid"], meta["starttime"] = 99999999, 12345
    (tmp_path / "jobs" / dead.job_id / "meta.json").write_text(json.dumps(meta))
    jm4 = JobManager(root=tmp_path / "jobs")
    assert jm4.get(dead.job_id).status == "lost"


def test_backtest_summary_includes_avg_pnl_tile():
    """Summary view exposes mean realized PnL/trade (path NE) as its own tile."""
    app_js = (REPO / "src" / "webui" / "static" / "app.js").read_text()
    start = app_js.index("function btSummary(")
    end = app_js.index("function btBlotter(")
    summary = app_js[start:end]
    assert 'tile("Avg PnL"' in summary
    assert "s.expectancy" in summary
    assert "per trade (NE)" in summary
    # Expectancy is the Avg PnL tile, not a Profit-factor subtitle.
    assert "expectancy ${fmtPtsUsd(s.expectancy" not in summary


def test_backtest_confidence_dropdown_starts_at_5pct_with_enable_checkbox():
    app_js = (REPO / "src" / "webui" / "static" / "app.js").read_text()
    start = app_js.index("async function viewBacktest(")
    end = app_js.index("window.btRun")
    bt = app_js[start:end]
    assert "for (let pct = 5; pct <= 95; pct += 5)" in bt
    assert 'id="bt-tau-on"' in bt
    assert "enable</label>" in bt
    assert "pct === 80" in bt
    assert "${pct}%" in bt
    run = app_js[app_js.index("window.btRun"):app_js.index("window.btTab")]
    assert "confidence_gate:" in run
    assert '$("#bt-tau-on")?.checked !== false' in run
    results = app_js[app_js.index("function renderBacktestResults("):
                     app_js.index("function btSummary(")]
    assert "confidence gate off" in results
    assert "c.confidence_gate === false" in results


def test_backtest_date_range_includes_mtd_ytd_default_mtd():
    app_js = (REPO / "src" / "webui" / "static" / "app.js").read_text()
    start = app_js.index("async function viewBacktest(")
    end = app_js.index("window.btRun")
    bt = app_js[start:end]
    assert 'option value="mtd" selected>MTD (month to date)</option>' in bt
    assert 'option value="ytd">YTD (year to date)</option>' in bt
    assert "function backtestRangeBounds(" in app_js
    assert 'rangeMode === "mtd" || rangeMode === "ytd"' in app_js
    # previous default must not stay selected
    assert 'option value="90" selected>' not in bt


def test_registered_model_ledger_does_not_render_trigger_prices():
    app_js = (REPO / "src" / "webui" / "static" / "app.js").read_text()
    assert "<th>target pts</th>" not in app_js
    assert "<th>stop pts</th>" not in app_js
    assert "r.tgt_pts" not in app_js
    assert "r.stop_pts" not in app_js
    assert "MKT + OnClose" in app_js


def test_live_streamer_polling_patches_slots_without_replacing_page():
    app_js = (REPO / "src" / "webui" / "static" / "app.js").read_text()
    start = app_js.index("function viewLiveStreamer()")
    end = app_js.index("/* ── custom test")
    live = app_js[start:end]

    assert "view.replaceChildren(next.content.cloneNode(true));" in live
    assert "current.innerHTML !== fresh.innerHTML" in live
    assert "currentSel.innerHTML = freshSel.innerHTML;" in live
    assert live.count("view.innerHTML = `") == 1  # error state only
    assert "let refreshInFlight = false;" in live
    assert "if (refreshInFlight) return;" in live
    assert "setPoll(() => pollStreamer().catch(() => {}), 5000);" in live
    assert "addPoll(() => loadAll(false).catch(() => {}), 15000);" in live
    assert "loadModels().then(() => paint());" in live
    assert "return Promise.resolve();" in live
    for slot in ("live-header", "live-model-summary", "live-ingest-body",
                 "live-pred-slot", "live-blotter-title", "live-blotter-summary",
                 "live-blotter-body"):
        assert slot in live
    assert "currentApply.disabled = freshApply.disabled;" in live

def test_live_scores_previous_complete_bar_when_edge_incomplete():
    import pandas as pd
    from src.webui import live as live_mod

    index = pd.date_range("2026-08-06 13:30", periods=2, freq="5s", tz="UTC")
    df = pd.DataFrame({
        "_close": [100.0, 101.0],
        "f1": [1.0, 1.0],
        "f2": [2.0, float("nan")],
    }, index=index)

    bar_idx, n_missing, note = live_mod._select_score_bar(df, ["f1", "f2"])

    assert bar_idx == index[0]
    assert n_missing == 0
    assert note is not None and "lags live edge" in note


def test_live_waits_only_when_no_complete_feature_vector_exists():
    import pandas as pd
    from src.webui import live as live_mod

    index = pd.date_range("2026-08-06 13:30", periods=2, freq="5s", tz="UTC")
    df = pd.DataFrame({
        "_close": [100.0, 101.0],
        "f1": [float("nan"), 1.0],
        "f2": [2.0, float("nan")],
    }, index=index)

    bar_idx, n_missing, note = live_mod._select_score_bar(df, ["f1", "f2"])

    assert bar_idx is None
    assert n_missing == 1
    assert note == "waiting for all 2 required model features (1/2 currently usable)"


def test_build_snapshot_keeps_blotter_when_latest_features_incomplete(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    import pandas as pd
    from src import serving
    from src.webui import live as live_mod

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))

    index = pd.date_range("2026-08-06 13:30", periods=2, freq="5s", tz="UTC")
    df = pd.DataFrame({
        "_close": [100.0, 101.0],
        "f1": [1.0, 1.0],
        "f2": [2.0, float("nan")],
    }, index=index)
    bundle = {
        "feature_cols": ["f1", "f2"],
        "trade": {
            "tau": 0.8,
            "move_threshold": 50.0,
            "mae_tolerance": 6.0,
            "horizon_s": 600,
            "objective_id": "nq-50pt-tbm",
            "instrument": "NQ",
            "point_value": 20.0,
            "scheme": "triple_barrier",
        },
        "resolved": {"source": "staging", "selection_mode": "auto"},
        "version": "test-model",
        "candidate_id": "test-candidate",
        "family": "xgboost",
        "path": "/opt/vtech/data/model_staging/nq-50pt-tbm/test-model",
        "manifest": {},
    }
    monkeypatch.setattr(live_mod, "_bundle", lambda: bundle)
    monkeypatch.setattr(live_mod, "resolve_feature_day", lambda _oid: {
        "df": df,
        "path": "/tmp/features.parquet",
        "session_date": index[0].date(),
        "source": "live_build",
        "note": None,
        "context": {
            "timezone": "America/Chicago",
            "trading_start": "08:00",
            "trading_end": "14:45",
            "in_session": False,
            "session_closed": True,
            "now_local": "2026-08-06T16:00:00-05:00",
        },
    })
    monkeypatch.setattr(serving, "predict_canonical", lambda one_df, _bundle: pd.DataFrame([{
        "ts": one_df.index[0],
        "direction": "up",
        "confidence": 0.91,
        "p_down": 0.03,
        "p_flat": 0.02,
        "p_up": 0.95,
    }], index=one_df.index).assign(ts=one_df.index))
    exported = {"count": 0}
    monkeypatch.setattr(
        live_mod, "_export_signals",
        lambda *_a, **_k: exported.__setitem__("count", exported["count"] + 1),
    )

    snap = live_mod.build_snapshot()
    assert snap["status"] == "ok"
    assert len(snap["ledger"]) == 2
    assert snap["ledger"][0]["feature_complete"] is False  # newest-first
    assert snap["ledger"][1]["feature_complete"] is True
    assert exported["count"] == 1

    page = live_mod.ledger(mode="bar")
    assert page["meta"]["source"] == "signals.jsonl"
    assert page["meta"]["n"] == 0


def test_feature_bar_age_seconds_handles_timezone_inputs():
    from datetime import datetime, timezone
    from src.webui import live as live_mod

    now = datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)
    assert live_mod._feature_bar_age_seconds(
        "2026-08-06T17:59:45+00:00", now=now) == pytest.approx(15.0)
    assert live_mod._feature_bar_age_seconds(
        "2026-08-06T12:59:45-05:00", now=now) == pytest.approx(15.0)
    assert live_mod._feature_bar_age_seconds(
        datetime(2026, 8, 6, 17, 59, 45), now=now) == pytest.approx(15.0)


def test_build_snapshot_rejects_stale_complete_bar_in_session(monkeypatch):
    from datetime import datetime, timedelta, timezone
    import pandas as pd
    from src import serving
    from src.webui import live as live_mod

    # Must exceed VTECH_LIVE_FEATURE_AGE_S / _FEATURE_LIVE_AGE_S (default 90s).
    bar_ts = datetime.now(timezone.utc) - timedelta(seconds=120)
    df = pd.DataFrame({"_close": [100.0], "f1": [1.0]}, index=[bar_ts])
    bundle = {
        "feature_cols": ["f1"],
        "trade": {
            "tau": 0.8,
            "move_threshold": 50.0,
            "mae_tolerance": 6.0,
            "horizon_s": 600,
            "objective_id": "nq-50pt-tbm",
            "instrument": "NQ",
            "point_value": 20.0,
            "scheme": "triple_barrier",
        },
        "resolved": {"source": "staging", "selection_mode": "auto"},
        "version": "test-model",
        "candidate_id": "test-candidate",
        "family": "xgboost",
        "path": "/opt/vtech/data/model_staging/nq-50pt-tbm/test-model",
        "manifest": {},
    }
    monkeypatch.setattr(live_mod, "_bundle", lambda: bundle)
    monkeypatch.setattr(live_mod, "resolve_feature_day", lambda _oid: {
        "df": df,
        "path": "/opt/vtech/data/parquet/features_2026-08-06.parquet",
        "session_date": bar_ts.date(),
        "source": "live_build",
        "note": None,
        "context": {
            "timezone": "America/Chicago",
            "trading_start": "08:00",
            "trading_end": "14:45",
            "in_session": True,
            "session_closed": False,
            "now_local": "2026-08-06T10:00:00-05:00",
        },
    })

    def _fail_predict(_df, _bundle):
        raise AssertionError("stale in-session bar must not be scored")

    monkeypatch.setattr(serving, "predict_canonical", _fail_predict)
    monkeypatch.setattr(live_mod, "_build_ledger", lambda *_a, **_k: [])
    export_called = {"value": False}
    monkeypatch.setattr(
        live_mod,
        "_export_signals",
        lambda *_a, **_k: export_called.__setitem__("value", True),
    )

    snap = live_mod.build_snapshot()

    assert snap["status"] == "warming"
    assert "stale feature bar during active session" in (snap.get("note") or "")
    assert "freshness threshold" in (snap.get("note") or "")
    assert snap["ledger"] == []
    assert export_called["value"] is False
    pred = snap["prediction"]
    assert pred["signal"] is False
    assert {"bar_ts", "n_present", "n_total", "n_nan"} <= set(pred)
    assert all(k not in pred for k in ("direction", "confidence", "p_up", "p_flat", "p_down"))


def test_build_snapshot_allows_stale_complete_bar_outside_session(monkeypatch):
    from datetime import datetime, timedelta, timezone
    import pandas as pd
    from src import serving
    from src.webui import live as live_mod

    bar_ts = datetime.now(timezone.utc) - timedelta(seconds=45)
    df = pd.DataFrame({"_close": [100.0], "f1": [1.0]}, index=[bar_ts])
    bundle = {
        "feature_cols": ["f1"],
        "trade": {
            "tau": 0.8,
            "move_threshold": 50.0,
            "mae_tolerance": 6.0,
            "horizon_s": 600,
            "objective_id": "nq-50pt-tbm",
            "instrument": "NQ",
            "point_value": 20.0,
            "scheme": "triple_barrier",
        },
        "resolved": {"source": "disk", "selection_mode": "auto"},
        "version": "test-model",
        "candidate_id": "test-candidate",
        "family": "xgboost",
        "path": "/opt/vtech/data/model_staging/nq-50pt-tbm/test-model",
        "manifest": {},
    }
    monkeypatch.setattr(live_mod, "_bundle", lambda: bundle)
    monkeypatch.setattr(live_mod, "resolve_feature_day", lambda _oid: {
        "df": df,
        "path": "/opt/vtech/data/parquet/features_2026-08-06.parquet",
        "session_date": bar_ts.date(),
        "source": "disk",
        "note": "session closed",
        "context": {
            "timezone": "America/Chicago",
            "trading_start": "08:00",
            "trading_end": "14:45",
            "in_session": False,
            "session_closed": True,
            "now_local": "2026-08-06T16:00:00-05:00",
        },
    })
    monkeypatch.setattr(serving, "predict_canonical", lambda one_df, _bundle: pd.DataFrame([{
        "ts": one_df.index[0],
        "direction": "up",
        "confidence": 0.91,
        "p_down": 0.03,
        "p_flat": 0.02,
        "p_up": 0.95,
    }]))
    monkeypatch.setattr(live_mod, "_build_ledger", lambda *_a, **_k: [{
        "ts": bar_ts.isoformat(), "sig": True, "feature_complete": True,
    }])
    exported = {"count": 0}
    monkeypatch.setattr(
        live_mod,
        "_export_signals",
        lambda *_a, **_k: exported.__setitem__("count", exported["count"] + 1),
    )

    snap = live_mod.build_snapshot()

    assert snap["status"] == "ok"
    assert snap["prediction"]["direction"] == "up"
    assert snap["prediction"]["signal"] is True
    assert exported["count"] == 1


def test_build_snapshot_persists_snapshot_cache(tmp_path, monkeypatch):
    from src.webui import live as live_mod

    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_PATH", tmp_path / "webui" / "live_snapshot_cache.json")
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_META", {"mtime_ns": None, "version": None})
    live_mod._replace_state({"status": "starting", "prediction": None})

    def _raise_bundle():
        raise RuntimeError("bundle boom")

    monkeypatch.setattr(live_mod, "_bundle", _raise_bundle)
    monkeypatch.setattr(live_mod, "active_objective_id", lambda: "nq-50pt-econ")

    snap = live_mod.build_snapshot()
    payload = json.loads(live_mod._SNAPSHOT_CACHE_PATH.read_text())
    assert snap["status"] == "error"
    assert payload["snapshot"]["status"] == "error"
    assert "bundle boom" in payload["snapshot"]["note"]
    assert isinstance(payload["version"], int)


def test_snapshot_reads_cache_only_when_changed(tmp_path, monkeypatch):
    from src.webui import live as live_mod

    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_PATH", tmp_path / "webui" / "live_snapshot_cache.json")
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_META", {"mtime_ns": None, "version": None})
    live_mod._replace_state({"status": "starting", "prediction": None})

    cache = live_mod._SNAPSHOT_CACHE_PATH
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "version": 1,
        "snapshot": {"status": "ok", "prediction": {"signal": False}},
    }))

    first = live_mod.snapshot()
    assert first["status"] == "ok"

    with live_mod._LOCK:
        live_mod._STATE["status"] = "local-mutated"
    second = live_mod.snapshot()
    assert second["status"] == "local-mutated"

    time.sleep(0.01)
    cache.write_text(json.dumps({
        "version": 2,
        "snapshot": {"status": "warming", "prediction": None},
    }))
    third = live_mod.snapshot()
    assert third["status"] == "warming"


def test_snapshot_cache_read_failure_keeps_state(tmp_path, monkeypatch, caplog):
    from src.webui import live as live_mod

    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_PATH", tmp_path / "webui" / "live_snapshot_cache.json")
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_META", {"mtime_ns": None, "version": None})
    live_mod._replace_state({"status": "ok", "prediction": {"signal": False}})
    cache = live_mod._SNAPSHOT_CACHE_PATH
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{not-json")

    with caplog.at_level("WARNING"):
        snap = live_mod.snapshot()
    assert snap["status"] == "ok"
    assert "live_snapshot_cache_read_failed" in caplog.text


def test_invalidate_cache_does_not_reload_stale_snapshot(tmp_path, monkeypatch):
    from src.webui import live as live_mod

    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_PATH", tmp_path / "webui" / "live_snapshot_cache.json")
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_META", {"mtime_ns": None, "version": None})

    cache = live_mod._SNAPSHOT_CACHE_PATH
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "version": 1,
        "snapshot": {"status": "ok", "prediction": {"signal": True}},
    }))
    live_mod.invalidate_cache()
    snap = live_mod.snapshot()
    assert snap["status"] == "starting"

    time.sleep(0.01)
    cache.write_text(json.dumps({
        "version": 2,
        "snapshot": {"status": "warming", "prediction": None},
    }))
    refreshed = live_mod.snapshot()
    assert refreshed["status"] == "warming"


def test_streamer_health_and_snapshot_use_cache_only(monkeypatch):
    from src.webui import live as live_mod
    import psycopg

    def _fail_connect(*_args, **_kwargs):
        raise AssertionError("psycopg.connect must not be called by snapshot/streamer_health")

    monkeypatch.setattr(psycopg, "connect", _fail_connect)
    monkeypatch.setattr(live_mod, "_STATE", {"status": "ok", "prediction": {"signal": False}})
    monkeypatch.setattr(live_mod, "_STREAMER_HEALTH", {
        "status": "ok",
        "tables": {"tbbo": "2026-08-06T16:40:00+00:00", "bbo_1s": None, "statistics": None},
        "tbbo_rows_10s": 7,
        "tbbo_age_s": 1.2,
        "live": True,
    })

    first = live_mod.streamer_health()
    first["tables"]["tbbo"] = "mutated"
    second = live_mod.streamer_health()
    assert second["tables"]["tbbo"] == "2026-08-06T16:40:00+00:00"

    snap = live_mod.snapshot()
    assert snap["streamer"]["tbbo_rows_10s"] == 7
    assert snap["streamer"]["tables"]["tbbo"] == "2026-08-06T16:40:00+00:00"


def test_lifespan_live_worker_respawns_after_exit(monkeypatch, tmp_path):
    from src.webui import app as app_mod

    spawned = []
    join_calls = []

    class _DeadProc:
        def __init__(self, pid: int):
            self.pid = pid
            self.exitcode = 0

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            join_calls.append((self.pid, timeout))

        def terminate(self) -> None:
            raise AssertionError("terminate should not be called for dead workers")

    def _spawn(_ctx):
        proc = _DeadProc(1000 + len(spawned))
        spawned.append(proc)
        return proc

    monkeypatch.setattr(app_mod, "_LIVE_LOOP_CADENCE_S", 0.01)
    monkeypatch.setattr(app_mod, "_LIVE_LOOP_POLL_S", 0.01)
    monkeypatch.setattr(app_mod.live, "refresh_streamer_health", lambda: {"status": "ok"})
    monkeypatch.setattr(app_mod, "_spawn_live_snapshot_process", _spawn)

    jm = JobManager(root=tmp_path / "jobs-live-worker")
    with TestClient(app_mod.create_app(jobs=jm, enable_live_worker=True), base_url="http://127.0.0.1"):
        time.sleep(0.08)

    assert len(spawned) >= 2
    assert any(timeout == 0.0 for _, timeout in join_calls)


def test_streamer_health_refreshes_while_snapshot_worker_alive(monkeypatch, tmp_path):
    """Feed health must keep updating even while a snapshot worker is running."""
    from src.webui import app as app_mod

    health_calls = []

    class _AliveProc:
        def __init__(self):
            self.pid = 4242
            self.exitcode = None

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(app_mod, "_LIVE_HEALTH_CADENCE_S", 0.01)
    monkeypatch.setattr(app_mod, "_LIVE_LOOP_CADENCE_S", 60.0)
    monkeypatch.setattr(app_mod, "_LIVE_LOOP_POLL_S", 0.01)
    monkeypatch.setattr(
        app_mod.live, "refresh_streamer_health",
        lambda: health_calls.append(time.monotonic()) or {"status": "ok"},
    )
    monkeypatch.setattr(app_mod, "_spawn_live_snapshot_process", lambda _ctx: _AliveProc())

    jm = JobManager(root=tmp_path / "jobs-live-health-independent")
    with TestClient(app_mod.create_app(jobs=jm, enable_live_worker=True), base_url="http://127.0.0.1"):
        time.sleep(0.08)

    assert len(health_calls) >= 3


def test_lifespan_startup_invalidates_persisted_snapshot_cache(monkeypatch, tmp_path):
    from src.webui import app as app_mod
    from src.webui import live as live_mod

    cache = tmp_path / "webui" / "live_snapshot_cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "version": 1,
        "snapshot": {"status": "ok", "prediction": {"signal": True}},
    }))
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_PATH", cache)
    monkeypatch.setattr(live_mod, "_SNAPSHOT_CACHE_META", {"mtime_ns": None, "version": None})
    live_mod._replace_state({"status": "starting", "prediction": None})

    class _AliveProc:
        def __init__(self):
            self.pid = 7331
            self.exitcode = None
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            self._alive = False
            self.exitcode = 0

    monkeypatch.setattr(app_mod, "_LIVE_LOOP_CADENCE_S", 60.0)
    monkeypatch.setattr(app_mod, "_LIVE_LOOP_POLL_S", 0.01)
    monkeypatch.setattr(app_mod.live, "refresh_streamer_health", lambda: {"status": "ok"})
    monkeypatch.setattr(app_mod, "_spawn_live_snapshot_process", lambda _ctx: _AliveProc())
    monkeypatch.setattr("src.features.live_runtime.is_runtime_fresh", lambda: False)

    jm = JobManager(root=tmp_path / "jobs-live-worker-invalidate")
    with TestClient(app_mod.create_app(jobs=jm, enable_live_worker=True), base_url="http://127.0.0.1") as c:
        snap = c.get("/api/live/streamer").json()
    assert snap["status"] == "starting"
    assert snap["prediction"] is None


def test_lifespan_live_worker_terminates_on_shutdown(monkeypatch, tmp_path):
    from src.webui import app as app_mod

    class _AliveProc:
        def __init__(self):
            self.pid = 4321
            self.exitcode = None
            self.join_calls = []
            self.terminate_called = False
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

        def terminate(self) -> None:
            self.terminate_called = True
            self._alive = False

    proc = _AliveProc()

    monkeypatch.setattr(app_mod, "_LIVE_LOOP_CADENCE_S", 10.0)
    monkeypatch.setattr(app_mod, "_LIVE_LOOP_POLL_S", 0.01)
    monkeypatch.setattr(app_mod.live, "refresh_streamer_health", lambda: {"status": "ok"})
    monkeypatch.setattr(app_mod, "_spawn_live_snapshot_process", lambda _ctx: proc)

    jm = JobManager(root=tmp_path / "jobs-live-worker-stop")
    with TestClient(app_mod.create_app(jobs=jm, enable_live_worker=True), base_url="http://127.0.0.1"):
        time.sleep(0.03)

    assert proc.terminate_called is True
    assert proc.join_calls[:2] == [0.2, 0.5]


def test_lifespan_live_worker_restarts_on_refresh_request(monkeypatch, tmp_path):
    from src.webui import app as app_mod

    spawned = []

    class _AliveProc:
        def __init__(self, pid: int):
            self.pid = pid
            self.exitcode = None
            self._alive = True
            self.terminate_called = False
            self.join_calls = []

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

        def terminate(self) -> None:
            self.terminate_called = True
            self._alive = False
            self.exitcode = 0

    def _spawn(_ctx):
        proc = _AliveProc(9000 + len(spawned))
        spawned.append(proc)
        return proc

    monkeypatch.setattr(app_mod, "_LIVE_LOOP_CADENCE_S", 60.0)
    monkeypatch.setattr(app_mod, "_LIVE_LOOP_POLL_S", 0.01)
    monkeypatch.setattr(app_mod.live, "refresh_streamer_health", lambda: {"status": "ok"})
    monkeypatch.setattr(app_mod, "_spawn_live_snapshot_process", _spawn)

    jm = JobManager(root=tmp_path / "jobs-live-worker-refresh")
    with TestClient(app_mod.create_app(jobs=jm, enable_live_worker=True), base_url="http://127.0.0.1") as c:
        time.sleep(0.05)
        assert len(spawned) == 1
        r = c.post("/api/live/reload", headers=H)
        assert r.status_code == 200
        assert r.json()["snapshot_refresh"] == "scheduled"
        time.sleep(0.08)
        assert len(spawned) >= 2
        assert spawned[0].terminate_called is True


def test_refresh_streamer_health_uses_ts_event_queries(monkeypatch):
    from datetime import datetime, timedelta, timezone
    import psycopg
    from src.common import db as db_mod
    from src.webui import live as live_mod

    latest_tbbo = datetime.now(timezone.utc) - timedelta(seconds=2)
    latest_stats = datetime.now(timezone.utc) - timedelta(seconds=20)
    seen_sql: list[str] = []

    class _Result:
        def __init__(self, value):
            self._value = value

        def fetchone(self):
            return self._value if isinstance(self._value, tuple) else (self._value,)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql):
            seen_sql.append(sql)
            if "FROM databento.tbbo" in sql:
                return _Result((latest_tbbo, 11, 40))
            if "FROM databento.statistics" in sql:
                return _Result((latest_stats, 0, 0))
            raise AssertionError(f"unexpected sql: {sql}")

    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: _Conn())
    monkeypatch.setattr(db_mod, "get_dsn", lambda: "postgresql://ignored")
    monkeypatch.setattr(live_mod, "_find_streamer_pids", lambda: [])
    monkeypatch.setattr(live_mod, "_probe_systemd_unit", lambda: {
        "unit": "vtech-live-streamer.service",
        "active_state": "active",
        "sub_state": "running",
        "main_pid": 1,
        "available": True,
    })
    monkeypatch.setattr(live_mod, "_probe_connections", lambda _pid: [
        {"role": "databento_live", "state": "ESTAB"},
        {"role": "postgres", "state": "ESTAB"},
    ])
    monkeypatch.setattr(live_mod, "_probe_raw_live", lambda: {"exists": True, "files": []})
    monkeypatch.setattr(live_mod, "_probe_disk", lambda: {
        "path": "/opt/vtech/data", "free_gb": 50.0, "used_pct": 50.0, "total_gb": 100.0,
    })
    # Process presence check: systemd active but no /proc match → error path avoided
    # by providing a fake process via _find_streamer_pids + _read_proc_process.
    monkeypatch.setattr(live_mod, "_find_streamer_pids", lambda: [4242])
    monkeypatch.setattr(live_mod, "_read_proc_process", lambda _pid: {
        "pid": 4242, "cmdline": "python -m src.acquisition.live_streamer",
        "state": "S", "rss_mb": 100.0, "etime": "01:00:00", "open_raw_files": [],
    })

    out = live_mod.refresh_streamer_health()
    assert out["status"] == "ok"
    assert out["tables"]["tbbo"] == latest_tbbo.isoformat()
    assert out["tables"]["statistics"] == latest_stats.isoformat()
    assert out["tables"]["bbo_1s"] is None
    assert out["tbbo_rows_10s"] == 11
    assert out["live"] is True
    # 20s stats lag is still healthy under kind=stats (10m / 60m).
    assert out["table_detail"]["statistics"]["status"] == "healthy"
    assert out["table_detail"]["statistics"]["kind"] == "stats"
    assert out["table_detail"]["bbo_1s"]["status"] == "unknown"
    assert out["health"] == "healthy"
    assert out.get("error") in (None, "")
    assert any("max(ts_event)" in q and "databento.tbbo" in q for q in seen_sql)
    assert any("max(ts_event)" in q and "databento.statistics" in q for q in seen_sql)
    assert any("WHERE ts_event >= now()" in q and "databento.tbbo" in q for q in seen_sql)
    assert any("WHERE ts_event >= now()" in q and "databento.statistics" in q for q in seen_sql)
    assert not any("bbo_1s" in q for q in seen_sql)


def test_refresh_streamer_health_stats_soft_lag_does_not_degrade(monkeypatch):
    """CME statistics at ~30m is table-degraded but must not force overall degraded."""
    from datetime import datetime, timedelta, timezone
    import psycopg
    from src.common import db as db_mod
    from src.webui import live as live_mod

    latest_tbbo = datetime.now(timezone.utc) - timedelta(seconds=3)
    latest_stats = datetime.now(timezone.utc) - timedelta(seconds=1800)  # 30 min

    class _Result:
        def __init__(self, value):
            self._value = value

        def fetchone(self):
            return self._value if isinstance(self._value, tuple) else (self._value,)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql):
            if "FROM databento.tbbo" in sql:
                return _Result((latest_tbbo, 50, 200))
            if "FROM databento.statistics" in sql:
                return _Result((latest_stats, 0, 0))
            raise AssertionError(f"unexpected sql: {sql}")

    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: _Conn())
    monkeypatch.setattr(db_mod, "get_dsn", lambda: "postgresql://ignored")
    monkeypatch.setattr(live_mod, "_find_streamer_pids", lambda: [4242])
    monkeypatch.setattr(live_mod, "_read_proc_process", lambda _pid: {
        "pid": 4242, "cmdline": "python -m src.acquisition.live_streamer",
        "state": "S", "rss_mb": 100.0, "etime": "01:00:00", "open_raw_files": [],
    })
    monkeypatch.setattr(live_mod, "_probe_systemd_unit", lambda: {
        "unit": "vtech-live-streamer.service",
        "active_state": "active",
        "sub_state": "running",
        "main_pid": 4242,
        "available": True,
    })
    monkeypatch.setattr(live_mod, "_probe_connections", lambda _pid: [
        {"role": "databento_live", "state": "ESTAB"},
        {"role": "postgres", "state": "ESTAB"},
    ])
    monkeypatch.setattr(live_mod, "_probe_raw_live", lambda: {"exists": True, "files": []})
    monkeypatch.setattr(live_mod, "_probe_disk", lambda: {
        "path": "/opt/vtech/data", "free_gb": 50.0, "used_pct": 50.0, "total_gb": 100.0,
    })

    out = live_mod.refresh_streamer_health()
    assert out["live"] is True
    assert out["table_detail"]["tbbo"]["status"] == "healthy"
    assert out["table_detail"]["statistics"]["status"] == "degraded"
    assert out["table_detail"]["statistics"]["kind"] == "stats"
    assert out["health"] == "healthy"
    assert any("statistics=degraded" in n for n in out.get("notes", []))
    assert out.get("error") in (None, "")


def test_refresh_streamer_health_caches_errors(monkeypatch):
    import psycopg
    from src.common import db as db_mod
    from src.webui import live as live_mod

    def _raise_connect(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(psycopg, "connect", _raise_connect)
    monkeypatch.setattr(db_mod, "get_dsn", lambda: "postgresql://ignored")
    out = live_mod.refresh_streamer_health()
    assert out["status"] == "error"
    assert out["live"] is False
    assert "RuntimeError: db down" in out["error"]
    assert "RuntimeError: db down" in live_mod.streamer_health()["error"]


def test_tick_ledger_is_cache_only_on_request_path(monkeypatch):
    """HTTP ticks blotter must not wait on the full-day paper replay."""
    import threading
    from src.webui import live as live_mod

    started = threading.Event()
    released = threading.Event()

    def _slow(**_kwargs):
        started.set()
        assert released.wait(timeout=2.0)
        return [{"ts": "2026-08-18T13:30:00+00:00", "dir": "up", "sig": True, "conf": 0.9}]

    monkeypatch.setattr(live_mod, "_build_ledger_live_ticks", _slow)
    live_mod._STATE.update({
        "status": "ok",
        "objective_id": "nq-50pt-econ",
        "model": "xgb",
        "model_path": "/tmp/model",
        "tau": 0.8,
        "session": {"date": "2026-08-18"},
        "barriers": {},
    })

    t0 = time.time()
    page = live_mod.ledger(mode="ticks", tau=0.8)
    assert time.time() - t0 < 0.25
    assert page["meta"]["building"] is True
    assert page["rows"] == []
    assert started.wait(timeout=1.0)

    released.set()
    page2 = None
    deadline = time.time() + 2.0
    while time.time() < deadline:
        page2 = live_mod.ledger(mode="ticks", tau=0.8)
        if not page2["meta"].get("building"):
            break
        time.sleep(0.02)
    assert page2 is not None
    assert page2["meta"]["building"] is False
    assert len(page2["rows"]) == 1
    assert page2["meta"]["source"] == "tbbo_ticks_cache"


def test_tick_ledger_cache_survives_snapshot_replace(monkeypatch):
    from src.webui import live as live_mod

    monkeypatch.setattr(
        live_mod, "_build_ledger_live_ticks",
        lambda **_k: [{"ts": "t", "dir": "down", "sig": True}],
    )
    live_mod._STATE.update({
        "status": "ok",
        "objective_id": "nq-50pt-econ",
        "model": "xgb",
        "model_path": "/tmp/model",
        "tau": 0.8,
        "session": {"date": "2026-08-18"},
        "barriers": {},
    })
    live_mod.ledger(mode="ticks", tau=0.8)
    deadline = time.time() + 2.0
    while time.time() < deadline and live_mod._TICKS_CACHE.get("building"):
        time.sleep(0.02)
    live_mod._replace_state({
        "status": "ok",
        "prediction": None,
        "model_path": "/tmp/model",
        "objective_id": "nq-50pt-econ",
        "tau": 0.8,
        "session": {"date": "2026-08-18"},
    })
    page = live_mod.ledger(mode="ticks", tau=0.8)
    assert page["meta"]["building"] is False
    assert len(page["rows"]) == 1


# ── custom test ───────────────────────────────────────────────────────────────

def test_custom_test_bank_and_default(client):
    bank = client.get("/api/custom-test/bank").json()
    assert bank.get("store") in (None, "5s") or bank["store"] == "5s"
    assert any(m["name"] == "xgboost" for m in bank["models"])
    assert any(v["name"] == "regime_core" for v in bank["views"])
    core = next(v for v in bank["views"] if v["name"] == "regime_core")
    assert core["source"] == "catalog"
    assert core["n_features"] >= 10
    assert bank["feature_builder"]["available"] is False
    xgb = next(m for m in bank["models"] if m["name"] == "xgboost")
    assert xgb["params_default"]["max_depth"] == 4
    assert "n_estimators" in xgb["search_space"]
    assert xgb["hpo_available"] is True
    lstm = next(m for m in bank["models"] if m["name"] == "attention_lstm")
    assert lstm["search_max_rows"] == 100_000
    assert "n_features" not in lstm["params_default"]
    d = client.get("/api/custom-test/default/nq-50pt-xgb-tcn").json()
    assert [m["family"] for m in d["models"]] == ["xgboost", "tcn"]
    assert d["meta"]["enabled"] is True
    assert d["meta"]["mode"] == "meta_stack"
    b15 = client.get("/api/custom-test/bank", params={"store": "15s"}).json()
    names = {f["name"] for f in b15.get("features") or []}
    assert "of_ofi" in names
    assert "bp_spread_mean" not in names
    dlg = client.get("/api/custom-test/default/nq-50pt-econ-lg-st").json()
    assert dlg["feature_store"] == "15s"
    assert dlg["head"] == "dual_binary"


def test_custom_test_preview_and_launch(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "custom_test_root",
                        lambda oid: tmp_path / "custom_tests" / oid)
    monkeypatch.setattr(ds, "build_argv_custom_test",
                        lambda oid, spec, topo, dry, **kw: ["/bin/echo", "custom-test", oid])
    board = {
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{
            "family": "xgboost",
            "items": [
                {"kind": "view", "key": "regime_core", "ablation": False},
                {"kind": "group", "key": "iv_", "ablation": True},
            ],
        }],
        "meta": {"enabled": True, "mode": "meta_stack"},
        "dry_run": True,
    }
    prev = client.post("/api/custom-test/preview", headers=H, json=board).json()
    assert prev["n_candidates"] == 2
    r = client.post("/api/custom-test/launch", headers=H, json=board)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["job"]["kind"] == "custom_test"
    assert body["compiled"]["n_candidates"] == 2
    assert body["compiled"]["topology"]["mode"] == "meta_stack"


def test_custom_test_preview_params_hpo(client):
    board = {
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{
            "family": "xgboost",
            "params": {"max_depth": 6, "colsample_bytree": 1},
            "hpo": True,
            "hpo_params": ["max_depth", "learning_rate"],
            "items": [{"kind": "view", "key": "regime_core", "ablation": False}],
        }],
        "meta": {"enabled": False},
        "tune_trials": 12,
        "dry_run": True,
    }
    prev = client.post("/api/custom-test/preview", headers=H, json=board).json()
    assert prev["tune_trials"] == 12
    cand = prev["candidates"][0]
    assert cand["params"]["max_depth"] == 6
    assert cand["params"]["colsample_bytree"] == 1.0
    assert set(cand["search_space"]) == {"max_depth", "learning_rate"}
    spec_cand = prev["manual_spec"]["candidates"][0]
    assert spec_cand["params"]["max_depth"] == 6
    assert set(spec_cand["search_space"]) == {"max_depth", "learning_rate"}


def test_custom_test_preview_lstm_train_cap(client):
    board = {
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "attention_lstm",
                    "items": [{"kind": "group", "key": "of_"}]}],
        "meta": {"enabled": False},
        "dry_run": True,
    }
    prev = client.post("/api/custom-test/preview", headers=H, json=board).json()
    assert prev["candidates"][0].get("max_rows") is None
    on = {**board, "models": [{**board["models"][0], "max_rows": 100_000}]}
    prev_on = client.post("/api/custom-test/preview", headers=H, json=on).json()
    assert prev_on["manual_spec"]["candidates"][0].get("max_rows") == 100_000


def test_custom_test_preview_walk_forward(client):
    board = {
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "momentum_edge"}]}],
        "meta": {"enabled": False},
        "walk_forward": {
            "enabled": True,
            "start": "2025-05-01", "stop": "2026-06-30",
            "train_n": 2, "train_unit": "months",
            "test_n": 2, "test_unit": "months",
            "val": {"enabled": False, "n": 10, "unit": "days", "cut_from": "train"},
        },
        "dry_run": True,
    }
    prev = client.post("/api/custom-test/preview", headers=H, json=board).json()
    wf = prev["walk_forward"]
    assert wf["walk_forward_folds"] == 6
    assert wf["walk_forward"]["mode"] == "expanding_calendar"
    assert wf["walk_forward"]["schedule"][0]["train"]["end"] == "2025-06-30"
    assert wf["walk_forward"]["schedule"][-1]["oof"]["end"] == "2026-06-30"


def test_custom_test_saved_view_api(client):
    r = client.post("/api/custom-test/views", headers=H, json={
        "name": "My LSTM Set",
        "features": ["vw_vwap_dist", "bp_spread_mean"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["view"]["name"] == "my_lstm_set"
    assert body["view"]["features"] == ["vw_vwap_dist", "bp_spread_mean"]
    assert any(v["name"] == "my_lstm_set" and v["source"] == "saved"
               for v in body["views"])
    bank = client.get("/api/custom-test/bank").json()
    assert any(v["name"] == "my_lstm_set" for v in bank["views"])
    clash = client.post("/api/custom-test/views", headers=H, json={
        "name": "my_lstm_set",
        "features": ["vw_vwap_dist"],
    })
    assert clash.status_code == 409
    over = client.post("/api/custom-test/views", headers=H, json={
        "name": "my_lstm_set",
        "features": ["vw_vwap_dist"],
        "overwrite": True,
    })
    assert over.status_code == 200
    assert over.json()["view"]["features"] == ["vw_vwap_dist"]
    reserved = client.post("/api/custom-test/views", headers=H, json={
        "name": "regime_core",
        "features": ["vw_vwap_dist"],
    })
    assert reserved.status_code == 400
    prev = client.post("/api/custom-test/preview", headers=H, json={
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "my_lstm_set"}]}],
        "meta": {"enabled": False},
        "dry_run": True,
    }).json()
    assert prev["candidates"][0]["features"] == {"columns": ["vw_vwap_dist"]}
    gone = client.delete("/api/custom-test/views/my_lstm_set", headers=H)
    assert gone.status_code == 200
    assert all(v["name"] != "my_lstm_set" for v in gone.json()["views"])


def test_custom_test_js_save_as_view_and_expand():
    app_js = (REPO / "src" / "webui" / "static" / "app.js").read_text()
    start = app_js.index("async function viewCustomTest(")
    end = app_js.index("/* ── Live Chart")
    ct = app_js[start:end]
    assert "save as view" in ct
    assert "window._ctSaveAs" in ct
    assert "window._ctSaveView" in ct
    assert "window._ctDelView" in ct
    assert "const expandItems = items =>" in ct
    assert "const columnsFor = (kind, key) =>" in ct
    assert "items: expandItems(m.items || [])" in ct
    assert "/api/custom-test/views" in ct
    assert "views and groups expand to" in ct


def test_custom_test_unknown_family_400(client):
    r = client.post("/api/custom-test/preview", headers=H, json={
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "not_a_model"}],
    })
    assert r.status_code == 400


def test_job_env_prepends_cwd(tmp_path, monkeypatch):
    import os
    from src.webui.jobs import job_env

    monkeypatch.setenv("PYTHONPATH", "/other")
    env = job_env(tmp_path, {"FOO": "1"})
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)
    assert "/other" in env["PYTHONPATH"].split(os.pathsep)
    assert env["FOO"] == "1"


def test_default_job_cwd_worktree_uses_host(tmp_path, monkeypatch):
    from src.webui import jobs as jmod

    fake_wt = tmp_path / ".grok" / "worktrees" / "opt-vtech" / "webui-check"
    fake_wt.mkdir(parents=True)
    monkeypatch.setattr(jmod, "_REPO_ROOT", fake_wt)
    monkeypatch.delenv("VTECH_REPO_ROOT", raising=False)
    assert jmod.default_job_cwd() == Path("/opt/vtech").resolve()


def test_default_job_cwd_env_override(tmp_path, monkeypatch):
    from src.webui import jobs as jmod

    (tmp_path / "src" / "search").mkdir(parents=True)
    (tmp_path / "src" / "search" / "controller.py").write_text("#")
    monkeypatch.setenv("VTECH_REPO_ROOT", str(tmp_path))
    assert jmod.default_job_cwd() == tmp_path.resolve()


def test_build_argv_custom_test_path_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "custom_test_root",
                        lambda oid: tmp_path / "custom_tests" / oid)
    dest = tmp_path / "custom_tests" / "nq-50pt-xgb-tcn"
    dest.mkdir(parents=True)
    spec = dest / "x.manual.yaml"
    spec.write_text("candidates: [{family: xgboost}]\n")
    topo = dest / "x.topology.yaml"
    topo.write_text("mode: none\n")
    argv = ds.build_argv_custom_test("nq-50pt-xgb-tcn", spec, topo, True)
    assert "--manual-spec" in argv and "--topology-overlay" in argv and "--dry-run" in argv
    assert argv[argv.index("--experiment-type") + 1] == "custom_test"
    assert "--tune-trials" not in argv
    assert "--walk-forward-overlay" not in argv
    tuned = ds.build_argv_custom_test(
        "nq-50pt-xgb-tcn", spec, topo, False, tune_trials=12)
    assert tuned[tuned.index("--tune-trials") + 1] == "12"
    wf = dest / "x.walk_forward.yaml"
    wf.write_text("date_range: {start: '2025-05-01', end: '2026-06-30'}\n")
    with_wf = ds.build_argv_custom_test(
        "nq-50pt-xgb-tcn", spec, topo, True, walk_forward_overlay=wf)
    assert with_wf[with_wf.index("--walk-forward-overlay") + 1] == str(wf.resolve())
    with pytest.raises(ValueError):
        ds.build_argv_custom_test("nq-50pt-xgb-tcn", "/etc/passwd", None, False)
