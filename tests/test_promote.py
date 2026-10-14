"""Registry promote + LATEST pointer (objective-keyed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.serving.promote import load_latest, promote, resolve_latest_bundle


def _fake_bundle(d: Path, *, objective_id: str = "t-obj", version: str = "xgb_v1") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "feature_schema.json").write_text(json.dumps({
        "feature_cols": ["a", "b"], "n_features": 2,
    }))
    (d / "model.ubj").write_bytes(b"fake")
    (d / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": objective_id,
        "candidate_id": "xgb",
        "family": "xgboost",
        "version": version,
        "search_run_ts": "20990101T000000Z",
        "search_score": {"precision_at_tau": 0.9},
    }))
    return d


def test_promote_writes_latest_and_bundle(tmp_path):
    stage = _fake_bundle(tmp_path / "staging" / "t-obj" / "xgb_v1", version="xgb_v1")
    reg = tmp_path / "registry"
    pointer = promote(stage, objective_id="t-obj", registry_root=reg, force=True)
    assert pointer["objective_id"] == "t-obj"
    assert pointer["version"] == "xgb_v1"
    assert (reg / "t-obj" / "LATEST.json").exists()
    dest = reg / "t-obj" / "xgb_v1"
    assert (dest / "model.ubj").exists()
    assert (dest / "feature_schema.json").exists()
    assert (dest / "_SUCCESS").exists()
    latest = load_latest("t-obj", registry_root=reg)
    assert latest is not None and latest["version"] == "xgb_v1"
    resolved = resolve_latest_bundle("t-obj", registry_root=reg)
    assert resolved is not None and resolved == dest.resolve()


def test_rd_resolve_baseline_registry_only(tmp_path, monkeypatch):
    from src.research_director import tools as T

    stage = _fake_bundle(tmp_path / "staging" / "t-obj" / "xgb_v1")
    reg = tmp_path / "registry"
    promote(stage, objective_id="t-obj", registry_root=reg, force=True)

    # staging-only would exist but must not be used when registry empty for other oid
    b = T.resolve_baseline(reg, "t-obj", "", objective_id="t-obj")
    assert b is not None
    assert b["source"] == "registry"
    assert b["version"] == "xgb_v1"

    cold = T.resolve_baseline(reg, "other-obj", "", objective_id="other-obj")
    assert cold is None


def test_rd_enable_flags(tmp_path, monkeypatch):
    from src.research_director import rd_enable as E
    from src.serving.promote import promote

    monkeypatch.setattr(E, "webui_dir", lambda: tmp_path / "webui")
    stage = _fake_bundle(tmp_path / "stg" / "o1" / "v1", objective_id="o1", version="v1")
    reg = tmp_path / "reg"
    promote(stage, objective_id="o1", registry_root=reg, force=True)
    monkeypatch.setattr(
        "src.serving.promote.model_registry_dir", lambda: reg)
    monkeypatch.setattr(E, "model_registry_dir", lambda: reg)

    # default: LATEST exists → enabled
    assert E.is_rd_enabled("o1", has_latest=True) is True
    E.set_rd_enabled("o1", False)
    assert E.is_rd_enabled("o1", has_latest=True) is False
    E.set_rd_enabled("o1", True)
    targets = E.list_rd_targets()
    assert any(t["objective_id"] == "o1" for t in targets)
