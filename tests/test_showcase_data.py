"""Scoped portable evidence tests; all generated state lives under tmp_path."""
from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.showcase_data import (
    example_root,
    load_example,
    prepare_example,
    relocate_prepared,
    tracking_uri,
    verify_example,
)

pytestmark = pytest.mark.showcase
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/transformer-regime"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorded_population_and_real_evaluation() -> None:
    view = load_example(EXAMPLE)
    assert len(view["data"]["rows"]) == 3000
    assert len(view["data"]["features"]) == 5
    assert not set(view["data"]["features"]) & set(view["data"]["outcomes"])
    assert view["summary"]["dataset_sha256"] == "150a89b38646ce3e2d46baf55671ba75faf1e05488ac46faa67fcf536a0a47e8"
    assert [len(rows) for rows in view["predictions"]["validation"].values()] == [435, 435]
    holdout = view["predictions"]["holdout"]
    assert len(holdout) == 289
    assert holdout[0]["ts"] == "2024-01-04T11:41:00.000Z"
    accuracy = sum(row["direction"] == row["direction_true"] for row in holdout) / len(holdout)
    assert accuracy == pytest.approx(view["summary"]["holdout_metrics"]["accuracy"])
    assert accuracy == pytest.approx(0.9342560553633218)
    assert view["evaluation"]["gates"]["holdout"]["passed"]
    assert len(view["evaluation"]["gates"]["holdout"]["checks"]) == 8
    assert not any(control["passed"] for control in view["evaluation"]["negative_controls"].values())
    winner = view["summary"]["selected_model_id"]
    ledger = view["backtest"][winner]
    assert len(ledger["trades"]) == 70
    assert ledger["trades"][-1]["cumulative_net_dollars"] == pytest.approx(ledger["economics"]["net_dollars"])
    assert view["summary"]["trading_activation"] is False


def test_viewer_does_not_import_training_or_operational_database(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"tensorflow", "psycopg", "psycopg2", "mlflow"}:
            raise AssertionError(f"Viewer imported prohibited dependency {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert load_example(EXAMPLE)["summary"]["n_rows"] == 3000


def test_tampered_and_unlisted_artifacts_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(EXAMPLE, fixture)
    model = fixture / "models/d7facd489218/model.keras"
    payload = bytearray(model.read_bytes())
    payload[100] ^= 1
    model.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        verify_example(fixture)
    shutil.copyfile(EXAMPLE / model.relative_to(fixture), model)
    (fixture / "unreviewed-secret.txt").write_text("must not be published")
    with pytest.raises(ValueError, match="allowlist"):
        verify_example(fixture)


@pytest.mark.parametrize("unsafe", ["../outside", "/tmp/outside", "models/../../outside", "models\\outside"])
def test_manifest_cannot_escape_fixture(tmp_path: Path, unsafe: str) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(EXAMPLE, fixture)
    manifest_path = fixture / "showcase_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["path"] = unsafe
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        verify_example(fixture)


def test_projection_omissions_are_documented_and_no_operator_paths() -> None:
    manifest = verify_example(EXAMPLE)
    projections = [a for a in manifest["artifacts"] if a["kind"] == "projection"]
    assert projections
    summary = next(a for a in projections if a["path"] == "evidence/summary.json")
    assert "tracking_uri" in summary["omitted_fields"]
    assert "run_directory" in summary["omitted_fields"]
    for artifact in projections:
        assert len(artifact["source_sha256"]) == 64
    for path in EXAMPLE.rglob("*.json"):
        assert "/opt/vtech/" not in path.read_text()
    assert not manifest["original_code_bundle_included"]
    assert not manifest["original_deployment_package_included"]


def test_required_consumed_file_must_be_listed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(EXAMPLE, fixture)
    path = fixture / "showcase_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"] = [a for a in manifest["artifacts"] if a["path"] != "evidence/summary.json"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allowlist|Required"):
        load_example(fixture)


def test_prepared_pointer_cannot_escape_output(tmp_path: Path) -> None:
    (tmp_path / "showcase.json").write_text(json.dumps({"example_root": "../outside"}))
    with pytest.raises(ValueError, match="Unsafe"):
        example_root(tmp_path)


def test_mlflow_import_preserves_history_and_relocates(tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    before = {p.relative_to(EXAMPLE).as_posix(): _digest(p) for p in EXAMPLE.rglob("*") if p.is_file()}
    output = tmp_path / "prepared"
    state = prepare_example(EXAMPLE, output)
    assert not Path(state["example_root"]).is_absolute()
    assert not Path(state["mlflow"]["tracking_db"]).is_absolute()
    assert "research/runs/synthetic-regime-service/" in state["example_root"]
    client = MlflowClient(tracking_uri=tracking_uri(output))
    mapping = state["mlflow"]["run_id_map"]
    export = json.loads((EXAMPLE / "mlflow_export.json").read_text())
    for source in export["runs"]:
        rid = mapping[source["source_run_id"]]
        assert rid != source["source_run_id"]
        run = client.get_run(rid)
        assert run.data.tags["source_run_id"] == source["source_run_id"]
        assert run.data.tags["source_start_time"] == str(source["source_start_time"])
        assert run.data.tags["evidence_kind"] == "recorded_example"
        assert run.info.status == "FINISHED"
        assert run.info.artifact_uri.startswith(output.as_uri())
        if "mlflow.parentRunId" in source["tags"]:
            assert run.data.tags["mlflow.parentRunId"] == mapping[export["parent_run_id"]]
        for key, history in source["metrics"].items():
            actual = client.get_metric_history(rid, key)
            assert [(p.value, p.timestamp, p.step) for p in actual] == [
                (p["value"], p["timestamp"], p["step"]) for p in history]
    parent = mapping[export["parent_run_id"]]
    downloaded = client.download_artifacts(parent, "recorded_example/synthetic.parquet")
    assert _digest(Path(downloaded)) == verify_example(EXAMPLE)["dataset_sha256"]
    moved = tmp_path / "different-location" / "prepared"
    shutil.copytree(output, moved)
    relocated = relocate_prepared(moved)
    assert relocated["mlflow"]["run_id_map"] != mapping
    assert list(moved.glob("showcase.previous-*.json"))
    relocated_client = MlflowClient(tracking_uri=tracking_uri(moved))
    relocated_id = relocated["mlflow"]["parent_run_id"]
    assert relocated_client.get_run(relocated_id).info.artifact_uri.startswith(moved.as_uri())
    assert _digest(Path(relocated_client.download_artifacts(relocated_id, "recorded_example/synthetic.parquet"))) == _digest(EXAMPLE / "synthetic.parquet")
    assert load_example(moved)["summary"]["source_run_id"] == "20261015T032845Z_c77c70f8"
    assert before == {p.relative_to(EXAMPLE).as_posix(): _digest(p) for p in EXAMPLE.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        prepare_example(EXAMPLE, output)


def test_preparation_cannot_write_inside_source_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(EXAMPLE, fixture)
    output = fixture / "generated"
    with pytest.raises(ValueError, match="outside.*source fixture"):
        prepare_example(fixture, output)
    assert not output.exists()
    verify_example(fixture)


def test_export_cannot_write_inside_original_run(tmp_path: Path) -> None:
    from scripts.showcase.export_transformer import export_example

    source = tmp_path / "source"
    source.mkdir()
    (source / "sentinel.txt").write_text("immutable")
    with pytest.raises(ValueError, match="outside.*source run"):
        export_example(source, source / "export")
    assert sorted(p.name for p in source.iterdir()) == ["sentinel.txt"]


def test_nested_manifest_is_not_exempt_from_allowlist(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(EXAMPLE, fixture)
    (fixture / "models/d7facd489218/showcase_manifest.json").write_text('{"private":true}')
    with pytest.raises(ValueError, match="allowlist"):
        verify_example(fixture)


def test_unlisted_directory_symlink_is_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(EXAMPLE, fixture)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("must never be imported")
    (fixture / "models/d7facd489218/unlisted").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        verify_example(fixture)
