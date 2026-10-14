"""Track B and Track C run through the SAME canonical architecture.

The point of the whole restructuring: one layout, one manifest, one model
metadata file, one artifact registry, one run reader -- for a Track B
multi-model Keras/XGBoost stack with a meta layer AND for a Track C two-stage
cascade. These tests take every migrated/emitted run on disk and assert the same
invariants of all of them, without knowing which track produced which.

DB-free: reads manifests and parquet schemas only. Skips cleanly on a machine
where no run has been emitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiment.artifacts import (
    classify_existing,
    validate_model_inputs,
)
from src.experiment.manifest import SCHEMA_VERSION, read_run_record
from src.common.data_paths import runs_dir


def _canonical_runs() -> list[Path]:
    root = runs_dir()
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.glob("*/*/manifest.json"))


_RUNS = _canonical_runs()
_IDS = [f"{p.parent.name}/{p.name}" for p in _RUNS]

# Research-protocol runs (cadence / noise / selective / profile-reclaim) carry a
# protocol.json and record discovery/validation evidence, not trained-model
# passes. The model-run invariants below (passes, model.json join specs, the
# classified-artifact inventory) do not apply to them; they get their own
# manifest-level checks in test_protocol_run_has_one_manifest_and_no_legacy_files.
_PROTOCOL_RUNS = [r for r in _RUNS if (r / "protocol.json").is_file()]
_PROTOCOL_IDS = [f"{p.parent.name}/{p.name}" for p in _PROTOCOL_RUNS]
_MODEL_RUNS = [r for r in _RUNS if r not in _PROTOCOL_RUNS]
_MODEL_IDS = [f"{p.parent.name}/{p.name}" for p in _MODEL_RUNS]

if not _RUNS:  # pragma: no cover - environment without emitted runs
    pytestmark = pytest.mark.skip(reason="no canonical run emitted on this machine")


def test_at_least_one_run_from_each_track_is_present():
    objectives = {p.parent.name for p in _RUNS}
    assert any(o.startswith("nq-Rscaled") for o in objectives), (
        f"no Track C cascade run among {sorted(objectives)}")
    assert any(o.startswith("nq-50pt") for o in objectives), (
        f"no Track B custom_test run among {sorted(objectives)}")


@pytest.mark.parametrize("run", _MODEL_RUNS, ids=_MODEL_IDS)
def test_one_reader_and_one_manifest_for_every_run(run: Path):
    record = read_run_record(run)
    assert record["objective_id"]
    assert record["passes"], "a canonical run records its passes"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["run_id"] == f"{run.parent.name}/{run.name}"
    # manifest.json is the only run-level metadata file.
    for legacy in ("run.json", "contract.json", "folds.json", "search_result.json"):
        assert not (run / legacy).exists(), f"{run} wrote legacy {legacy}"


@pytest.mark.parametrize("run", _PROTOCOL_RUNS, ids=_PROTOCOL_IDS)
def test_protocol_run_has_one_manifest_and_no_legacy_files(run: Path):
    record = read_run_record(run)
    assert record["objective_id"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["run_id"] == f"{run.parent.name}/{run.name}"
    for legacy in ("run.json", "contract.json", "folds.json", "search_result.json"):
        assert not (run / legacy).exists(), f"{run} wrote legacy {legacy}"


@pytest.mark.parametrize("run", _MODEL_RUNS, ids=_MODEL_IDS)
def test_every_model_has_a_usable_join_spec_and_a_compact_oof(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    seen = 0
    for entry in manifest["passes"]:
        for model in entry.get("models") or []:
            folder = run / f"pass_{entry['pass']}" / f"model_{model['model_id']}"
            meta = json.loads((folder / "model.json").read_text())
            for field in ("model_id", "model_type", "hyperparameters", "target",
                          "feature_columns", "inputs"):
                assert field in meta, f"{folder} model.json lacks {field}"
            assert validate_model_inputs(meta["inputs"]) == [], (
                f"{folder} has an unusable join spec")
            oof = folder / "oof_predictions.parquet"
            if oof.is_file():
                import pyarrow.parquet as pq
                columns = list(pq.read_schema(oof).names)
                assert sorted(columns) == sorted(meta["oof_columns"]), (
                    f"{folder}: OOF columns disagree with model.json")
                overlap = set(columns) & set(meta["feature_columns"])
                assert not overlap, (
                    f"{folder}: feature columns leaked into the compact OOF: "
                    f"{sorted(overlap)[:5]}")
            seen += 1
    assert seen, f"{run} records no model"


@pytest.mark.parametrize("run", _MODEL_RUNS, ids=_MODEL_IDS)
def test_downstream_passes_name_the_upstream_model_they_consume(run: Path):
    """The multi-pass rule, identical for a Track C cascade and a Track B meta."""
    manifest = json.loads((run / "manifest.json").read_text())
    passes = {int(p["pass"]): p for p in manifest["passes"]}
    if max(passes) < 2:
        pytest.skip("single-pass run")
    # A meta-only run has no pass 1 of its own: its bases live in other runs and
    # are addressed by objective/run_ts. That is a real topology, not a defect.
    upstream_ids = {m["model_id"] for m in (passes.get(1) or {}).get("models") or []}
    for n in sorted(passes):
        if n == 1:
            continue
        for model in passes[n].get("models") or []:
            folder = run / f"pass_{n}" / f"model_{model['model_id']}"
            meta = json.loads((folder / "model.json").read_text())
            predictions = meta["inputs"].get("predictions") or []
            assert predictions, f"{folder} names no upstream predictions"
            for ref in predictions:
                assert ref["semantic_type"] == "predictions"
                if ref.get("upstream_run_id"):
                    # A meta layer whose base lives in another run: the pointer
                    # is objective/run_ts-relative, so resolve from the root.
                    target = run.parent.parent / ref["location"]
                    assert ref["upstream_run_id"] in str(ref["location"]), ref
                else:
                    assert ref["model_id"] in upstream_ids, (
                        f"{folder} consumes unknown upstream {ref['model_id']!r}")
                    target = run / ref["location"]
                assert target.is_file(), (
                    f"{folder} points at a missing upstream OOF {ref['location']}")
                # Identity, not a path: the upstream fit's configuration hash.
                assert ref.get("build_id"), f"{folder} upstream lacks build identity"
    # The manifest's own dataset inventory carries those predictions too.
    kinds = {d.get("semantic_type") for d in manifest["datasets"]}
    assert "predictions" in kinds, f"{run} manifest omits its prediction inputs"


@pytest.mark.parametrize("run", _MODEL_RUNS, ids=_MODEL_IDS)
def test_no_file_in_a_canonical_run_is_unclassified(run: Path):
    """Every file a canonical run contains is a known artifact class -- the
    property that was false for every model weight file before this pass."""
    unknown: list[str] = []
    for path in run.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(run))
        if classify_existing(rel) == ("unclassified", None):
            unknown.append(rel)
    assert unknown == [], f"{run} holds unclassified files: {unknown}"


@pytest.mark.parametrize("run", _RUNS, ids=_IDS)
def test_evidence_is_reported_truthfully_and_gaps_are_explained(run: Path):
    """A gap must be reported, and a reported gap must say why it is absent --
    never silently filled with a reconstructed value."""
    manifest = json.loads((run / "manifest.json").read_text())
    missing = set(manifest["missing_evidence"])
    if not missing:
        assert manifest["evidence_level"] != "INCOMPLETE"
        return
    notes = (manifest.get("extra") or {}).get("evidence") or {}
    unexplained = sorted(missing - set(notes) - {"hypothesis"})
    assert unexplained == [], (
        f"{run} reports {unexplained} as missing with no recorded reason; "
        f"record why in extra.evidence rather than fabricating a value")
