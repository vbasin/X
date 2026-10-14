"""Tests for src.experiment.cli -- exercises each subcommand via build_parser()/
main(), never via subprocess (keeps this fast and free of shell/DB dependence).
"""
import yaml

from src.experiment.cli import main
from src.experiment.manifest import RunManifest


def test_status_runs_and_returns_zero(capsys):
    rc = main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "responsibilities:" in out
    assert "tracks:" in out


def test_validate_config_against_a_clean_fixture(tmp_path, capsys):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"objective_id": "x"}))
    rc = main(["validate-config", "--path", str(p)])
    assert rc == 0
    assert "no findings" in capsys.readouterr().out


def test_validate_config_nonzero_exit_on_error(tmp_path, capsys):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({}))  # missing objective_id
    rc = main(["validate-config", "--path", str(p)])
    assert rc == 1
    assert "ERROR" in capsys.readouterr().out


def test_validate_config_also_checks_objective_doc(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"objective_id": "x"}))
    obj = tmp_path / "obj.yaml"
    obj.write_text(yaml.safe_dump({"costs": {"slippage_model": "fixed_1pt"}}))
    rc = main(["validate-config", "--path", str(cfg), "--objective", str(obj)])
    assert rc == 1  # the objective doc has the slippage bug
    out = capsys.readouterr().out
    assert "slippage_points" in out


def _m(run_id, **kw):
    """A FULLY-evidenced manifest.

    validation/seeds/execution_assumptions are populated deliberately: the
    comparison module treats an unrecorded dimension as missing evidence
    rather than as agreement, so a partially-filled fixture can never produce
    a controlled comparison.
    """
    from src.experiment.manifest import (
        CodeIdentity, DatasetRef, ValidationDesign,
    )
    base = dict(
        run_id=run_id, objective_id="nq-50pt-econ", hypothesis="h",
        responsibility_versions={"feature_builder": "x@1"},
        code=CodeIdentity(git_sha="abc", dirty=False),
        datasets=(DatasetRef(logical_id="labels_15s", contract_id="c1"),),
        validation=ValidationDesign(split="expanding_equal_count", n_folds=3),
        seeds={"numpy": 42},
        execution_assumptions={"slippage_points_per_side": 1.0},
        config_snapshot={"labels": {"move_threshold": 50}},
        decision_policy={"gate": "p_move>=0.95"},
        calibration={"method": "none"},
    )
    base.update(kw)
    return RunManifest(**base)


def test_run_inspect_prints_manifest_json(tmp_path, capsys):
    _m("run-1").write(tmp_path)
    rc = main(["run-inspect", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "run-1" in out
    assert "scientific_fingerprint" in out


def test_run_compare_reports_controlled_comparison(tmp_path, capsys):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _m("a").write(dir_a)
    _m("b", responsibility_versions={"feature_builder": "x@2"}).write(dir_b)
    rc = main(["run-compare", str(dir_a), str(dir_b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "controlled_comparison: True" in out
    assert "single varying dimension: responsibility_versions" in out


def test_run_compare_surfaces_missing_evidence(tmp_path, capsys):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _m("a", datasets=()).write(dir_a)
    _m("b", datasets=(), responsibility_versions={"feature_builder": "x@2"}).write(dir_b)
    rc = main(["run-compare", str(dir_a), str(dir_b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "controlled_comparison: False" in out
    assert "missing evidence" in out


def test_resolve_run_rejects_unknown_identifier():
    import pytest
    from src.experiment.cli import resolve_run
    with pytest.raises(FileNotFoundError, match="no run directory found"):
        resolve_run("definitely-not-a-real-run-id-xyz")


def test_verify_actually_invokes_pytest_and_defaults_to_canonical_tests(monkeypatch, capsys):
    """`verify` must RUN the suite, not print a command (see cli._cmd_verify)."""
    import subprocess

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = main(["verify"])
    assert rc == 0
    assert "pytest" in seen["cmd"]
    assert "tests/canonical/" in seen["cmd"]


def test_verify_propagates_nonzero_exit_code(monkeypatch, capsys):
    """A verification command that cannot fail is not a verification command."""
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = main(["verify"])
    assert rc == 3
    assert "FAILED" in capsys.readouterr().err
