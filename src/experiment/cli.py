"""Canonical command surface: ``python -m src.experiment.cli <subcommand>``.

Deliberately small (per docs/RESEARCH_WORKFLOW.md / ADR-136's
avoid-over-engineering principle): status inspection, config validation, run
inspection/comparison, and a pointer to the test-based verification loop.
It does NOT wrap or reimplement any Track B/C training/evaluation entry
point -- those remain the existing scripts (docs/RESEARCH_WORKFLOW.md
section 1's table, docs/SCRIPT_INVENTORY.md). ``status`` prints that
mapping rather than pretending a canonical training command exists.

No subcommand here touches a database or mutates any file under ``data/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.experiment.comparison import compare_results, compare_runs
from src.experiment.config_validation import (
    validate_config_yaml,
    validate_objective_yaml,
)
from src.experiment.manifest import RunManifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cmd_status(_args: argparse.Namespace) -> int:
    manifest_path = REPO_ROOT / "project_manifest.yaml"
    if not manifest_path.exists():
        print(f"project_manifest.yaml not found at {manifest_path}", file=sys.stderr)
        return 1
    data = yaml.safe_load(manifest_path.read_text())
    print(f"canonical docs: {manifest_path}")
    print(f"\narchitecture model: {data.get('architecture_model', 'unknown')}")
    print("\nresponsibilities:")
    for r in data.get("responsibilities", []):
        # plane defaults to research: see project_manifest.yaml's `planes:` note
        print(f"  [{r.get('plane', 'research'):<8}] {r['id']:<24} {r['status']}")
    print("\ntracks:")
    for name, info in data.get("tracks", {}).items():
        print(f"  {name}: {info.get('name')} -- {info.get('status')}")
    print(f"\nsee {data.get('canonical_docs', {}).get('architecture', 'docs/ARCHITECTURE.md')} "
          f"for full detail, {data.get('canonical_docs', {}).get('workflow', 'docs/RESEARCH_WORKFLOW.md')} "
          f"for the legacy entry-point mapping (no canonical train/eval command exists yet)")
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    exit_code = 0
    result = validate_config_yaml(args.path)
    print(result.report())
    if not result.ok:
        exit_code = 1

    if args.objective:
        obj_result = validate_objective_yaml(args.objective)
        print()
        print(obj_result.report())
        if not obj_result.ok:
            exit_code = 1
    return exit_code


def _run_roots() -> list[Path]:
    """Directories under which runs are searched when given a bare run id.

    One resolver for the whole repo: ``run_layout.run_roots()`` yields
    ``research/runs/`` (where every new run is written) followed by the
    historical roots, so a bare run id resolves the canonical copy first.
    """
    try:
        from src.common.run_layout import run_roots
    except ImportError:  # stay defensive; the CLI must still print status
        return []
    try:
        return [p for p in run_roots() if p.is_dir()]
    except OSError:
        return []


def resolve_run(identifier: str) -> Path:
    """Resolve a run directory from a path OR a canonical run id.

    Resolving by identity rather than by filesystem path alone is the point
    of having a ``run_id``: a caller that must already know the physical
    directory has gained nothing from the manifest. A path is still accepted
    because that is how a run being actively worked on is referred to.
    """
    candidate = Path(identifier)
    if (candidate / "manifest.json").is_file():
        return candidate

    matches: list[Path] = []
    for root in _run_roots():
        for manifest_path in root.rglob("manifest.json"):
            try:
                if RunManifest.read(manifest_path.parent).run_id == identifier:
                    matches.append(manifest_path.parent)
            except (ValueError, TypeError, OSError):
                continue

    if not matches:
        raise FileNotFoundError(
            f"no run directory found for {identifier!r}: it is neither a "
            f"directory containing manifest.json nor a run_id recorded under "
            f"{[str(r) for r in _run_roots()] or 'any known run root'}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"run_id {identifier!r} is ambiguous -- found in "
            f"{[str(m) for m in matches]}. run_id must be unique."
        )
    return matches[0]


def _cmd_run_inspect(args: argparse.Namespace) -> int:
    print(RunManifest.read(resolve_run(args.run)).to_json())
    return 0


def _cmd_run_compare(args: argparse.Namespace) -> int:
    dir_a, dir_b = resolve_run(args.run_a), resolve_run(args.run_b)
    a, b = RunManifest.read(dir_a), RunManifest.read(dir_b)
    report = compare_runs(a, b)
    print(f"comparing {report.run_id_a} vs {report.run_id_b}")
    print(f"evidence: {a.evidence_level} vs {b.evidence_level}")
    for m, label in ((a, "a"), (b, "b")):
        if m.missing_evidence():
            print(f"  {label} missing required provenance: {', '.join(m.missing_evidence())}")
    print("\n--- SETUP ---")
    print(f"controlled_comparison: {report.controlled_comparison}")
    if report.attributable_dimension:
        print(f"single varying dimension: {report.attributable_dimension}")
    if report.missing_evidence:
        print(f"missing evidence: {', '.join(report.missing_evidence)}")
    if report.warning:
        print(f"\nWARNING: {report.warning}")
    for dim, leaves in report.differences.items():
        print(f"\n{dim}:")
        for leaf, (va, vb) in leaves.items():
            print(f"  {leaf}: {va!r} -> {vb!r}")

    results = compare_results(dir_a, dir_b, a, b)
    print("\n--- RESULTS ---")
    for artifact, side, err in results.invalid_evidence:
        print(f"{artifact}: INVALID in {side} -- present but unreadable ({err})")
    if results.clean:
        print("no result artifacts differ"
              + (f" (not applicable to either run: {', '.join(results.not_applicable)})"
                 if results.not_applicable else ""))
    for artifact, side in results.one_sided.items():
        print(f"{artifact}: present only in {side}")
    for artifact, side in results.missing_expected:
        print(f"{artifact}: MISSING in {side} -- its manifest implies this step ran")
    for artifact, leaves in results.differences.items():
        print(f"\n{artifact}:")
        for leaf, (va, vb) in leaves.items():
            print(f"  {leaf}: {va!r} -> {vb!r}")
    if results.not_applicable and not results.clean:
        print(f"\nnot applicable to either run: {', '.join(results.not_applicable)}")

    # A result difference between runs whose SETUP comparison is uncontrolled
    # is an observation, not a finding -- say so rather than leaving the two
    # sections to be read as one conclusion.
    if results.differences and not report.controlled_comparison:
        print("\nNOTE: the setup comparison above is not controlled, so these "
              "result differences are observations, not attributable findings.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Actually run the fast agent-safe verification and propagate its exit code.

    This runs pytest in a subprocess rather than printing a command for the
    caller to copy: a verification subcommand that cannot fail is not a
    verification subcommand. Scope defaults to ``tests/canonical/`` (DB-free,
    ~0.3s); ``--path`` narrows or widens it.
    """
    import subprocess

    target = args.path or "tests/canonical/"
    cmd = [sys.executable, "-m", "pytest", target, "-q"]
    print(f"$ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        print(
            f"\nverification FAILED (pytest exit {completed.returncode}) for {target!r}",
            file=sys.stderr,
        )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiment.cli",
        description="Canonical command surface (ADR-136). See docs/ARCHITECTURE.md.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="print canonical architecture status from project_manifest.yaml") \
        .set_defaults(func=_cmd_status)

    p_validate = sub.add_parser("validate-config", help="fail-loud validation of config.yaml (+ optional objective doc)")
    p_validate.add_argument("--path", default=str(REPO_ROOT / "config.yaml"))
    p_validate.add_argument("--objective", default=None, help="path to a config/objectives/<id>.yaml to also validate")
    p_validate.set_defaults(func=_cmd_validate_config)

    p_inspect = sub.add_parser("run-inspect", help="print a RunManifest, by run_id or directory")
    p_inspect.add_argument("run", help="canonical run_id, or a path to a run directory")
    p_inspect.set_defaults(func=_cmd_run_inspect)

    p_compare = sub.add_parser("run-compare", help="compare two runs, by run_id or directory")
    p_compare.add_argument("run_a", help="canonical run_id, or a path to a run directory")
    p_compare.add_argument("run_b", help="canonical run_id, or a path to a run directory")
    p_compare.set_defaults(func=_cmd_run_compare)

    p_verify = sub.add_parser(
        "verify", help="RUN the fast DB-free verification suite and propagate its exit code"
    )
    p_verify.add_argument(
        "--path", default=None,
        help="pytest target (default: tests/canonical/, the fast DB-free path)",
    )
    p_verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
