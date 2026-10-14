"""Promote a staged serve bundle → ``model_registry/<objective_id>/``.

Layout (objective-keyed SSOT)::

    models/model_registry/<objective_id>/
      LATEST.json                 # production pin for Live + Research Director
      <candidate>_<run_ts>/       # copy of the serve bundle

``LATEST.json`` shape::

    {
      "objective_id": "nq-50pt-econ",
      "version": "xgb_tbm_topgroups_20260715T234308Z",
      "path": "…/model_registry/nq-50pt-econ/xgb_tbm_topgroups_…",  # absolute or relative
      "candidate_id": "xgb_tbm_topgroups",
      "family": "xgboost",
      "promoted_at": "…",
      "staging_dir": "…",
      "git_sha": "…"
    }

Only registry LATEST models enter the Research Director baseline loop.
Staging remains the pre-production serve sandbox (Live may still score it).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.data_paths import model_registry_dir, model_staging_dir
from src.serving.bundle import is_serving_bundle

logger = logging.getLogger("vtech.serving.promote")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).decode().strip()
    except Exception:
        return "unknown"


def registry_objective_dir(objective_id: str, *, registry_root: Path | None = None) -> Path:
    return (registry_root or model_registry_dir()) / objective_id


def latest_pointer_path(objective_id: str, *, registry_root: Path | None = None) -> Path:
    return registry_objective_dir(objective_id, registry_root=registry_root) / "LATEST.json"


def load_latest(objective_id: str, *, registry_root: Path | None = None) -> dict[str, Any] | None:
    """Load registry ``LATEST.json`` for *objective_id*, or None."""
    p = latest_pointer_path(objective_id, registry_root=registry_root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def resolve_latest_bundle(
    objective_id: str, *, registry_root: Path | None = None,
) -> Path | None:
    """Absolute path to the serve bundle pointed at by LATEST, if valid."""
    latest = load_latest(objective_id, registry_root=registry_root)
    if not latest:
        return None
    root = registry_root or model_registry_dir()
    path_s = latest.get("path") or ""
    version = latest.get("version") or ""
    candidates: list[Path] = []
    if path_s:
        p = Path(path_s)
        if not p.is_absolute():
            p = root / path_s
        candidates.append(p)
    if version:
        candidates.append(root / objective_id / version)
    for p in candidates:
        if is_serving_bundle(p):
            return p.resolve()
    return None


def promote(
    staging_dir: str | Path,
    *,
    objective_id: str | None = None,
    registry_root: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy *staging_dir* into registry under its objective and write LATEST.json.

    Returns the LATEST pointer dict.
    """
    src = Path(staging_dir).resolve()
    if not is_serving_bundle(src):
        raise ValueError(
            f"not a serving bundle (need feature_schema.json + model artifact): {src}")

    manifest: dict[str, Any] = {}
    mp = src / "finalize_manifest.json"
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text())
        except Exception:
            manifest = {}

    oid = objective_id or manifest.get("objective_id") or src.parent.name
    version = manifest.get("version") or src.name
    reg_root = registry_root or model_registry_dir()
    dest = reg_root / oid / version

    if dest.exists() and not force and not dry_run:
        # Allow re-pointing LATEST to existing version without full recopy if same
        if dest.resolve() != src.resolve():
            raise FileExistsError(
                f"registry version already exists: {dest} (pass force=True to overwrite)")

    pointer = {
        "objective_id": oid,
        "version": version,
        "candidate_id": manifest.get("candidate_id") or version,
        "family": manifest.get("family"),
        "path": str(dest.resolve()) if not dry_run else str(dest),
        "staging_dir": str(src),
        "search_run_ts": manifest.get("search_run_ts"),
        "search_run_dir": manifest.get("search_run_dir"),
        "search_score": manifest.get("search_score"),
        "promoted_at": _ts(),
        "git_sha": _git_sha(),
        "promoted_by": "src.serving.promote",
    }

    logger.info(
        "promote objective=%s version=%s src=%s dest=%s dry_run=%s",
        oid, version, src, dest, dry_run,
    )
    if dry_run:
        pointer["path"] = str(dest)
        return pointer

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    # Ensure finalize_manifest carries promotion note
    man_out = dict(manifest)
    man_out["promoted_at"] = pointer["promoted_at"]
    man_out["registry_path"] = str(dest.resolve())
    (dest / "finalize_manifest.json").write_text(
        json.dumps(man_out, indent=2, default=str))

    pointer["path"] = str(dest.resolve())
    latest_path = latest_pointer_path(oid, registry_root=reg_root)
    latest_path.write_text(json.dumps(pointer, indent=2, default=str))
    (dest / "provenance.json").write_text(json.dumps(pointer, indent=2, default=str))
    (dest / "_SUCCESS").write_text("")

    logger.info("promoted LATEST → %s", latest_path)
    return pointer


def promote_cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Promote staged serve bundle → model_registry/<objective>/LATEST")
    p.add_argument(
        "--staging-dir", type=Path, default=None,
        help="Serve bundle dir under model_staging (required unless --objective auto)")
    p.add_argument(
        "--objective", default=None,
        help="objective_id (default: from finalize_manifest or staging parent)")
    p.add_argument(
        "--version", default=None,
        help="If set with --objective, use staging/<objective>/<version>")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    src = args.staging_dir
    if src is None:
        if not args.objective or not args.version:
            print("ERROR: pass --staging-dir or both --objective and --version",
                  file=sys.stderr)
            return 2
        src = model_staging_dir() / args.objective / args.version
    try:
        pointer = promote(
            src,
            objective_id=args.objective,
            force=args.force,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(pointer, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(promote_cli())
