"""Shared git-commit identity -- the one place this repository asks git who it is.

Extracted from ``src/search/run_manifest.py::_git_identity()`` (tenth pass, Storage
Implementation Tranche 1) so any writer needing genuine code provenance -- Track B's
``RunManifest.code``, Track C's outcome-dataset build sidecar
(``src/training/label_store.py::write_build_sidecar``) -- uses one mechanism rather
than a second copy of the same subprocess calls.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_identity() -> tuple[str, bool] | None:
    """``(git_sha, dirty)`` for the current commit, or ``None`` if git is unavailable.

    ``dirty`` is a real working-tree check, not assumed ``False``: a bare SHA with
    no dirty guard is an unverifiable claim about what code actually ran.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if not sha:
            return None
        porcelain = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return sha, bool(porcelain.strip())
    except (OSError, subprocess.SubprocessError):
        return None
