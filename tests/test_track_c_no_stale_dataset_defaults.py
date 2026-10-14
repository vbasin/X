"""No Track C consumer may silently select a version-suffixed historical dataset
(docs/STORAGE_INVENTORY.md §4 STOR-2, tenth pass). Argparse-level only: no data
loaded, no training, no DB.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / script), *args],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )


def test_eval_move_model_requires_an_explicit_input():
    """Either an explicit legacy `--dir` or a resolvable run (`--objective`
    plus `--run-ts`). Never a silent version-suffixed default."""
    r = _run("scripts/features/eval_move_model.py")
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "--dir" in msg and "--objective" in msg and "--run-ts" in msg


def test_eval_target_model_requires_an_explicit_input():
    r = _run("scripts/features/eval_target_model.py")
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "--dir" in msg and "--objective" in msg and "--run-ts" in msg


def test_build_target_matrix_requires_explicit_move_dir_unless_context_none():
    r = _run("scripts/features/build_target_matrix.py", "--scores", "/tmp/does-not-exist.parquet")
    assert r.returncode == 1
    assert "--move-dir is required" in r.stdout + r.stderr


def test_build_target_matrix_context_none_does_not_require_move_dir():
    """--context none legitimately needs no move-dir -- the required-arg check
    must not overreach into forbidding a genuinely optional input."""
    r = _run("scripts/features/build_target_matrix.py", "--scores", "/tmp/does-not-exist.parquet",
             "--context", "none")
    # Fails later (no --scores file / no data root), never on the --move-dir check.
    assert "--move-dir is required" not in (r.stdout + r.stderr)


def test_no_script_hardcodes_a_version_suffixed_default_dir():
    """Static guard: no default= value in these scripts' --dir/--move-dir/--scores
    argparse definitions may point at a version-suffixed or unsuffixed stale
    directory literal (move_matrix_v3, target_matrix, etc)."""
    import ast
    for script in ("scripts/features/eval_move_model.py",
                   "scripts/features/eval_target_model.py"):
        tree = ast.parse((REPO / script).read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--dir"):
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                assert "default" not in kwargs or (
                    isinstance(kwargs["default"], ast.Constant)
                    and kwargs["default"].value is None
                ), f"{script}: --dir must not have a non-None default"
