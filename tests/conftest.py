"""Shared pytest configuration: scope markers and a guard on the real data tree.

Two problems this solves.

**The suite used to write into ``data/``.** ``tests/test_iterate.py`` called
``iterate_until_target`` against the real ``nq-50pt`` objective, and because
``src.search.iterate`` resolves its output through ``src/common/data_paths.py``
that landed real artifacts in the live tree: run directories under
``data/research/runs/nq-50pt/``, appended rows in
``data/models/leaders/nq-50pt/leaderboard_v1.jsonl``, an overwritten
``latest_leaders.json``, and MLflow runs under ``data/mlflow_artifacts/``. The
architecture tests then parametrise over those run directories and fail on
them, so the suite's failure count grew every time it ran.

**Nothing stopped it.** There was no conftest at all, so no test was isolated
and no violation was detectable.

The fix is deliberately two-sided, because a blanket redirect of
``VTECH_DATA_DIR`` for every test would silently change the behaviour of tests
that legitimately read the live tree:

* :func:`_data_tree_guard` is the backstop for what the redirect cannot catch:
  a test that hardcodes ``/opt/vtech/data`` never consults ``data_paths``, so
  only a before/after comparison of the real tree will see it. Module-scoped so
  the report names the offender.
* :func:`sandboxed_data_root` is **default-deny**: every test's
  ``VTECH_DATA_DIR`` and ``MLFLOW_TRACKING_URI`` point at a temporary directory
  unless it carries ``@pytest.mark.real_data``. ``data_root()`` reads the
  variable on every call, so no cache invalidation is needed.

Scope markers (``acquisition``, ``db_write``) exist so a change can be gated on
the tests that cover it rather than on a suite that has no green baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Roots under the live data tree that hold durable research state. A test that
# creates anything here has escaped its sandbox.
_PROTECTED = (
    "research/runs",
    "research/search",
    "models/leaders",
    "models/registry",
    "models/staging",
    "mlflow_artifacts",
    "evaluation",
)
_MAX_DEPTH = 3


def _snapshot(root: Path) -> set[str]:
    """Paths at most ``_MAX_DEPTH`` levels below ``root``, relative to it."""
    found: set[str] = set()
    if not root.is_dir():
        return found
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        depth = len(here.parts) - base_depth
        if depth >= _MAX_DEPTH:
            dirnames[:] = []
        for name in list(dirnames) + filenames:
            found.add(str((here / name).relative_to(root)))
    return found


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "showcase: portable recorded Transformer WebUI and MLflow")
    config.addinivalue_line("markers", "ml_workflow: bounded synthetic Transformer workflow")
    config.addinivalue_line(
        "markers", "publication: synthetic offline public release behavior",
    )
    config.addinivalue_line(
        "markers",
        "real_data: this test reads the live /opt/vtech/data tree, so exempt it "
        "from the default sandbox. Use sparingly and say why.",
    )
    config.addinivalue_line(
        "markers",
        "acquisition: covers Databento acquisition — subscriptions, the live "
        "streamer, the raw DBN archive",
    )
    config.addinivalue_line(
        "markers",
        "db_write: covers writes into TimescaleDB — ingest, COPY/upsert paths",
    )


@pytest.fixture(autouse=True)
def sandboxed_data_root(request, tmp_path_factory):
    """Point every test's data root at a sandbox unless it opts out.

    **Default-deny, deliberately.** The first version of this was opt-in — a
    test wrote to the live tree unless someone remembered
    ``@pytest.mark.isolated_data``. That is the same failure mode the fixture
    exists to prevent, and it failed exactly as you would expect: one offender
    was found and marked, the problem declared solved, and two more
    (``test_manual_override``, ``test_research_director``) kept writing run
    directories and a ``_fold_cache`` entry until a later full-suite run
    happened to catch them.

    Opt-in cannot establish the invariant "no test writes to the live tree",
    because the set of offenders is unknown. Default-deny inverts the failure:
    forgetting a marker now makes a test *safe*, and a test that genuinely
    needs the live tree has to say so and is visible in one grep.

    Two roots are redirected, because not everything resolves through
    ``data_paths``:

    * ``VTECH_DATA_DIR`` — everything ``data_paths.data_root()`` backs.
    * ``MLFLOW_TRACKING_URI`` — MLflow logs to the tracking server, which
      writes its artifact store under ``data/mlflow_artifacts/``. Missing this
      is why the first fix looked complete and was not.

    A hardcoded path still escapes both, which is what ``_data_tree_guard``
    remains for.
    """
    if request.node.get_closest_marker("real_data") is not None:
        yield
        return
    sandbox = tmp_path_factory.mktemp("vtech_data")
    overrides = {
        "VTECH_DATA_DIR": str(sandbox),
        "MLFLOW_TRACKING_URI": f"file://{sandbox / 'mlruns'}",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, was in previous.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was


@pytest.fixture(scope="session")
def _live_data_root() -> Path:
    """The real data root, pinned once.

    Captured at session scope on purpose: ``sandboxed_data_root`` rewrites
    ``VTECH_DATA_DIR`` while a test runs, and the guard must keep watching the
    live tree rather than follow a sandbox.
    """
    return Path(os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))


@pytest.fixture(scope="session")
def _data_tree_cursor() -> dict:
    """Carries the previous module's snapshot so each one is taken once."""
    return {"snapshot": None}


@pytest.fixture(scope="module", autouse=True)
def _data_tree_guard(request, _live_data_root, _data_tree_cursor):
    """Fail a module that created state under the live data tree.

    Module scope rather than session scope so the report names the offender. A
    session-scoped check detects the write but attributes it to whichever test
    happened to run last, which is useless for fixing it. Per-test scope would
    be better still, but a snapshot costs ~119 ms against these roots — 262 s
    over the suite, against ~17 s per module once each snapshot is reused as the
    next module's baseline.

    Deliberately a detector, not a redirect: it changes no test's behaviour, so
    it is safe to leave on for every run while making the next violation
    impossible to miss — including one that hardcodes a path and so never
    consults ``data_paths``.
    """
    root = _live_data_root
    before = _data_tree_cursor["snapshot"]
    if before is None:
        before = {name: _snapshot(root / name) for name in _PROTECTED}
    yield
    after = {name: _snapshot(root / name) for name in _PROTECTED}
    _data_tree_cursor["snapshot"] = after

    added: list[str] = []
    for name in _PROTECTED:
        added.extend(f"{name}/{e}" for e in sorted(after[name] - before[name]))
    if added:
        shown = "\n  ".join(added[:40])
        more = f"\n  ... and {len(added) - 40} more" if len(added) > 40 else ""
        module = getattr(request.module, "__name__", "<unknown module>")
        pytest.fail(
            f"{module} wrote into the live data tree under {root} — mark the "
            "offending test with @pytest.mark.real_data only if it must read "
            "the live tree; otherwise stop it writing a hardcoded path to "
            f"tmp dir instead:\n  {shown}{more}",
            pytrace=False,
        )
