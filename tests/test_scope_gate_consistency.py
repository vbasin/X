"""The scoped gate's path list must equal its marker set.

``make test-acquisition`` runs an explicit list of test files rather than
``pytest tests/ -m acquisition``, because marker selection still *imports* all
147 test modules in order to collect them — so any module-level side effect in
an unrelated test runs during a supposedly narrow gate. Explicit paths never
import what they do not run.

The cost of that choice is a list that can drift: someone marks a new module
and forgets the Makefile, or removes a marker and leaves a stale path. This
test closes that gap by comparing the two directly, so the gate cannot silently
stop covering something.

It parses the Makefile as text on purpose — importing Make variables is not
possible, and shelling out to ``make`` to print them would make a unit test
depend on a working Make environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.acquisition

REPO = Path(__file__).resolve().parents[1]
MAKEFILE = REPO / "Makefile"
TESTS_DIR = REPO / "tests"

# The gate variable in the Makefile, e.g. ACQUISITION_TESTS, keyed by marker.
GATES = {"acquisition": "ACQUISITION_TESTS", "publication": "PUBLICATION_TESTS", "showcase": "SHOWCASE_TESTS", "ml_workflow": "ML_WORKFLOW_TESTS"}


def _makefile_paths(variable: str) -> set[str]:
    """Test paths assigned to ``variable``, following backslash continuations."""
    text = MAKEFILE.read_text()
    match = re.search(rf"^{variable}\s*=\s*((?:.*\\\n)*.*)$", text, re.MULTILINE)
    assert match, f"{variable} not found in {MAKEFILE}"
    body = match.group(1).replace("\\\n", " ")
    return {token for token in body.split() if token.endswith(".py")}


def _marked_modules(marker: str) -> set[str]:
    """Test files declaring ``pytestmark = pytest.mark.<marker>`` at module level."""
    pattern = re.compile(
        rf"^pytestmark\s*=\s*pytest\.mark\.{re.escape(marker)}\b", re.MULTILINE
    )
    found = set()
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        if pattern.search(path.read_text()):
            found.add(str(path.relative_to(REPO)))
    return found


@pytest.mark.parametrize(("marker", "variable"), sorted(GATES.items()))
def test_gate_paths_match_the_marker_set(marker: str, variable: str) -> None:
    listed = _makefile_paths(variable)
    marked = _marked_modules(marker)

    missing_from_makefile = marked - listed
    stale_in_makefile = listed - marked

    assert not missing_from_makefile, (
        f"these modules carry @pytest.mark.{marker} but are not in {variable}, "
        f"so the gate does not run them: {sorted(missing_from_makefile)}"
    )
    assert not stale_in_makefile, (
        f"these paths are in {variable} but no longer carry "
        f"@pytest.mark.{marker}: {sorted(stale_in_makefile)}"
    )


@pytest.mark.parametrize(("_marker", "variable"), sorted(GATES.items()))
def test_every_gate_path_exists(_marker: str, variable: str) -> None:
    """A renamed or deleted test file must not silently drop out of the gate."""
    missing = [p for p in sorted(_makefile_paths(variable)) if not (REPO / p).is_file()]
    assert not missing, f"{variable} names files that do not exist: {missing}"


def test_the_gate_is_not_empty() -> None:
    """A gate that selects nothing passes vacuously and proves nothing."""
    for marker, variable in GATES.items():
        assert _makefile_paths(variable), f"{variable} is empty"
        assert _marked_modules(marker), f"no module carries @pytest.mark.{marker}"


def test_this_module_is_itself_in_the_gate() -> None:
    """The consistency check has to run for the consistency check to mean anything.

    It is listed in ``ACQUISITION_TESTS`` like any other gate member rather than
    appended to the recipe, so there is no special case for the drift check
    itself to miss.
    """
    assert "pytest.mark.acquisition" in Path(__file__).read_text()
    this_module = str(Path(__file__).resolve().relative_to(REPO))
    assert this_module in _makefile_paths("ACQUISITION_TESTS"), (
        "the gate must run this consistency test, or drift goes unnoticed"
    )
