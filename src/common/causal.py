"""The repository-wide causal ("as-of") invariant, as one tested primitive.

Rule: for a decision at timestamp T, no feature or context value may depend
on information that was not available at or before T. T itself is included:
a row at 13:01:00 may use data through 13:01:00 and must not use 13:01:01
or later. A label may use the forward price path to define the outcome;
that path is not a feature. Concretely, an as-of lookup is "the position of
the last row at or before each query timestamp."

This is a direct, faithful port of Track C's ``_asof()`` helper
(``src/training/entry_state.py:335-337``), independently re-verified clean
this session (adversarial re-audit found no leak in its use across
``entry_state.py``, ``flow_state.py``, ``structural_transitions.py``, and
``cash_session_profile.py``'s session-history filtering). It is extracted
here, unchanged in behavior, so future as-of lookups anywhere in the
codebase can use a single tested primitive instead of re-deriving the
search/clamp logic by hand -- the exact failure mode this guards against
(an unclamped or ``side="left"`` index silently reading a future value) was
*not* found in the current cascade code, but is worth preventing by
construction rather than by repeated manual audit.

Two call sites elsewhere in the repo intentionally use ``side="left"`` for a
different, non-violating purpose (resolving "first fill at-or-after
decision", which is forward-looking by design, not a state lookup):
``src/training/move_label.py:120``, ``scripts/features/build_target_matrix.py:148``.
Do not "fix" those to use this helper -- they are solving a different
problem. See ``docs/DATA_CONTRACTS.md`` section 3 for the full list of
known non-violating look-alikes.
"""
from __future__ import annotations

import numpy as np


def as_of_index(reference_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Index into ``reference_ts`` of the last entry at or before each ``query_ts``.

    Both arrays must be sorted ascending (``reference_ts`` strictly; this is
    the caller's responsibility -- a tick tape or bar-grid index already is).
    Returns ``-1`` for a query timestamp before the first reference entry;
    callers must mask that case explicitly rather than silently reading
    index 0 (see ``as_of_index_masked`` below for the common pattern).

    >>> ref = np.array([10, 20, 30], dtype="int64")
    >>> as_of_index(ref, np.array([5, 10, 15, 30, 99], dtype="int64"))
    array([-1,  0,  0,  2,  2])
    """
    reference_ts = np.asarray(reference_ts)
    query_ts = np.asarray(query_ts)
    return np.searchsorted(reference_ts, query_ts, side="right") - 1


def as_of_index_masked(
    reference_ts: np.ndarray, query_ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """As :func:`as_of_index`, plus an explicit ``seen`` mask for ``pos < 0``.

    Returns ``(pos, seen)`` where ``pos`` has every ``-1`` replaced by ``0``
    (a *safe*, not meaningful, index -- callers must gate any read through
    ``pos`` on ``seen`` before trusting it) and ``seen[i]`` is ``False``
    exactly where the original position was ``-1`` (no reference entry at or
    before that query timestamp). This mirrors the ``pos >= 0`` / ``safe =
    np.maximum(pos, 0)`` pattern in ``entry_state.py`` exactly, so a caller
    porting existing cascade code over keeps identical behavior.

    >>> ref = np.array([10, 20, 30], dtype="int64")
    >>> pos, seen = as_of_index_masked(ref, np.array([5, 10, 99], dtype="int64"))
    >>> pos.tolist(), seen.tolist()
    ([0, 0, 2], [False, True, True])
    """
    pos = as_of_index(reference_ts, query_ts)
    seen = pos >= 0
    safe = np.maximum(pos, 0)
    return safe, seen


class CausalityError(ValueError):
    """A joined feature or context value is knowable only after the decision."""


def assert_known_at_or_before(available_ts: np.ndarray, decision_ts: np.ndarray) -> None:
    """Raise if any attached value became available after its decision timestamp.

    Both arrays are int64 timestamps in the same unit. Callers pass only the
    rows that were actually attached. An empty attachment is valid.
    """
    available = np.asarray(available_ts)
    decision = np.asarray(decision_ts)
    if available.shape != decision.shape:
        raise CausalityError("availability and decision timestamps differ in length")
    if available.size and np.any(available > decision):
        raise CausalityError(
            "a joined value is available only after its decision timestamp"
        )
