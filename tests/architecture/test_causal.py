"""Tests for src.common.causal — the repository-wide
as-of primitive ported from Track C's verified-clean entry_state.py pattern.
"""
import numpy as np
import pytest

from src.common.causal import (
    CausalityError,
    as_of_index,
    as_of_index_masked,
    assert_known_at_or_before,
)


def test_as_of_index_exact_and_between_matches():
    ref = np.array([10, 20, 30], dtype="int64")
    query = np.array([5, 10, 15, 30, 99], dtype="int64")
    got = as_of_index(ref, query)
    np.testing.assert_array_equal(got, np.array([-1, 0, 0, 2, 2]))


def test_as_of_index_never_reads_forward():
    """For every query timestamp, the returned reference entry's timestamp
    must be <= the query timestamp -- the entire point of the invariant."""
    rng = np.random.default_rng(0)
    ref = np.sort(rng.integers(0, 1_000_000, size=500)).astype("int64")
    query = rng.integers(-100, 1_000_100, size=2_000).astype("int64")
    pos = as_of_index(ref, query)
    seen = pos >= 0
    assert np.all(ref[pos[seen]] <= query[seen])


def test_as_of_index_masked_matches_entry_state_pattern():
    """Mirrors entry_state.py's `pos >= 0` / `safe = np.maximum(pos, 0)`
    pattern exactly."""
    ref = np.array([10, 20, 30], dtype="int64")
    query = np.array([5, 10, 99], dtype="int64")
    pos, seen = as_of_index_masked(ref, query)
    np.testing.assert_array_equal(pos, np.array([0, 0, 2]))
    np.testing.assert_array_equal(seen, np.array([False, True, True]))


def test_as_of_index_empty_reference_all_unseen():
    ref = np.array([], dtype="int64")
    query = np.array([1, 2, 3], dtype="int64")
    pos, seen = as_of_index_masked(ref, query)
    assert not seen.any()
    np.testing.assert_array_equal(pos, np.zeros(3, dtype="int64"))


def test_assert_known_at_or_before_rejects_a_later_value():
    assert_known_at_or_before(np.array([10], dtype="int64"), np.array([10], dtype="int64"))
    with pytest.raises(CausalityError):
        assert_known_at_or_before(np.array([11], dtype="int64"), np.array([10], dtype="int64"))


@pytest.mark.parametrize("query_val,expected_idx", [(0, -1), (100, 9), (1000, 99)])
def test_as_of_index_boundary_values(query_val, expected_idx):
    ref = np.arange(1, 101, dtype="int64") * 10  # 10, 20, ..., 1000
    got = as_of_index(ref, np.array([query_val], dtype="int64"))
    assert got[0] == expected_idx
