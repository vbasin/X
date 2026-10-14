"""Tests for src.serving.lifecycle."""
import pytest

from src.serving.lifecycle import LifecycleStatus, is_valid_transition


def test_all_five_canonical_statuses_exist():
    assert {s.value for s in LifecycleStatus} == {
        "EXPERIMENTAL", "CANDIDATE", "PRODUCTION", "DEPRECATED", "ARCHIVED",
    }


@pytest.mark.parametrize("current,target", [
    (LifecycleStatus.EXPERIMENTAL, LifecycleStatus.CANDIDATE),
    (LifecycleStatus.CANDIDATE, LifecycleStatus.PRODUCTION),
    (LifecycleStatus.CANDIDATE, LifecycleStatus.EXPERIMENTAL),
    (LifecycleStatus.PRODUCTION, LifecycleStatus.DEPRECATED),
    (LifecycleStatus.DEPRECATED, LifecycleStatus.ARCHIVED),
])
def test_valid_forward_transitions(current, target):
    assert is_valid_transition(current, target)


def test_production_cannot_skip_deprecated_back_to_candidate():
    """A production model that needs rework must be DEPRECATED first (an
    explicit, recorded decision) -- it cannot silently revert to CANDIDATE.
    This mirrors the audit's finding that Track B's one promoted model was
    never formally deprecated despite badly missing its backtest in paper
    trading."""
    assert not is_valid_transition(LifecycleStatus.PRODUCTION, LifecycleStatus.CANDIDATE)
    assert not is_valid_transition(LifecycleStatus.PRODUCTION, LifecycleStatus.EXPERIMENTAL)


def test_archived_is_terminal():
    for target in LifecycleStatus:
        assert not is_valid_transition(LifecycleStatus.ARCHIVED, target)


def test_experimental_cannot_jump_directly_to_production():
    assert not is_valid_transition(LifecycleStatus.EXPERIMENTAL, LifecycleStatus.PRODUCTION)
