"""The stdlib log formatter must keep ``extra={}`` fields.

Marked ``acquisition`` because the acquisition path is what this exists for: a
week-long live session is only diagnosable through these fields. The live
streamer logs ``live_option_universe`` (resolved symbol counts and chain roots)
and ``reconnect_gap`` (the exact boundaries of a live data gap, handed over by
the Databento client's reconnect callback) entirely through ``extra``, and both
were being discarded before they reached the journal.
"""

from __future__ import annotations

import logging

import pytest

from src.common.logging import ExtraFormatter

pytestmark = pytest.mark.acquisition


def _render(msg: str, **extra) -> str:
    record = logging.LogRecord(
        name="vtech.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return ExtraFormatter("%(message)s").format(record)


def test_message_only_record_is_unchanged() -> None:
    """No extras means byte-identical output, so existing greps are unaffected."""
    assert _render("live_option_universe") == "live_option_universe"


def test_message_stays_the_leading_token() -> None:
    """``journalctl | grep <event>`` must keep working."""
    out = _render("reconnect_gap", gap_s=25.0)
    assert out.startswith("reconnect_gap")


def test_extras_are_rendered() -> None:
    out = _render("live_option_symbols_added", n_new=74, n_total=2568)
    assert "n_new=74" in out
    assert "n_total=2568" in out


def test_extras_are_sorted_for_stable_output() -> None:
    out = _render("evt", zebra=1, alpha=2, middle=3)
    assert out == "evt alpha=2 middle=3 zebra=1"


def test_values_containing_spaces_or_equals_are_quoted() -> None:
    """Otherwise a value with a space silently splits into two fields."""
    out = _render("evt", path="/a b/c.dbn", plain="simple")
    assert "path='/a b/c.dbn'" in out
    assert "plain=simple" in out


def test_the_fields_the_streamer_actually_logs_survive() -> None:
    """The three events that were unobservable, with their real payloads."""
    universe = _render(
        "live_option_universe",
        session_date="2026-09-29",
        n_symbols=2568,
        roots={"0dte": "Q5BU6", "qtr": "NQZ6"},
    )
    assert "n_symbols=2568" in universe
    assert "session_date=2026-09-29" in universe
    assert "Q5BU6" in universe

    gap = _render("reconnect_gap", start="2026-09-30 05:03:59", end="2026-09-30 05:04:49", gap_s=50.0)
    assert "gap_s=50.0" in gap
    assert "start='2026-09-30 05:03:59'" in gap

    rotated = _render(
        "dbn_archive_opened",
        path="/opt/vtech/data/sources/raw/live/2026-09-29/stream_050228.dbn",
        session_date="2026-09-29",
        files_opened=2,
    )
    assert "files_opened=2" in rotated
    assert "stream_050228.dbn" in rotated


def test_reserved_record_attributes_are_not_leaked() -> None:
    """Only caller-supplied fields appear — not LogRecord's own internals."""
    out = _render("evt", n_new=1)
    for reserved in ("levelname", "pathname", "lineno", "msg", "args", "name", "created"):
        assert f"{reserved}=" not in out


def test_setup_logging_installs_the_formatter() -> None:
    """The real entrypoint must wire it, not just the class existing."""
    from src.common.logging import setup_logging

    setup_logging()
    handlers = logging.getLogger().handlers
    assert handlers, "setup_logging left the root logger with no handler"
    assert any(isinstance(h.formatter, ExtraFormatter) for h in handlers)
