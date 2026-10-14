"""One live session per week, with the archive rotating inside it.

The streamer used to run five sessions a week, tearing the ``db.Live``
connection down whenever the session date advanced. That boundary was at
Chicago midnight — in live Asian-session trading — and cost 10–50 s of
unrecoverable quotes each time (measured 25 s, 50 s, 42 s and 10 s across the
2026-09-28 week; the Databento client replays no live gap).

Now the connection is held from the CME Sunday open to the Friday close, the
session date advances at 16:05 CT inside the window where no quote is written,
and ``RotatingDbnStream`` keeps the archive's one-directory-per-session-date
layout by rotating its file on the same calendar.

These tests cover the lifecycle contract. The rotation mechanics are in
tests/test_rotating_dbn_stream.py; the session calendar is in
tests/test_globex_session_date.py.
"""

from __future__ import annotations

import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.acquisition.live_streamer import LiveStreamer
from src.acquisition.rotating_stream import RotatingDbnStream

pytestmark = pytest.mark.acquisition


def _streamer(session: date, *, greeks: bool = False) -> LiveStreamer:
    s = LiveStreamer.__new__(LiveStreamer)
    s._option_symbols = set()
    s._option_session_date = session
    s._greeks_enabled = greeks
    s._archive = None
    s._known_symbols = set()
    s._synthetic_symbols = set()
    s._definitions_view_degraded = True   # gap reporting is tested elsewhere
    return s


def _universe(session: date, symbols: tuple[str, ...] = ("NQZ6 C25000",)):
    return SimpleNamespace(symbols=symbols, session_date=session, missing=(), complete=True)


# ── the roll keeps the connection ────────────────────────────────


def test_roll_adopts_the_new_date_and_keeps_the_client(monkeypatch) -> None:
    s = _streamer(date(2026, 9, 28))
    s._resolve_option_universe = lambda: _universe(date(2026, 9, 29))
    monkeypatch.setattr("src.acquisition.live_streamer.cme_weekend_closed", lambda: False)

    client = Mock()
    s._refresh_option_subscriptions(client)

    assert s._option_session_date == date(2026, 9, 29)
    client.stop.assert_not_called()


def test_roll_subscribes_the_new_chains_additively(monkeypatch) -> None:
    """Subscriptions accumulate; there is no unsubscribe API to shed the old."""
    s = _streamer(date(2026, 9, 28))
    s._option_symbols = {"Q4AU6 C25000"}
    s._resolve_option_universe = lambda: _universe(
        date(2026, 9, 29), ("Q4AU6 C25000", "Q5BU6 C25000")
    )
    monkeypatch.setattr("src.acquisition.live_streamer.cme_weekend_closed", lambda: False)

    client = Mock()
    s._refresh_option_subscriptions(client)

    for call in client.subscribe.call_args_list:
        assert call.kwargs["symbols"] == ["Q5BU6 C25000"], "only the new strike"
    assert s._option_symbols == {"Q4AU6 C25000", "Q5BU6 C25000"}


def test_no_roll_when_the_session_date_is_unchanged(monkeypatch) -> None:
    s = _streamer(date(2026, 9, 29))
    s._resolve_option_universe = lambda: _universe(date(2026, 9, 29))
    monkeypatch.setattr("src.acquisition.live_streamer.cme_weekend_closed", lambda: False)
    rolled: list = []
    s._roll_session = lambda d: rolled.append(d)

    s._refresh_option_subscriptions(Mock())
    assert rolled == []


def test_weekend_close_ends_the_week(monkeypatch) -> None:
    """The CME Friday close is the only thing that stops the client."""
    s = _streamer(date(2026, 10, 2))
    monkeypatch.setattr("src.acquisition.live_streamer.cme_weekend_closed", lambda: True)
    client = Mock()
    s._refresh_option_subscriptions(client)
    client.stop.assert_called_once()


def test_an_unresolvable_universe_does_not_roll_or_stop(monkeypatch) -> None:
    """A vendor metadata failure must not advance the session or kill the week."""
    s = _streamer(date(2026, 9, 29))
    s._resolve_option_universe = lambda: None
    monkeypatch.setattr("src.acquisition.live_streamer.cme_weekend_closed", lambda: False)

    client = Mock()
    s._refresh_option_subscriptions(client)

    assert s._option_session_date == date(2026, 9, 29)
    client.stop.assert_not_called()
    client.subscribe.assert_not_called()


# ── per-session buffers must flush on the roll ───────────────────


def test_roll_flushes_the_per_session_greeks_buffers() -> None:
    """There is no longer a teardown per session to flush them.

    Without this the whole week's surface rows would accumulate and land in
    Friday's parquet partition, because the partition date comes from the
    wall clock at flush time.
    """
    s = _streamer(date(2026, 9, 28), greeks=True)
    calls: list[str] = []
    s._flush_greeks_second = lambda force=False: calls.append(f"greeks(force={force})")
    s._flush_surface_parquet = lambda: calls.append("surface")

    s._roll_session(date(2026, 9, 29))

    assert calls == ["greeks(force=True)", "surface"]
    assert s._option_session_date == date(2026, 9, 29)


def test_roll_does_not_flush_when_greeks_are_disabled() -> None:
    s = _streamer(date(2026, 9, 28), greeks=False)
    s._flush_greeks_second = lambda force=False: pytest.fail("must not flush")
    s._flush_surface_parquet = lambda: pytest.fail("must not flush")
    s._roll_session(date(2026, 9, 29))
    assert s._option_session_date == date(2026, 9, 29)


def test_a_flush_fault_still_rolls_the_session(caplog) -> None:
    """A flush error must not leave the session stuck on the old date."""
    s = _streamer(date(2026, 9, 28), greeks=True)

    def boom(force: bool = False) -> None:
        raise RuntimeError("disk full")

    s._flush_greeks_second = boom
    s._flush_surface_parquet = lambda: None

    with caplog.at_level(logging.ERROR, logger="vtech.live"):
        s._roll_session(date(2026, 9, 29))

    assert s._option_session_date == date(2026, 9, 29)
    assert any(r.message == "session_roll_flush_failed" for r in caplog.records)


def test_roll_is_logged_with_both_dates(caplog) -> None:
    s = _streamer(date(2026, 9, 28))
    s._option_symbols = {"a", "b"}
    with caplog.at_level(logging.INFO, logger="vtech.live"):
        s._roll_session(date(2026, 9, 29))
    record = next(r for r in caplog.records if r.message == "live_option_session_roll")
    assert record.to == "2026-09-29"
    assert getattr(record, "from") == "2026-09-28"
    assert record.n_subscribed == 2


# ── the archive is wired to the same calendar ────────────────────


def test_archive_rotates_on_the_session_calendar(tmp_path) -> None:
    """The stream the session hands to add_stream rotates where the roll does.

    Asserted through the real class rather than a mock, so the wiring and the
    rotation contract are checked together.
    """
    from src.acquisition.options_universe import globex_session_date

    archive = RotatingDbnStream(
        root=tmp_path, session_date_fn=globex_session_date
    )
    # Same callable the session passes; prove it is the session calendar by
    # driving it with explicit instants.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ct = ZoneInfo("America/Chicago")
    moments = iter(
        [
            datetime(2026, 9, 28, 15, 0, tzinfo=ct),   # Monday session
            datetime(2026, 9, 28, 16, 30, tzinfo=ct),  # past the 16:05 roll
        ]
    )
    archive = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: globex_session_date(next(moments))
    )
    archive.write(b"DBN\x03" + b"\x00" * 40)
    archive.write(b"\x01" * 40)
    archive.write(b"\x02" * 40)
    archive.close()

    dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert dirs == ["2026-09-28", "2026-09-29"]
    assert archive.rotations == 1


def test_the_session_opens_a_rotating_archive_on_the_session_calendar() -> None:
    """The wiring: what _run_session hands to add_stream.

    Rooted under the resolver's live archive directory, and driven by the same
    calendar the subscriptions roll on — otherwise the archive and the
    subscriptions would disagree about which session a record belongs to.
    """
    from src.common.data_paths import raw_dir
    from src.acquisition.options_universe import live_option_session_date

    archive = LiveStreamer._open_archive()

    assert isinstance(archive, RotatingDbnStream)
    assert archive._root == raw_dir() / "live"
    assert archive._session_date_fn is live_option_session_date
    # Nothing is created until the first record, so merely opening the session
    # cannot leave an empty directory behind.
    assert archive.current_path is None
    assert archive.files_opened == 0


def test_the_archive_satisfies_the_client_stream_contract() -> None:
    """add_stream validates the object before the session accepts it."""
    from databento.common.types import ClientStream

    wrapped = ClientStream(LiveStreamer._open_archive())
    assert not wrapped.is_closed
    assert wrapped.stream_name == "RotatingDbnStream"
