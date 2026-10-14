"""A missing option definition must be visible, not permanent.

``_refresh_option_subscriptions`` used to diff only on subscription state::

    new_symbols = [s for s in universe.symbols if s not in self._option_symbols]
    if not new_symbols:
        return

A strike that resolved, was subscribed, and whose ``definition`` message never
arrived was therefore in ``_option_symbols`` forever, so every later refresh
returned early and never noticed. Measured consequence: 74 of 246 quoted NQZ6
strikes held no definition row on every session from the 2026-09-18 quarterly
roll onward, and because ``FeatureEngine._load_bbo_1s`` resolves its
instrument-id filter *from* ``databento.definitions``, those strikes were
silently absent from every options feature with nothing logged.
"""

from __future__ import annotations

import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.acquisition.live_streamer import LiveStreamer

pytestmark = pytest.mark.acquisition


def _streamer(known: set[str], synthetic: set[str]) -> LiveStreamer:
    s = LiveStreamer.__new__(LiveStreamer)
    s._known_symbols = set(known)
    s._synthetic_symbols = set(synthetic)
    s._option_symbols = set()
    return s


def _universe(symbols: tuple[str, ...], session: date = date(2026, 9, 29)):
    return SimpleNamespace(symbols=symbols, session_date=session, missing=())



def _hold_session(monkeypatch, session: date) -> None:
    """Pin the clock-derived session date so the roll branch is not taken.

    ``_refresh_option_subscriptions`` compares ``live_option_session_date()``
    against the streamer's own session date and tears the client down on a
    mismatch; these tests are about the subscription diff, not the roll.
    """
    monkeypatch.setattr(
        "src.acquisition.live_streamer.cme_weekend_closed", lambda: False
    )
    monkeypatch.setattr(
        "src.acquisition.live_streamer.live_option_session_date",
        lambda *_a, **_k: session,
    )


# ── gap detection ────────────────────────────────────────────────


def test_no_gaps_when_every_symbol_has_a_vendor_row() -> None:
    s = _streamer({"NQZ6 C25000", "QNEU6 P24000"}, set())
    assert s.definition_gaps(["NQZ6 C25000", "QNEU6 P24000"]) == []


def test_a_symbol_with_no_row_is_reported() -> None:
    s = _streamer({"NQZ6 C25000"}, set())
    assert s.definition_gaps(["NQZ6 C25000", "NQZ6 C25500"]) == ["NQZ6 C25500"]


def test_a_legacy_synthetic_row_requires_vendor_acquisition() -> None:
    """Legacy reconstructions do not satisfy vendor-only readiness."""
    s = _streamer(set(), {"QNEU6 P24000"})
    assert s.definition_gaps(["QNEU6 P24000"]) == ["QNEU6 P24000"]


def test_gaps_are_sorted_and_deduplicated_by_input() -> None:
    s = _streamer(set(), set())
    assert s.definition_gaps(["NQZ6 C25500", "NQZ6 C25000"]) == [
        "NQZ6 C25000",
        "NQZ6 C25500",
    ]


def test_the_measured_quarterly_shortfall() -> None:
    """The real shape: 246 quoted, 172 with rows, 74 missing."""
    quoted = [f"NQZ6 C{17750 + 500 * i}" for i in range(246)]
    s = _streamer(set(quoted[:172]), set())
    assert len(s.definition_gaps(quoted)) == 74


# ── reporting ────────────────────────────────────────────────────


def test_gap_report_names_counts_and_chains(caplog: pytest.LogCaptureFixture) -> None:
    s = _streamer({"NQZ6 C25000"}, set())
    universe = _universe(("NQZ6 C25000", "NQZ6 C25500", "NQZ6 C26000", "QNEU6 P24000"))

    with caplog.at_level(logging.WARNING, logger="vtech.live"):
        s._report_definition_gaps(universe)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.message == "option_definitions_missing"
    assert record.n_missing == 3
    assert record.n_resolved == 4
    assert record.by_root == {"NQZ6": 2, "QNEU6": 1}
    assert record.session_date == "2026-09-29"


def test_silent_when_there_is_nothing_to_report(caplog: pytest.LogCaptureFixture) -> None:
    """A clean session must not log a warning every 15 minutes."""
    s = _streamer({"NQZ6 C25000"}, set())
    with caplog.at_level(logging.WARNING, logger="vtech.live"):
        s._report_definition_gaps(_universe(("NQZ6 C25000",)))
    assert caplog.records == []


def test_degraded_definitions_view_suppresses_the_gap_report(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed table read must not masquerade as 3,900 missing definitions.

    ``_bootstrap_definitions`` retries and then starts degraded with empty
    symbol sets — observed in the journal as ``definitions_bootstrap_degraded``
    on 2026-10-06. Without this guard every resolved symbol would be reported
    missing on every refresh, burying the real signal.
    """
    s = _streamer(set(), set())
    s._definitions_view_degraded = True
    universe = _universe(tuple(f"NQZ6 C{25000 + 50 * i}" for i in range(40)))

    with caplog.at_level(logging.WARNING, logger="vtech.live"):
        s._report_definition_gaps(universe)

    assert len(caplog.records) == 1
    assert caplog.records[0].message == "option_definitions_gap_check_skipped"
    assert caplog.records[0].reason == "definitions_view_degraded"
    assert caplog.records[0].n_resolved == 40


def test_healthy_view_still_reports(caplog: pytest.LogCaptureFixture) -> None:
    s = _streamer({"NQZ6 C25000"}, set())
    s._definitions_view_degraded = False
    with caplog.at_level(logging.WARNING, logger="vtech.live"):
        s._report_definition_gaps(_universe(("NQZ6 C25000", "NQZ6 C25500")))
    assert caplog.records[0].message == "option_definitions_missing"
    assert caplog.records[0].n_missing == 1


# ── the refresh no longer returns early ──────────────────────────


def test_refresh_checks_definitions_even_when_nothing_new_is_subscribed(monkeypatch) -> None:
    """The regression that made the gap permanent.

    Everything resolved is already subscribed, so the subscribe branch is
    skipped — but the definition check must still run.
    """
    s = LiveStreamer.__new__(LiveStreamer)
    s._known_symbols = {"NQZ6 C25000"}
    s._synthetic_symbols = set()
    s._option_symbols = {"NQZ6 C25000", "NQZ6 C25500"}
    s._option_session_date = date(2026, 9, 29)
    universe = _universe(("NQZ6 C25000", "NQZ6 C25500"))
    s._resolve_option_universe = lambda: universe

    client = Mock()
    s._definition_acquirer = Mock()
    reported: list = []
    s._report_definition_gaps = lambda u: reported.append(u)
    _hold_session(monkeypatch, universe.session_date)

    s._refresh_option_subscriptions(client)

    client.subscribe.assert_not_called()      # nothing new to subscribe
    client.stop.assert_not_called()           # and no teardown
    assert reported == [universe], "definition check must run regardless"
    submitted = set(s._definition_acquirer.submit.call_args.args[0])
    assert submitted == set(universe.symbols), "already subscribed symbols still acquire metadata"


def test_refresh_subscribes_new_symbols_and_still_checks(monkeypatch) -> None:
    s = LiveStreamer.__new__(LiveStreamer)
    s._known_symbols = set()
    s._synthetic_symbols = set()
    s._option_symbols = {"NQZ6 C25000"}
    s._option_session_date = date(2026, 9, 29)
    universe = _universe(("NQZ6 C25000", "NQZ6 C25500"))
    s._resolve_option_universe = lambda: universe

    client = Mock()
    s._definition_acquirer = Mock()
    reported: list = []
    s._report_definition_gaps = lambda u: reported.append(u)
    _hold_session(monkeypatch, universe.session_date)

    s._refresh_option_subscriptions(client)

    schemas = [c.kwargs["schema"] for c in client.subscribe.call_args_list]
    assert schemas == ["definition", "bbo-1s"], "definition first, then quotes"
    for call in client.subscribe.call_args_list:
        assert call.kwargs["symbols"] == ["NQZ6 C25500"]
        assert call.kwargs["stype_in"] == "raw_symbol"
    assert "NQZ6 C25500" in s._option_symbols
    assert reported == [universe]


def test_resolver_outage_keeps_definition_acquisition_pending(monkeypatch) -> None:
    s = _streamer(set(), set())
    s._option_symbols = {"NQZ6 C25000"}
    s._option_session_date = date(2026, 9, 29)
    s._definition_acquirer = Mock()
    s._resolve_option_universe = lambda: None
    _hold_session(monkeypatch, s._option_session_date)
    client = Mock()

    s._refresh_option_subscriptions(client)

    client.subscribe.assert_not_called()
    client.stop.assert_not_called()
    assert set(s._definition_acquirer.submit.call_args.args[0]) == s._option_symbols
