import queue
import threading
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from databento_dbn import FIXED_PRICE_SCALE, InstrumentClass, StatUpdateAction

from src.acquisition.live_streamer import (
    LiveStreamer,
    _enum_int,
    _instrument_class,
    _parse_resolved_weekly_option_symbol,
    subscribe_live_feeds,
)
from src.acquisition.options_universe import (
    LIVE_OPTION_CATEGORIES,
    NQ_END_OF_MONTH_OPTION_PARENT,
    NQ_OPTION_PARENTS,
    NQ_QUARTERLY_OPTION_PARENT,
    NQ_WEEKLY_OPTION_PARENTS,
    live_option_session_date,
    resolve_session_option_universe,
    root_for,
    session_option_targets,
)

pytestmark = pytest.mark.acquisition


def test_nq_option_parents_cover_backfill_universe() -> None:
    assert NQ_QUARTERLY_OPTION_PARENT == "NQ.OPT"
    assert NQ_END_OF_MONTH_OPTION_PARENT == "QNE.OPT"
    assert len(NQ_WEEKLY_OPTION_PARENTS) == 24
    assert len(NQ_OPTION_PARENTS) == 26
    assert len(set(NQ_OPTION_PARENTS)) == len(NQ_OPTION_PARENTS)

    expected_weeklies = {
        *(f"Q{week}{weekday}.OPT" for week in range(1, 6) for weekday in "ABCD"),
        *(f"QN{week}.OPT" for week in range(1, 5)),
    }
    assert set(NQ_WEEKLY_OPTION_PARENTS) == expected_weeklies


def test_session_option_targets_friday_five_chains() -> None:
    targets = {t.category: t for t in session_option_targets(date(2026, 8, 14))}
    assert targets["0dte"].root == "QN2Q6" and targets["0dte"].parent == "QN2.OPT"
    assert targets["1dte"].root == "Q3AQ6" and targets["1dte"].expiry == date(2026, 8, 17)
    assert targets["qne1"].root == "QNEQ6" and targets["qne1"].expiry == date(2026, 8, 31)
    assert targets["qne2"].root == "QNEU6" and targets["qne2"].expiry == date(2026, 9, 30)
    assert targets["qtr"].root == "NQU6" and targets["qtr"].expiry == date(2026, 9, 18)


def test_root_for_month_end_is_qne_not_weekly() -> None:
    assert root_for(date(2026, 4, 30)) == "QNEJ6"
    assert root_for(date(2026, 9, 18)) == "NQU6"


def test_month_end_session_does_not_collapse_qne1_onto_0dte() -> None:
    targets = {t.category: t for t in session_option_targets(date(2026, 4, 30))}
    assert targets["0dte"].root == "QNEJ6"
    assert targets["qne1"].expiry == date(2026, 5, 29)
    assert targets["qne1"].root != targets["0dte"].root


def test_live_option_session_date_skips_weekend_to_monday() -> None:
    saturday = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)
    assert live_option_session_date(saturday) == date(2026, 8, 17)


def test_resolve_session_option_universe_filters_to_five_roots() -> None:
    def fake_resolve(day, parent):
        if parent == "QN2.OPT":
            return {
                "QN2Q6": {"QN2Q6 C100": 1, "QN2Q6 P100": 2},
                "QN2U6": {"QN2U6 C100": 99},  # extra month — must be dropped
            }
        if parent == "Q3A.OPT":
            return {"Q3AQ6": {"Q3AQ6 C100": 3}}
        if parent == "QNE.OPT":
            return {
                "QNEQ6": {"QNEQ6 C100": 4},
                "QNEU6": {"QNEU6 C100": 5},
                "QNEV6": {"QNEV6 C100": 98},
            }
        if parent == "NQ.OPT":
            return {
                "NQU6": {"NQU6 C100": 6},
                "NQZ6": {"NQZ6 C100": 97},
            }
        return {}

    uni = resolve_session_option_universe(
        date(2026, 8, 14), resolve_parent=fake_resolve
    )
    assert uni.complete
    assert set(uni.symbols) == {
        "QN2Q6 C100", "QN2Q6 P100", "Q3AQ6 C100",
        "QNEQ6 C100", "QNEU6 C100", "NQU6 C100",
    }
    assert "QN2U6 C100" not in uni.symbols
    assert "QNEV6 C100" not in uni.symbols
    assert "NQZ6 C100" not in uni.symbols


def test_subscribe_live_feeds_uses_raw_symbol_option_strikes() -> None:
    client = Mock()
    strikes = ["QN2Q6 C24000", "QN2Q6 P24000", "NQU6 C24000"]

    subscribe_live_feeds(client, option_symbols=strikes)

    option_calls = [
        call.kwargs
        for call in client.subscribe.call_args_list
        if call.kwargs["schema"] in {"bbo-1s", "definition"}
        and call.kwargs.get("stype_in") == "raw_symbol"
    ]
    assert {call["schema"] for call in option_calls} == {"bbo-1s", "definition"}
    assert all(call["symbols"] == strikes for call in option_calls)
    assert all(call["dataset"] == "GLBX.MDP3" for call in option_calls)

    parent_option = [
        call.kwargs
        for call in client.subscribe.call_args_list
        if call.kwargs.get("stype_in") == "parent"
        and call.kwargs["schema"] in {"bbo-1s", "definition"}
        and call.kwargs["symbols"] == NQ_OPTION_PARENTS
    ]
    assert parent_option == []

    client.subscribe.assert_any_call(
        dataset="GLBX.MDP3", schema="bbo-1s", symbols="NQ.FUT", stype_in="parent"
    )
    client.subscribe.assert_any_call(
        dataset="GLBX.MDP3", schema="tbbo", symbols="ES.FUT", stype_in="parent"
    )


def test_subscribe_live_feeds_omits_options_when_universe_empty() -> None:
    client = Mock()
    subscribe_live_feeds(client, option_symbols=[])
    schemas = [call.kwargs["schema"] for call in client.subscribe.call_args_list]
    assert "definition" not in schemas
    assert schemas.count("bbo-1s") == 1  # NQ.FUT only


def test_option_refresh_subscribes_new_strikes_only(monkeypatch) -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._option_symbols = {"QN2Q6 C100"}
    streamer._option_session_date = date(2026, 8, 14)
    streamer._session_stop = threading.Event()
    # The refresh now also checks definition presence, which reads the
    # bootstrap's in-memory view of databento.definitions. Treat every resolved
    # symbol as having a row so this test stays about the subscription diff.
    streamer._definitions_view_degraded = False
    streamer._known_symbols = {
        "QN2Q6 C100", "QN2Q6 P100", "Q3AQ6 C100", "QNEQ6 C100",
        "QNEU6 C100", "NQU6 C100",
    }
    streamer._synthetic_symbols = set()
    monkeypatch.setattr(
        "src.acquisition.live_streamer.cme_weekend_closed",
        lambda _now=None: False,
    )
    monkeypatch.setattr(
        "src.acquisition.live_streamer.live_option_session_date",
        lambda _now=None: date(2026, 8, 14),
    )

    def fake_resolve(day, parent):
        if parent == "QN2.OPT":
            return {"QN2Q6": {"QN2Q6 C100": 1, "QN2Q6 C101": 2}}
        if parent == "Q3A.OPT":
            return {"Q3AQ6": {"Q3AQ6 C100": 3}}
        if parent == "QNE.OPT":
            return {"QNEQ6": {"QNEQ6 C100": 4}, "QNEU6": {"QNEU6 C100": 5}}
        if parent == "NQ.OPT":
            return {"NQU6": {"NQU6 C100": 6}}
        return {}

    streamer._resolve_option_universe = lambda: resolve_session_option_universe(
        date(2026, 8, 14), resolve_parent=fake_resolve
    )
    client = Mock()
    streamer._refresh_option_subscriptions(client)
    added = [
        call.kwargs["symbols"]
        for call in client.subscribe.call_args_list
        if call.kwargs.get("stype_in") == "raw_symbol"
    ]
    assert added
    assert "QN2Q6 C101" in added[0]
    assert "QN2Q6 C100" not in added[0]
    assert "QN2Q6 C101" in streamer._option_symbols


def test_option_refresh_rolls_the_session_without_stopping_the_client(monkeypatch) -> None:
    """A session roll keeps the connection and subscribes the new chains.

    Replaces the previous contract, which tore the client down on every roll.
    That cost 10-50s of unrecoverable quotes four times a week (measured 25s,
    50s, 42s, 10s across the 2026-09-28 week) because the roll was at Chicago
    midnight, inside live Asian-session trading, and the Databento client
    replays no live gap. The roll is now at 16:05 CT inside the dead window —
    see tests/test_globex_session_date.py.
    """
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._option_symbols = set()
    streamer._option_session_date = date(2000, 1, 3)
    streamer._greeks_enabled = False
    streamer._archive = None
    streamer._known_symbols = set()
    streamer._synthetic_symbols = set()
    streamer._definitions_view_degraded = True   # suppress the gap report here
    monkeypatch.setattr(
        "src.acquisition.live_streamer.cme_weekend_closed",
        lambda _now=None: False,
    )
    streamer._resolve_option_universe = lambda: SimpleNamespace(
        symbols=("NQZ6 C25000",), session_date=date(2026, 10, 6),
        missing=(), complete=True,
    )
    client = Mock()
    streamer._refresh_option_subscriptions(client)

    client.stop.assert_not_called()
    # The new session's date is adopted, not deferred to a reconnect.
    assert streamer._option_session_date != date(2000, 1, 3)
    # And its chains are subscribed on the live connection.
    schemas = [c.kwargs["schema"] for c in client.subscribe.call_args_list]
    assert schemas == ["definition", "bbo-1s"]
    assert streamer._option_symbols, "new chains must be recorded as subscribed"


def test_weekend_close_is_the_only_thing_that_stops_the_client(monkeypatch) -> None:
    """The weekly session ends at the CME Friday close, nothing else."""
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._option_symbols = set()
    streamer._option_session_date = date(2026, 8, 21)
    monkeypatch.setattr(
        "src.acquisition.live_streamer.cme_weekend_closed",
        lambda _now=None: True,
    )
    client = Mock()
    streamer._refresh_option_subscriptions(client)
    client.stop.assert_called_once()
    client.subscribe.assert_not_called()


def test_option_refresh_stops_client_when_cme_weekend_closed(monkeypatch) -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._option_symbols = set()
    streamer._option_session_date = date(2026, 8, 21)
    monkeypatch.setattr(
        "src.acquisition.live_streamer.cme_weekend_closed",
        lambda _now=None: True,
    )
    client = Mock()
    streamer._refresh_option_subscriptions(client)
    client.stop.assert_called_once()
    client.subscribe.assert_not_called()


def test_start_skips_when_cme_weekend_closed(monkeypatch) -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    monkeypatch.setattr(
        "src.acquisition.live_streamer.cme_weekend_closed",
        lambda _now=None: True,
    )
    monkeypatch.setattr("src.acquisition.live_streamer.close_pool", lambda: None)
    ran = []
    streamer._run_session = lambda: ran.append(True)
    streamer.start()
    assert ran == []


def test_live_archive_date_is_session_not_utc() -> None:
    # Saturday 07:50 UTC = Saturday 02:50 CDT → next trading day Monday.
    saturday_utc = datetime(2026, 8, 22, 7, 50, tzinfo=timezone.utc)
    assert live_option_session_date(saturday_utc) == date(2026, 8, 24)
    friday_rth = datetime(2026, 8, 21, 18, 0, tzinfo=timezone.utc)
    assert live_option_session_date(friday_rth) == date(2026, 8, 21)


def test_definition_refreshes_reused_instrument_id_without_duplicate_insert() -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._sym_cache = {42: "STALE HISTORICAL SYMBOL"}
    streamer._known_symbols = {"NQ1D6 C30000"}
    streamer._opt_meta = {}
    streamer._0dte_ids = set()
    streamer._greeks_enabled = False
    streamer._pool = Mock()

    definition = SimpleNamespace(
        instrument_id=42,
        raw_symbol="NQ1D6 C30000",
        instrument_class="C",
        strike_price=30_000_000_000_000,
        expiration=1_786_035_600_000_000_000,
        ts_event=1_786_000_000_000_000_000,
        underlying="NQU6",
        exchange="XCME",
        currency="USD",
        min_price_increment=250_000_000,
        multiplier=20_000_000_000,
    )

    streamer._upsert_definition(definition)

    assert streamer._sym_cache[42] == "NQ1D6 C30000"
    streamer._pool.connection.assert_not_called()


def test_instrument_class_normalizes_databento_wrapper() -> None:
    assert _instrument_class(InstrumentClass.CALL) == "C"
    assert _instrument_class(InstrumentClass.PUT) == "P"
    assert _instrument_class("F") == "F"
    assert _instrument_class(None) is None


def test_integer_enum_normalizes_databento_wrapper() -> None:
    assert _enum_int(StatUpdateAction.NEW) == 1
    assert _enum_int(2) == 2
    assert _enum_int(None) is None


def test_definition_bootstrap_loads_symbols_without_historical_ids() -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._pool = MagicMock()
    connection = streamer._pool.connection.return_value.__enter__.return_value
    connection.execute.return_value.fetchall.return_value = [
        ("NQ1D6 C30000", "databento"),
        ("NQ1D6 P30000", "databento"),
        ("Q1DQ6 C29580", "synthetic"),
    ]

    streamer._bootstrap_definitions()

    # Vendor rows suppress a redundant vendor write; synthetic rows must NOT,
    # or the vendor definition could never correct them.
    assert streamer._known_symbols == {"NQ1D6 C30000", "NQ1D6 P30000"}
    assert streamer._synthetic_symbols == {"Q1DQ6 C29580"}
    assert streamer._sym_cache == {}
    assert streamer._opt_meta == {}


def test_insert_bbo_1s_queues_option_quote_while_vendor_metadata_is_pending() -> None:
    scale = int(FIXED_PRICE_SCALE)
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._bbo_queue = queue.Queue(maxsize=4)
    streamer._bbo_writer = None
    streamer._bbo_writer_exc = None
    streamer._bbo_enqueue_timeout_s = 0.01
    streamer._bbo_backpressure_last_log_s = 0.0
    streamer._resolve_symbol = Mock(return_value="Q1DQ6 C29580")
    streamer._greeks_enabled = True
    streamer._on_bbo_for_greeks = Mock()

    ts = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    rec = SimpleNamespace(
        ts_recv=int(ts.timestamp() * 1_000_000_000),
        instrument_id=42,
        price=101 * scale,
        size=9,
        levels=[SimpleNamespace(bid_px=100 * scale, ask_px=102 * scale, bid_sz=5, ask_sz=7)],
    )

    streamer._insert_bbo_1s(rec)

    row = streamer._bbo_queue.get_nowait()
    assert row[1:] == (42, "Q1DQ6 C29580", 101.0, 9, 100.0, 102.0, 5, 7)

    streamer._on_bbo_for_greeks.assert_not_called()
    # A quote without authoritative metadata is retained but cannot create
    # reconstructed metadata or reach the option Greeks calculation.
    assert 42 not in getattr(streamer, "_opt_meta", {})
    assert 42 not in getattr(streamer, "_0dte_ids", set())


def test_enqueue_bbo_row_raises_when_writer_failed() -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._bbo_queue = queue.Queue(maxsize=1)
    streamer._bbo_writer = None
    streamer._bbo_writer_exc = RuntimeError("writer crashed")
    streamer._bbo_enqueue_timeout_s = 0.01
    streamer._bbo_backpressure_last_log_s = 0.0

    with pytest.raises(RuntimeError, match="bbo_1s_writer_failed"):
        streamer._enqueue_bbo_row((None, 1, "SYM", 1.0, 1, 1.0, 1.1, 1, 1))


def test_bbo_table_conflict_probe_reads_catalog_bool() -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._pool = MagicMock()
    conn = streamer._pool.connection.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = (True,)
    assert streamer._bbo_table_has_conflict_arbiter() is True
    conn.execute.return_value.fetchone.return_value = (False,)
    assert streamer._bbo_table_has_conflict_arbiter() is False


def test_bbo_writer_loop_flushes_pending_batch_on_stop_token() -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._pool = MagicMock()
    conn = streamer._pool.connection.return_value.__enter__.return_value
    conn.execute = Mock()
    conn.commit = Mock()
    conn.rollback = Mock()
    streamer._bbo_queue = queue.Queue()
    streamer._bbo_stop_token = object()
    streamer._bbo_batch_size = 10
    streamer._bbo_flush_interval_s = 60.0
    streamer._bbo_write_mode = "insert"
    streamer._bbo_insert_sql_cache = {}
    streamer._bbo_writer_exc = None
    streamer._client = None

    ts = datetime.now(timezone.utc)
    row1 = (ts, 1, "SYM1", 1.0, 1, 1.0, 1.1, 1, 2)
    row2 = (ts, 2, "SYM2", 2.0, 2, 2.0, 2.1, 3, 4)
    streamer._bbo_queue.put(row1)
    streamer._bbo_queue.put(row2)
    streamer._bbo_queue.put(streamer._bbo_stop_token)

    streamer._bbo_writer_loop()

    assert conn.execute.call_count == 1
    sql, params = conn.execute.call_args.args
    assert "INSERT INTO databento.bbo_1s" in sql
    assert "ON CONFLICT DO NOTHING" in sql
    assert params == [*row1, *row2]
    conn.commit.assert_called_once()


def test_start_bbo_writer_chooses_mode_from_constraints(monkeypatch) -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._bbo_table_has_conflict_arbiter = Mock(return_value=False)
    streamer._bbo_batch_size = 2_000
    streamer._bbo_queue = queue.Queue(maxsize=100)
    streamer._bbo_flush_interval_s = 0.25
    streamer._bbo_insert_sql_cache = {}
    streamer._bbo_writer_exc = None
    streamer._bbo_writer_loop = Mock()
    fake_thread = Mock()
    fake_thread.start = Mock()
    thread_ctor = Mock(return_value=fake_thread)
    monkeypatch.setattr("src.acquisition.live_streamer.threading.Thread", thread_ctor)

    streamer._start_bbo_writer()

    assert streamer._bbo_write_mode == "copy"
    thread_ctor.assert_called_once()
    fake_thread.start.assert_called_once()


def test_parse_resolved_weekly_option_symbol_examples() -> None:
    parsed = _parse_resolved_weekly_option_symbol(
        "Q1DQ6 C29580",
        reference=date(2026, 8, 6),
    )
    assert parsed is not None
    assert parsed["instrument_class"] == "C"
    assert parsed["strike_price"] == 29580.0
    assert parsed["expiry"] == date(2026, 8, 6)
    assert parsed["expiration"].tzinfo == timezone.utc

    parsed_qn = _parse_resolved_weekly_option_symbol(
        "QN1Q6 P29580",
        reference=date(2026, 8, 6),
    )
    assert parsed_qn is not None
    assert parsed_qn["instrument_class"] == "P"
    assert parsed_qn["expiry"] == date(2026, 8, 7)

    parsed_q2a = _parse_resolved_weekly_option_symbol(
        "Q2AQ6 C29580",
        reference=date(2026, 8, 6),
    )
    assert parsed_q2a is not None
    assert parsed_q2a["expiry"] == date(2026, 8, 10)

    assert _parse_resolved_weekly_option_symbol("NQZ6 C29580", reference=date(2026, 8, 6)) is None
    assert _parse_resolved_weekly_option_symbol("Q1DQ6 X29580", reference=date(2026, 8, 6)) is None


def test_bbo_writer_preserves_quotes_without_synthetic_definition_writes() -> None:
    scale = int(FIXED_PRICE_SCALE)
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._pool = MagicMock()
    conn = streamer._pool.connection.return_value.__enter__.return_value
    conn.execute = Mock()
    conn.commit = Mock()
    conn.rollback = Mock()
    streamer._bbo_queue = queue.Queue()
    streamer._bbo_stop_token = object()
    streamer._bbo_batch_size = 10
    streamer._bbo_flush_interval_s = 60.0
    streamer._bbo_write_mode = "insert"
    streamer._bbo_insert_sql_cache = {}
    streamer._definition_insert_sql_cache = {}
    streamer._bbo_writer = None
    streamer._bbo_writer_exc = None
    streamer._client = None
    streamer._bbo_enqueue_timeout_s = 0.01
    streamer._bbo_backpressure_last_log_s = 0.0
    streamer._resolve_symbol = Mock(side_effect=["Q1DQ6 C29580", "NQZ6 C29580"])
    streamer._greeks_enabled = False
    streamer._known_symbols = set()
    streamer._opt_meta = {}
    streamer._0dte_ids = set()
    streamer._pending_weekly_definitions = {}

    ts = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    rec1 = SimpleNamespace(
        ts_recv=int(ts.timestamp() * 1_000_000_000),
        instrument_id=1001,
        price=101 * scale,
        size=9,
        levels=[SimpleNamespace(bid_px=100 * scale, ask_px=102 * scale, bid_sz=5, ask_sz=7)],
    )
    rec2 = SimpleNamespace(
        ts_recv=int(ts.timestamp() * 1_000_000_000),
        instrument_id=1002,
        price=201 * scale,
        size=4,
        levels=[SimpleNamespace(bid_px=200 * scale, ask_px=202 * scale, bid_sz=2, ask_sz=3)],
    )
    streamer._insert_bbo_1s(rec1)
    streamer._insert_bbo_1s(rec2)
    streamer._bbo_queue.put(streamer._bbo_stop_token)

    streamer._bbo_writer_loop()

    assert conn.execute.call_count == 1
    bbo_sql, bbo_params = conn.execute.call_args.args
    assert "INSERT INTO databento.bbo_1s" in bbo_sql
    assert "Q1DQ6 C29580" in bbo_params
    assert "NQZ6 C29580" in bbo_params
    assert streamer._pending_weekly_definitions == {}
    assert "Q1DQ6 C29580" not in getattr(streamer, "_synthetic_symbols", set())
    assert streamer._opt_meta == {}

# ── consumer side: the feature engine must scope to the same five chains ──


def test_feature_engine_chain_roots_match_streamer_subscription() -> None:
    """engine._session_chain_roots must equal the streamer's subscribed roots.

    This is the producer/consumer contract from ADR-034. If it ever drifts, the
    feature engine either starves (asks for chains nobody captures) or over-reads
    (pulls chains no feature consumes, which is what caused the 3.2x over-read).
    """
    from src.features.engine import FeatureEngine

    for d in (date(2026, 8, 13), date(2026, 8, 20), date(2026, 9, 18)):
        subscribed = sorted({t.root for t in session_option_targets(d)})
        consumed = FeatureEngine._session_chain_roots(d)
        assert consumed == subscribed, f"drift on {d}: {consumed} != {subscribed}"
        assert len(consumed) == len(set(consumed))


def test_feature_engine_chain_roots_are_five_categories() -> None:
    from src.features.engine import FeatureEngine

    roots = FeatureEngine._session_chain_roots(date(2026, 8, 20))
    # 0DTE, 1DTE, QNE1, QNE2, front quarterly — QNE1/QNE2 never collapse, but
    # 0DTE/1DTE can coincide with a QNE or quarterly expiry, so assert <= 5.
    assert 1 <= len(roots) <= len(LIVE_OPTION_CATEGORIES)
    assert roots == sorted(set(roots))


def test_feature_engine_chain_roots_empty_on_non_trading_day() -> None:
    """Weekends/holidays yield [] so the loader falls back rather than starving."""
    from src.features.engine import FeatureEngine

    assert FeatureEngine._session_chain_roots(date(2026, 8, 15)) == []  # Saturday
    assert FeatureEngine._session_chain_roots(date(2026, 8, 16)) == []  # Sunday


# ---------------------------------------------------------------------------
# Underlying-future resolution and settle time (ADR-151)
#
# Regression guard: `underlying` was previously derived as "NQ" + the OPTION's
# own month code, which mislabelled every serial/post-roll expiry and invented
# contracts that have never existed (NQ futures are quarterly-only). `expiration`
# was hardcoded to 21:00 UTC, an hour late for the whole CDT half of the year.
# ---------------------------------------------------------------------------

def test_weekly_option_settles_into_next_quarterly_after_the_roll() -> None:
    """A Sep-21 daily outlives NQU6 (expired Sep 18) and settles into NQZ6."""
    parsed = _parse_resolved_weekly_option_symbol(
        "Q3AU6 C25000", reference=date(2026, 9, 18)
    )
    assert parsed is not None
    assert parsed["expiry"] == date(2026, 9, 21)
    assert parsed["underlying"] == "NQZ6"


def test_weekly_option_before_the_roll_keeps_the_front_quarterly() -> None:
    parsed = _parse_resolved_weekly_option_symbol(
        "Q1BU6 C25000", reference=date(2026, 9, 1)
    )
    assert parsed is not None
    assert parsed["expiry"] == date(2026, 9, 1)
    assert parsed["underlying"] == "NQU6"


@pytest.mark.parametrize(
    ("symbol", "reference", "expected"),
    [
        ("Q1DQ6 C25000", date(2026, 8, 6), "NQU6"),   # August serial -> Sep future
        ("Q1DV6 C25000", date(2026, 10, 1), "NQZ6"),  # October serial -> Dec future
        ("Q1DF7 C25000", date(2027, 1, 6), "NQH7"),   # January serial -> Mar future
    ],
)
def test_serial_months_never_resolve_to_a_nonexistent_future(
    symbol: str, reference: date, expected: str
) -> None:
    """NQ lists only H/M/U/Z; NQQ6 and NQV6 are not tradeable contracts."""
    parsed = _parse_resolved_weekly_option_symbol(symbol, reference=reference)
    assert parsed is not None
    assert parsed["underlying"] == expected
    assert parsed["underlying"][2] in "HMUZ"


def test_expiration_is_1500_ct_across_both_dst_regimes() -> None:
    """15:00 CT is 20:00 UTC under CDT and 21:00 UTC under CST."""
    cdt = _parse_resolved_weekly_option_symbol(
        "Q1BU6 C25000", reference=date(2026, 9, 1)
    )
    assert cdt is not None
    assert cdt["expiration"] == datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc)

    cst = _parse_resolved_weekly_option_symbol(
        "Q1BZ6 C25000", reference=date(2026, 12, 1)
    )
    assert cst is not None
    assert cst["expiration"] == datetime(2026, 12, 1, 21, 0, tzinfo=timezone.utc)


def test_weekly_expiring_on_quarterly_friday_rolls_to_next_quarterly() -> None:
    """The future settles 08:30 CT; a 15:00 CT weekly that day outlives it."""
    parsed = _parse_resolved_weekly_option_symbol(
        "QN3U6 C25000", reference=date(2026, 9, 14)
    )
    assert parsed is not None
    assert parsed["expiry"] == date(2026, 9, 18)
    assert parsed["underlying"] == "NQZ6"


def test_definition_upsert_never_clobbers_usable_contract_terms() -> None:
    """Vendor definitions carry no usable contract terms for these options.

    `multiplier` is NULL and `min_price_increment` is NaN or a leaked DBN
    UNDEF_PRICE sentinel on 100% of vendor option rows, so a blind
    `SET col = EXCLUDED.col` would replace correct reconstructed values (20,
    0.25) with garbage on every synthetic -> vendor upgrade.
    """
    from src.acquisition.live_streamer import _DEF_UPSERT_TAIL

    sql = " ".join(_DEF_UPSERT_TAIL.split())

    # Guarded columns must not be assigned straight from EXCLUDED.
    assert "multiplier = EXCLUDED.multiplier" not in sql
    assert "min_price_increment = EXCLUDED.min_price_increment" not in sql
    assert "COALESCE(EXCLUDED.multiplier" in sql
    # NaN is never equal to itself -- that self-comparison is the NaN test.
    assert "EXCLUDED.min_price_increment <> EXCLUDED.min_price_increment" in sql

    # Directionality: vendor overwrites synthetic, never the reverse.
    assert "databento.definitions.source = 'synthetic'" in sql
    assert "EXCLUDED.source <> 'synthetic'" in sql


def test_vendor_metadata_binds_to_current_live_id_before_greeks() -> None:
    from src.acquisition.definition_acquisition import VendorDefinition

    scale = int(FIXED_PRICE_SCALE)
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._bbo_queue = queue.Queue(maxsize=4)
    streamer._bbo_writer = None
    streamer._bbo_writer_exc = None
    streamer._bbo_enqueue_timeout_s = 0.01
    streamer._bbo_backpressure_last_log_s = 0.0
    streamer._greeks_enabled = True
    streamer._on_bbo_for_greeks = Mock()
    streamer._opt_meta = {}
    streamer._0dte_ids = set()
    streamer._sym_cache = {42: "Q1CV6 P25000"}
    streamer._resolve_symbol = Mock(return_value="Q1CV6 P25000")
    streamer._definition_acquirer = Mock()
    definitions = {
        "Q1CV6 P25000": VendorDefinition.from_row((
            datetime(2026, 10, 4, 13, 43, 29, tzinfo=timezone.utc),
            42648901, "Q1CV6 P25000", "P", 25000.0,
            datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc),
            "NQZ6", "XCME", "USD", None, 20.0, "databento",
        )),
    }
    streamer._definition_acquirer.get.side_effect = definitions.get
    ts = datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc)
    rec = SimpleNamespace(
        ts_recv=int(ts.timestamp() * 1e9), instrument_id=42,
        price=101 * scale, size=9,
        levels=[SimpleNamespace(bid_px=100 * scale, ask_px=102 * scale, bid_sz=5, ask_sz=7)],
    )

    streamer._insert_bbo_1s(rec)

    assert streamer._sym_cache == {42: "Q1CV6 P25000"}
    assert 42648901 not in streamer._opt_meta
    assert streamer._opt_meta[42]["underlying"] == "NQZ6"
    assert streamer._opt_meta[42]["strike"] == 25000.0
    assert streamer._opt_meta[42]["is_call"] is False
    assert 42 in streamer._0dte_ids
    streamer._on_bbo_for_greeks.assert_called_once()


def test_live_id_reassignment_clears_stale_option_metadata() -> None:
    streamer = LiveStreamer.__new__(LiveStreamer)
    streamer._sym_cache = {42: "Q1CV6 P25000"}
    streamer._opt_meta = {42: {"symbol": "Q1CV6 P25000"}}
    streamer._0dte_ids = {42}
    streamer._greeks_buf = {42: (100.0, 102.0)}
    streamer._definition_acquirer = Mock()
    streamer._definition_acquirer.get.return_value = None
    rec = SimpleNamespace(instrument_id=42, stype_out_symbol="NQZ6 C30000")

    streamer._handle_symbol_mapping(rec)

    assert streamer._sym_cache[42] == "NQZ6 C30000"
    assert 42 not in streamer._opt_meta
    assert 42 not in streamer._0dte_ids
    assert 42 not in streamer._greeks_buf


def test_future_vendor_metadata_is_withheld_at_quote_time() -> None:
    from src.acquisition.definition_acquisition import VendorDefinition

    streamer = LiveStreamer.__new__(LiveStreamer)
    decision = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)
    symbol = "Q1CV6 P25000"
    streamer._vendor_definitions = {symbol: VendorDefinition.from_row((
        datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc), 999, symbol,
        "P", 25000.0, datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc),
        "NQZ6", "XCME", "USD", None, 20.0, "databento",
    ))}
    assert streamer._bind_option_definition(42, symbol, decision) is False
    assert streamer._opt_meta == {}
    assert 42 not in streamer._0dte_ids


def test_incomplete_live_metadata_does_not_shadow_complete_acquisition() -> None:
    from src.acquisition.definition_acquisition import VendorDefinition

    streamer = LiveStreamer.__new__(LiveStreamer)
    decision = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)
    symbol = "Q1CV6 P25000"
    complete = VendorDefinition.from_row((
        datetime(2026, 10, 4, 13, 43, 29, tzinfo=timezone.utc), 999, symbol,
        "P", 25000.0, datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc),
        "NQZ6", "XCME", "USD", None, 20.0, "databento",
    ))
    incomplete_row = list(complete.row())
    incomplete_row[5] = None
    streamer._vendor_definitions = {symbol: VendorDefinition.from_row(tuple(incomplete_row))}
    streamer._definition_acquirer = Mock()
    streamer._definition_acquirer.get.return_value = complete

    assert streamer._bind_option_definition(42, symbol, decision) is True
    assert streamer._opt_meta[42]["symbol"] == symbol


def test_live_definition_id_reassignment_clears_old_quote_buffer() -> None:
    from datetime import timedelta

    streamer = LiveStreamer.__new__(LiveStreamer)
    now = datetime.now(timezone.utc)
    symbol = "Q1CV6 P25000"
    streamer._known_symbols = {symbol}
    streamer._synthetic_symbols = set()
    streamer._sym_cache = {42: "Q1CV6 C25000"}
    streamer._opt_meta = {42: {"symbol": "Q1CV6 C25000"}}
    streamer._0dte_ids = {42}
    streamer._greeks_buf = {42: (100.0, 102.0)}
    record = SimpleNamespace(
        instrument_id=42, raw_symbol=symbol, instrument_class="P",
        ts_event=int((now - timedelta(minutes=1)).timestamp() * 1e9),
        strike_price=25_000_000_000_000,
        expiration=int((now + timedelta(hours=1)).timestamp() * 1e9),
        underlying="NQZ6", exchange="XCME", currency="USD",
        unit_of_measure_qty=20_000_000_000,
    )

    streamer._upsert_definition(record)

    assert streamer._sym_cache[42] == symbol
    assert streamer._opt_meta[42]["symbol"] == symbol
    assert 42 not in streamer._greeks_buf


def test_future_live_definition_does_not_shadow_eligible_acquired_terms() -> None:
    from dataclasses import replace
    from datetime import timedelta
    from src.acquisition.definition_acquisition import VendorDefinition

    streamer = LiveStreamer.__new__(LiveStreamer)
    decision = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)
    symbol = "Q1CV6 P25000"
    eligible = VendorDefinition.from_row((
        decision - timedelta(days=1), 999, symbol,
        "P", 25000.0, datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc),
        "NQZ6", "XCME", "USD", None, 20.0, "databento",
    ))
    streamer._vendor_definitions = {
        symbol: replace(eligible, ts_event=decision + timedelta(seconds=1)),
    }
    streamer._definition_acquirer = Mock()
    streamer._definition_acquirer.get.return_value = eligible

    assert streamer._bind_option_definition(42, symbol, decision) is True
    assert streamer._opt_meta[42]["symbol"] == symbol
    assert streamer._opt_meta[42]["expiration"] == eligible.expiration
