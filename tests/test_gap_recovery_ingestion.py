"""Bounded raw recovery preserves source events and legacy live DB precision.

All inputs are generated; the default gate neither contacts a vendor nor opens
a database. Actual SQL multiset semantics are also checked separately using
transaction-local PostgreSQL TEMP tables during implementation validation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from types import SimpleNamespace
from unittest.mock import MagicMock
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ingestion.gap_recovery import prepare_frame, _validate_receive_window

pytestmark = pytest.mark.acquisition

BASE_NS = 1_791_304_052_000_000_000


def test_ingest_phase_requires_existing_archives_without_vendor_calls(monkeypatch, tmp_path) -> None:
    from scripts.data import recover_databento_gaps as runner
    from src.acquisition.gap_recovery import RecoveryRequest
    request = RecoveryRequest("missing", "bbo-1s", "raw_symbol",
                              "2026-10-06T16:27:32Z", "2026-10-06T16:27:35Z", ("NQZ6",))
    monkeypatch.setattr(runner, "load_requests", lambda path: [request])
    monkeypatch.setattr(runner, "find_saved_receipt", lambda *args: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("ingest phase must not call the vendor")
    monkeypatch.setattr(runner, "download_requests", forbidden)
    monkeypatch.setattr(runner, "enable_strict_dbn_validation", lambda: None)
    monkeypatch.setattr("sys.argv", ["recover", "--job-dir", str(tmp_path / "job"), "--phase", "ingest"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def raw_frame(schema: str, *, ns: list[int] | None = None) -> pd.DataFrame:
    stamps = ns or [BASE_NS + 653_984_615, BASE_NS + 654_227_617]
    n = len(stamps)
    values = {
        "ts_event": np.asarray(stamps, dtype=np.uint64),
        "instrument_id": [261401] * n,
        "symbol": ["NQZ6"] * n,
        "price": [31532.25] * n,
        "size": [1] * n,
        "bid_px_00": [31532.25] * n,
        "ask_px_00": [31532.5] * n,
        "bid_sz_00": [3] * n,
        "ask_sz_00": [4] * n,
    }
    if schema == "tbbo":
        values.update(side=["A"] * n, action=["T"] * n,
                      flags=[0] * n, sequence=[145038996] * n,
                      bid_ct_00=[2] * n, ask_ct_00=[3] * n,
                      ts_in_delta=[14002] * n)
    elif schema == "statistics":
        values.update(stat_type=[1] * n, quantity=[0] * n,
                      sequence=[0] * n, ts_ref=np.asarray(stamps, dtype=np.uint64),
                      update_action=[0] * n, stat_flags=[0] * n)
    return pd.DataFrame(values, index=pd.Index(np.asarray(stamps, dtype=np.uint64),
                                             name="ts_recv"))


@pytest.mark.parametrize("schema", ["tbbo", "bbo-1s", "statistics"])
def test_source_integer_timestamps_match_existing_live_rounding(schema: str) -> None:
    # A broad deterministic sample detects the epoch-float multiplication bug:
    # rounding (seconds * 1e6) disagrees with datetime.fromtimestamp for ~12%.
    stamps = BASE_NS + np.random.default_rng(3).integers(0, 10**9, 2048)
    frame = raw_frame(schema, ns=list(map(int, stamps)))
    result = prepare_frame(frame, schema)
    expected = pd.DatetimeIndex([
        datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc) for ns in stamps
    ])
    assert list(result["ts_recv"]) == list(expected)
    if schema != "bbo-1s":
        assert list(result["ts_event"]) == list(expected)


@pytest.mark.parametrize("schema", ["tbbo", "bbo-1s", "statistics"])
def test_normalized_payload_keeps_same_timestamp_distinct_events(schema: str) -> None:
    frame = raw_frame(schema, ns=[BASE_NS + 653_984_615] * 2)
    frame.loc[:, "price"] = [31532.25, 31532.5]
    result = prepare_frame(frame, schema)
    assert len(result) == 2
    assert list(result["price"]) == [31532.25, 31532.5]
    assert result["dataset"].eq("GLBX.MDP3").all()


def test_bbo_preserves_parent_spread_symbol_and_omits_generated_column() -> None:
    frame = raw_frame("bbo-1s")
    frame.loc[:, "symbol"] = ["NQZ6", "NQZ6-NQH7"]
    result = prepare_frame(frame, "bbo-1s")
    # PostgreSQL owns GENERATED ALWAYS AS (ask_px - bid_px); COPY cannot write it.
    assert "spread" not in result.columns
    assert list(result["symbol"]) == ["NQZ6", "NQZ6-NQH7"]


def test_statistics_zero_quantity_sequence_and_flags_are_real_values() -> None:
    result = prepare_frame(raw_frame("statistics"), "statistics")
    for name in ("quantity", "sequence", "update_action", "stat_flags"):
        assert result[name].eq(0).all()


def test_statistics_undefined_quantity_and_reference_are_nullable() -> None:
    frame = raw_frame("statistics")
    frame.loc[:, "quantity"] = np.int64(2**63 - 1)
    frame.loc[:, "ts_ref"] = np.uint64(2**64 - 1)
    result = prepare_frame(frame, "statistics")
    assert result["quantity"].isna().all()
    assert result["ts_ref"].isna().all()


def test_statistics_nullable_quantity_keeps_other_64_bit_values_exact() -> None:
    frame = raw_frame("statistics")
    frame.loc[:, "quantity"] = np.asarray([2**63 - 1, 2**53 + 3], dtype=np.int64)
    result = prepare_frame(frame, "statistics")
    assert pd.isna(result["quantity"].iloc[0])
    assert int(result["quantity"].iloc[1]) == 2**53 + 3


def test_sdk_null_prices_remain_null_in_new_rows() -> None:
    frame = raw_frame("bbo-1s")
    frame.loc[:, "price"] = np.nan
    frame.loc[:, "bid_px_00"] = np.nan
    result = prepare_frame(frame, "bbo-1s")
    assert result["price"].isna().all()
    assert result["bid_px"].isna().all()


def test_tbbo_preserves_all_book_counts_and_delta() -> None:
    result = prepare_frame(raw_frame("tbbo"), "tbbo")
    assert result["bid_ct"].eq(2).all()
    assert result["ask_ct"].eq(3).all()
    assert result["ts_in_delta"].eq(14002).all()
    assert result["bid_px"].eq(31532.25).all()


def test_unsupported_schema_fails_before_any_database_connection() -> None:
    with pytest.raises((ValueError, KeyError)):
        prepare_frame(raw_frame("tbbo"), "mbp-10")


@pytest.mark.parametrize("symbol", [None, ""])
def test_unmapped_symbol_refuses_static_instrument_id_fallback(symbol) -> None:
    frame = raw_frame("bbo-1s")
    frame.loc[:, "symbol"] = symbol
    with pytest.raises(ValueError, match="unmapped"):
        prepare_frame(frame, "bbo-1s")


def test_exact_request_right_edge_remains_valid_after_db_rounding() -> None:
    frame = raw_frame("bbo-1s", ns=[BASE_NS + 3_000_000_000 - 1])
    start, end = pd.Timestamp("2026-10-06T16:27:32Z"), pd.Timestamp("2026-10-06T16:27:35Z")
    _validate_receive_window(frame, start, end)
    assert prepare_frame(frame, "bbo-1s")["ts_recv"].iloc[0] == end
    frame = raw_frame("bbo-1s", ns=[BASE_NS + 3_000_000_000])
    with pytest.raises(ValueError, match="outside"):
        _validate_receive_window(frame, start, end)


def test_truncated_sdk_warning_is_an_error_before_merge() -> None:
    from databento.common.error import BentoWarning
    from src.ingestion.gap_recovery import enable_strict_dbn_validation
    # Single-thread fixture context restores the gate's global warning policy.
    with warnings.catch_warnings():
        enable_strict_dbn_validation()
        with pytest.raises(BentoWarning, match="incomplete"):
            warnings.warn(BentoWarning("DBN file is truncated or contains an incomplete record"))


def test_merge_lock_retries_nonblocking_advisory_probes() -> None:
    from src.ingestion.gap_recovery import _wait_for_merge_lock
    cur = MagicMock()
    cur.fetchone.side_effect = [(False,), (False,), (True,)]
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    _wait_for_merge_lock(cur, "gap_recovery:databento.bbo_1s", monotonic=lambda: clock.now,
                         sleep=sleep)
    assert clock.sleeps == [0.15, 0.15]
    assert cur.execute.call_count == 3
    assert all("pg_try_advisory_lock" in call.args[0] for call in cur.execute.call_args_list)
    assert all(call.args[1] == ("gap_recovery:databento.bbo_1s",) for call in cur.execute.call_args_list)


def test_merge_lock_wait_has_deadline_and_periodic_progress() -> None:
    from src.ingestion.gap_recovery import _wait_for_merge_lock
    cur = MagicMock()
    cur.fetchone.return_value = (False,)
    clock = SimpleNamespace(now=0.0)
    progress = MagicMock()

    def sleep(seconds):
        clock.now += seconds

    with pytest.raises(TimeoutError, match="11s"):
        _wait_for_merge_lock(cur, "gap_recovery:databento.bbo_1s", timeout_seconds=11.0,
                             progress=progress, monotonic=lambda: clock.now, sleep=sleep)
    assert clock.now == pytest.approx(11.0)
    assert [call.args[0] for call in progress.call_args_list] == ["waiting_merge", "waiting_merge"]
    assert [call.args[1] for call in progress.call_args_list] == [5, 10]


def recovery_request():
    from src.acquisition.gap_recovery import RecoveryRequest
    return RecoveryRequest(request_id="unit_fixture", schema="bbo-1s", stype_in="raw_symbol",
                           symbols=("NQZ6",), start="2026-10-06T16:27:32Z",
                           end="2026-10-06T16:27:35Z")


@pytest.mark.parametrize("schema,dataset", [("tbbo", "GLBX.MDP3"), ("bbo-1s", "OPRA.PILLAR")])
def test_wrong_archive_metadata_fails_before_opening_database(monkeypatch, tmp_path, schema, dataset) -> None:
    from src.ingestion import gap_recovery as recovery
    monkeypatch.setattr(recovery.db.DBNStore, "from_file", lambda path: SimpleNamespace(
        schema=schema, dataset=dataset))
    connection_factory = MagicMock(side_effect=AssertionError("must not open a connection"))
    with pytest.raises(ValueError, match="schema/dataset"):
        recovery.ingest_recovery_file(tmp_path / "fake.dbn", recovery_request(),
                                      connection_factory=connection_factory)
    connection_factory.assert_not_called()


@pytest.mark.parametrize("bad_case", ["symbol", "timestamp", "truncated"])
def test_unaccepted_source_never_reaches_target_insert(monkeypatch, tmp_path, bad_case) -> None:
    from databento.common.error import BentoWarning
    from src.ingestion import gap_recovery as recovery
    frame = raw_frame("bbo-1s", ns=[BASE_NS + 1_000_000_000])
    if bad_case == "symbol":
        frame.loc[:, "symbol"] = "ESZ6"
    elif bad_case == "timestamp":
        frame.index = pd.Index(np.asarray([BASE_NS + 3_000_000_000], dtype=np.uint64), name="ts_recv")

    def decode(**kwargs):
        yield frame
        if bad_case == "truncated":
            warnings.warn(BentoWarning("DBN file is truncated or contains an incomplete record"))

    monkeypatch.setattr(recovery.db.DBNStore, "from_file", lambda path: SimpleNamespace(
        schema="bbo-1s", dataset="GLBX.MDP3", to_df=decode))
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with warnings.catch_warnings():
        recovery.enable_strict_dbn_validation()
        with pytest.raises((ValueError, BentoWarning)):
            recovery.ingest_recovery_file(tmp_path / "fake.dbn", recovery_request(),
                                          connection_factory=lambda: conn)
    statements = [str(call.args[0]) for call in cur.execute.call_args_list]
    assert not any("INSERT INTO" in statement for statement in statements)
    conn.commit.assert_not_called()


class DefinitionStore:
    schema = "definition"
    dataset = "GLBX.MDP3"

    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)


def definition_record(*, publication=BASE_NS, instrument_id=100, symbol="NQZ6 P30000"):
    return SimpleNamespace(raw_symbol=symbol, ts_event=publication, instrument_id=instrument_id,
                           instrument_class="P", strike_price=30_000_000_000_000,
                           expiration=1_797_622_200_000_000_000, underlying="NQZ6",
                           exchange="XCME", currency="USD", min_price_increment=250_000_000,
                           unit_of_measure_qty=20_000_000_000)


def definition_request():
    from src.acquisition.gap_recovery import RecoveryRequest
    return RecoveryRequest(request_id="definitions_fixture", schema="definition", stype_in="raw_symbol",
                           symbols=("NQZ6 P30000",), start="2026-10-01T00:00:00Z",
                           end="2026-10-06T16:50:00Z")


@pytest.mark.parametrize("invalid", ["metadata", "header_only", "unrequested"])
def test_definition_archive_validation_precedes_any_db_write(monkeypatch, tmp_path, invalid) -> None:
    from scripts.data import recover_databento_gaps as runner
    records = [definition_record()]
    if invalid == "header_only":
        records = [SimpleNamespace(msg="header only")]
    elif invalid == "unrequested":
        records = [definition_record(symbol="Q1CV6 P25000")]
    store = DefinitionStore(records)
    if invalid == "metadata":
        store.dataset = "OPRA.PILLAR"
    monkeypatch.setattr(runner.db.DBNStore, "from_file", lambda path: store)
    connect = MagicMock(side_effect=AssertionError("invalid source must not reach DB"))
    monkeypatch.setattr(runner.psycopg, "connect", connect)
    with pytest.raises(ValueError):
        runner.ingest_definitions(tmp_path / "fake.dbn", definition_request())
    connect.assert_not_called()


def test_definition_recovery_copies_latest_vendor_row_and_shared_guard(monkeypatch, tmp_path) -> None:
    import csv
    import io
    from scripts.data import recover_databento_gaps as runner
    store = DefinitionStore([definition_record(publication=BASE_NS + 2, instrument_id=102),
                             definition_record(publication=BASE_NS, instrument_id=100),
                             definition_record(publication=BASE_NS + 1, instrument_id=101)])
    monkeypatch.setattr(runner.db.DBNStore, "from_file", lambda path: store)
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = []
    conn.execute.return_value.rowcount = 1
    copy = conn.cursor.return_value.copy.return_value.__enter__.return_value
    monkeypatch.setattr(runner.psycopg, "connect", lambda dsn: conn)
    receipt = runner.ingest_definitions(tmp_path / "fake.dbn", definition_request())
    rows = list(csv.reader(io.StringIO(copy.write.call_args.args[0])))
    assert receipt["records"] == 3
    assert receipt["symbols"] == 1
    assert receipt["absent_from_delivery"] == []
    assert rows[0][1] == "102"
    assert rows[0][10] == "20.0"
    assert rows[0][11] == "databento"
    statements = [call.args[0].as_string() if hasattr(call.args[0], "as_string") else str(call.args[0])
                  for call in conn.execute.call_args_list]
    assert any(runner.DEFINITION_UPSERT_TAIL in statement for statement in statements)


def test_definition_decode_prefix_with_truncated_tail_never_writes_database(monkeypatch, tmp_path) -> None:
    from databento.common.error import BentoWarning
    from scripts.data import recover_databento_gaps as runner

    class TruncatedDefinitions(DefinitionStore):
        def __iter__(self):
            yield definition_record()
            warnings.warn(BentoWarning("DBN file is truncated or contains an incomplete record"))

    monkeypatch.setattr(runner.db.DBNStore, "from_file", lambda path: TruncatedDefinitions([]))
    connect = MagicMock(side_effect=AssertionError("prefix must not be accepted"))
    monkeypatch.setattr(runner.psycopg, "connect", connect)
    with warnings.catch_warnings():
        runner.enable_strict_dbn_validation()
        with pytest.raises(BentoWarning):
            runner.ingest_definitions(tmp_path / "fake.dbn", definition_request())
    connect.assert_not_called()


@pytest.mark.skipif(os.environ.get("VTECH_RECOVERY_TEMP_DB_TEST") != "1",
                    reason="Explicit TEMP-table SQL integration opt-in only")
@pytest.mark.parametrize("legacy_undefined_price", [False, True])
@pytest.mark.parametrize("near_right_boundary", [False, True])
def test_real_sql_reconciles_repeated_events_across_frames_and_retry(monkeypatch, tmp_path,
                                                                   legacy_undefined_price,
                                                                   near_right_boundary) -> None:
    """Exercise real EXCEPT ALL; only session-local TEMP tables are touched.

    Necessary because TBBO/BBO lack unique keys, and mocked cursor rowcounts
    cannot verify SQL multiplicity, NULL equality, or generated-column handling.
    """
    import psycopg
    from src.common.db import get_dsn
    from src.ingestion import gap_recovery as recovery

    raw_ns = BASE_NS + (3_000_000_000 - 1 if near_right_boundary else 1_000_000_000)
    frame_a = raw_frame("bbo-1s", ns=[raw_ns])
    for field in ("price", "bid_px_00", "ask_px_00"):
        frame_a[field] = (frame_a[field] * 1e9).astype("int64")
    if legacy_undefined_price:
        frame_a.loc[:, "price"] = np.int64(2**63 - 1)
    frame_b = frame_a.copy()
    frame_b.loc[:, "price"] = 31_532_500_000_000
    frames = [frame_a.copy(), pd.concat([frame_a.copy(), frame_b])]
    monkeypatch.setattr(recovery.db.DBNStore, "from_file", lambda path: SimpleNamespace(
        schema="bbo-1s", dataset="GLBX.MDP3",
        to_df=lambda **kwargs: iter([frame.copy() for frame in frames])))
    request = recovery.RecoveryRequest(request_id="temp_multiset", schema="bbo-1s",
                   stype_in="raw_symbol", symbols=("NQZ6",),
                   start="2026-10-06T16:27:32Z", end="2026-10-06T16:27:35Z")
    path = tmp_path / "recovery.dbn"
    path.write_bytes(b"fake store is generated by the test")
    with psycopg.connect(get_dsn()) as conn:
        conn.execute("CREATE TEMP TABLE recovery_target (ts_recv timestamptz, "
                     "instrument_id integer, symbol text, price float8, size integer, "
                     "bid_px float8, ask_px float8, bid_sz integer, ask_sz integer, "
                     "spread float8 GENERATED ALWAYS AS (ask_px - bid_px) STORED, "
                     "dataset text DEFAULT 'GLBX.MDP3')")
        conn.execute("INSERT INTO recovery_target (ts_recv,instrument_id,symbol,price,size,"
                     "bid_px,ask_px,bid_sz,ask_sz) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                     (datetime.fromtimestamp(raw_ns / 1e9, timezone.utc),261401,"NQZ6",
                      recovery.LEGACY_UNDEF_PRICE if legacy_undefined_price else 31532.25,
                      1,31532.25,31532.5,3,4))
        conn.commit()

        @contextmanager
        def connection_factory():
            yield conn

        first = recovery.ingest_recovery_file(path, request, connection_factory=connection_factory,
                                             target_override="pg_temp.recovery_target", chunk_size=1)
        second = recovery.ingest_recovery_file(path, request, connection_factory=connection_factory,
                                              target_override="pg_temp.recovery_target", chunk_size=1)
        assert first.inserted == 2
        assert second.inserted == 0
        result = conn.execute("SELECT price,count(*) FROM recovery_target GROUP BY price ORDER BY price").fetchall()
        if legacy_undefined_price:
            assert result == [(31532.5, 1), (recovery.LEGACY_UNDEF_PRICE, 1), (None, 1)]
        else:
            assert result == [(31532.25, 2), (31532.5, 1)]
        assert conn.execute("SELECT spread FROM recovery_target").fetchall() == [(0.25,)] * 3


@pytest.mark.skipif(os.environ.get("VTECH_RECOVERY_TEMP_DB_TEST") != "1",
                    reason="Explicit TEMP-table SQL integration opt-in only")
@pytest.mark.parametrize("conflicting_payload", [False, True])
def test_real_statistics_legacy_sentinels_match_and_conflicts_are_visible(monkeypatch, tmp_path,
                                                                       conflicting_payload) -> None:
    import psycopg
    from src.common.db import get_dsn
    from src.ingestion import gap_recovery as recovery
    frame = raw_frame("statistics", ns=[BASE_NS + 1_000_000_000])
    frame.loc[:, "price"] = np.int64(31_532_500_000_000 if conflicting_payload else 31_532_250_000_000)
    frame.loc[:, "quantity"] = np.int64(2**63 - 1)
    frame.loc[:, "ts_ref"] = np.uint64(2**64 - 1)
    monkeypatch.setattr(recovery.db.DBNStore, "from_file", lambda path: SimpleNamespace(
        schema="statistics", dataset="GLBX.MDP3", to_df=lambda **kwargs: iter([frame.copy()])))
    request = recovery.RecoveryRequest(request_id="temp_stats", schema="statistics", stype_in="raw_symbol",
                                      symbols=("NQZ6",), start="2026-10-06T16:27:32Z",
                                      end="2026-10-06T16:27:35Z")
    with psycopg.connect(get_dsn()) as conn:
        conn.execute("CREATE TEMP TABLE recovery_stats (ts_event timestamptz,ts_recv timestamptz,"
                     "instrument_id integer,symbol text,stat_type smallint,price float8,quantity bigint,"
                     "sequence bigint,ts_ref timestamptz,update_action smallint,stat_flags smallint,"
                     "dataset text DEFAULT 'GLBX.MDP3',UNIQUE(instrument_id,ts_event,stat_type,sequence))")
        conn.execute("INSERT INTO recovery_stats VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                     ("2026-10-06T16:27:33Z","2026-10-06T16:27:33Z",261401,"NQZ6",1,31532.25,
                      2**63-1,0,recovery.LEGACY_UNDEF_REF,0,0,"GLBX.MDP3"))
        conn.commit()

        @contextmanager
        def connection_factory():
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise

        if conflicting_payload:
            with pytest.raises(psycopg.errors.UniqueViolation):
                recovery.ingest_recovery_file(tmp_path / "fake.dbn", request,
                                               connection_factory=connection_factory,
                                               target_override="pg_temp.recovery_stats")
        else:
            receipt = recovery.ingest_recovery_file(tmp_path / "fake.dbn", request,
                                                    connection_factory=connection_factory,
                                                    target_override="pg_temp.recovery_stats")
            assert receipt.inserted == 0
            assert receipt.matched == 1
        assert conn.execute("SELECT count(*),min(price) FROM recovery_stats").fetchone() == (1,31532.25)
