"""Vendor definitions are normal, concurrent acquisition with strict cost gating.

Every API and DB connection is fake: this module cannot fetch vendor data or
read/write the live definitions table.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from databento_dbn import UNDEF_PRICE

from src.acquisition.definition_acquisition import DefinitionAcquirer, VendorDefinition

pytestmark = pytest.mark.acquisition

NOW = datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc)
EXPIRATION = datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc)
SYMBOL = "Q1CV6 P25000"


def _row(symbol: str = SYMBOL, *, source: str = "databento", instrument_id: int = 42648901):
    return (
        datetime(2026, 10, 4, 13, 43, 29, tzinfo=timezone.utc),
        instrument_id, symbol, "P", 25000.0, EXPIRATION,
        "NQZ6", "XCME", "USD", None, 20.0, source,
    )


def _record(symbol: str = SYMBOL, **overrides):
    values = dict(
        ts_event=int(_row()[0].timestamp() * 1e9),
        ts_recv=int(datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp() * 1e9),
        instrument_id=42648901, raw_symbol=symbol, instrument_class="P",
        strike_price=25_000_000_000_000, expiration=int(EXPIRATION.timestamp() * 1e9),
        underlying="NQZ6", exchange="XCME", currency="USD",
        min_price_increment=UNDEF_PRICE, unit_of_measure_qty=20_000_000_000,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Pool:
    def __init__(self, rows=()):
        self.rows = {row[2]: row for row in rows}
        self.active_connections = 0
        self.inserts = []
        self.queries = []
        self.commits = 0

    @contextmanager
    def connection(self):
        self.active_connections += 1
        try:
            yield self
        finally:
            self.active_connections -= 1

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            symbols = set(params[0])
            return SimpleNamespace(fetchall=lambda: [
                r for s, r in self.rows.items() if s in symbols and r[-1] == params[1]
            ])
        if sql.lstrip().upper().startswith("INSERT"):
            self._insert(params)
            return SimpleNamespace(rowcount=len(params) // 12)
        raise AssertionError(f"Unexpected SQL: {sql}")

    @contextmanager
    def cursor(self):
        yield self

    def executemany(self, sql, rows):
        self.queries.append((sql, rows))
        for row in rows:
            self._insert(row)

    def _insert(self, params):
        self.inserts.append(tuple(params))
        for offset in range(0, len(params), 12):
            row = tuple(params[offset:offset + 12])
            old = self.rows.get(row[2])
            if old is None or old[-1] == "synthetic":
                self.rows[row[2]] = row

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _setup(rows=(), records=None, *, cost=0.0, **kwargs):
    pool = _Pool(rows)
    client = MagicMock()
    client.metadata.get_dataset_range.return_value = {
        "start": "2010-06-06T00:00:00Z", "end": "2026-10-06T15:45:00Z",
        "schema_ranges": {"definition": {
            "start": "2010-06-06T00:00:00Z", "end": "2026-10-06T15:30:00Z",
        }},
    }
    client.metadata.get_cost.return_value = cost
    vendor_records = [_record()] if records is None else records
    client.timeseries.get_range.return_value = vendor_records
    acquirer = DefinitionAcquirer(
        pool, client_factory=lambda: client, workers=1, clock=lambda: NOW, **kwargs,
    )
    return acquirer, pool, client


def test_vendor_definition_normalizes_core_terms_without_inventing_tick() -> None:
    definition = VendorDefinition.from_record(_record())
    row = definition.row()
    assert row[1:5] == (42648901, SYMBOL, "P", 25000.0)
    assert row[5] == EXPIRATION
    assert row[6:9] == ("NQZ6", "XCME", "USD")
    assert row[9] is None
    assert row[10] == 20.0
    assert row[11] == "databento"


def test_vendor_definition_from_db_row_preserves_symbol_identity() -> None:
    definition = VendorDefinition.from_row(_row())
    assert definition.row() == _row()


def test_existing_vendor_row_requires_no_vendor_request() -> None:
    acquirer, pool, client = _setup([_row()])
    result = acquirer.acquire([SYMBOL])
    assert result[SYMBOL].row() == _row()
    client.metadata.get_cost.assert_not_called()
    client.timeseries.get_range.assert_not_called()
    assert pool.inserts == []


def test_exact_missing_symbol_is_cost_checked_then_written() -> None:
    acquirer, pool, client = _setup()
    result = acquirer.acquire([SYMBOL, SYMBOL])
    assert result[SYMBOL].row()[2] == SYMBOL
    assert pool.rows[SYMBOL][-1] == "databento"
    cost_kwargs = client.metadata.get_cost.call_args.kwargs
    download_kwargs = client.timeseries.get_range.call_args.kwargs
    for kwargs in (cost_kwargs, download_kwargs):
        assert kwargs["symbols"] == [SYMBOL]
        assert kwargs["stype_in"] == "raw_symbol"
        assert kwargs["schema"] == "definition"
        assert kwargs["dataset"] == "GLBX.MDP3"
        assert pd.Timestamp(kwargs["end"]) <= pd.Timestamp("2026-10-06T15:30:00Z")
    for key in ("start", "end", "symbols", "schema", "dataset", "stype_in"):
        assert cost_kwargs[key] == download_kwargs[key]


def test_legacy_synthetic_row_is_upgraded_by_standard_acquisition() -> None:
    acquirer, pool, client = _setup([_row(source="synthetic")])
    result = acquirer.acquire([SYMBOL])
    client.timeseries.get_range.assert_called_once()
    assert pool.rows[SYMBOL][-1] == "databento"
    assert result[SYMBOL].row()[-1] == "databento"


@pytest.mark.parametrize("cost", [0.01, -0.01, float("nan"), float("inf")])
def test_only_exact_zero_cost_authorizes_download(cost) -> None:
    acquirer, pool, client = _setup(cost=cost)
    assert acquirer.acquire([SYMBOL]) == {}
    client.timeseries.get_range.assert_not_called()
    assert pool.inserts == []


def test_vendor_call_holds_no_database_connection() -> None:
    acquirer, pool, client = _setup()

    def fetch(**kwargs):
        assert pool.active_connections == 0
        return [_record()]

    def quote(**kwargs):
        assert pool.active_connections == 0
        return 0.0

    client.metadata.get_cost.side_effect = quote
    client.timeseries.get_range.side_effect = fetch
    assert SYMBOL in acquirer.acquire([SYMBOL])


def test_live_definition_race_returns_committed_authoritative_row() -> None:
    acquirer, pool, client = _setup()
    live_row = _row(instrument_id=999)

    def fetch(**kwargs):
        # The normal live callback writes authoritative metadata while the
        # historical request is in flight. Its row must win over older data.
        pool.rows[SYMBOL] = live_row
        return [_record()]

    client.timeseries.get_range.side_effect = fetch
    result = acquirer.acquire([SYMBOL])
    assert pool.rows[SYMBOL] == live_row
    assert result[SYMBOL].row() == live_row


def test_unrequested_and_future_definitions_are_not_written() -> None:
    acquirer, pool, client = _setup(records=[
        _record("NQZ6 C30000"),
        _record(ts_recv=int(datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp() * 1e9)),
    ])
    assert acquirer.acquire([SYMBOL]) == {}
    assert pool.inserts == []


def test_empty_response_can_be_retried_without_new_subscription() -> None:
    acquirer, pool, client = _setup(records=[])
    assert acquirer.acquire([SYMBOL]) == {}
    client.timeseries.get_range.return_value = [_record()]
    assert SYMBOL in acquirer.acquire([SYMBOL])
    assert client.timeseries.get_range.call_count == 2


def test_submit_is_nonblocking_and_never_calls_vendor_inline() -> None:
    acquirer, pool, client = _setup()
    acquirer.submit([SYMBOL, SYMBOL])
    client.metadata.get_dataset_range.assert_not_called()
    client.metadata.get_cost.assert_not_called()
    client.timeseries.get_range.assert_not_called()
    assert pool.queries == []


def test_background_task_retries_pending_symbols_without_resubscription() -> None:
    import threading

    acquirer, pool, client = _setup(retry_s=0.01)
    client.timeseries.get_range.side_effect = [[], [_record()]]
    finished = threading.Event()
    acquire = acquirer.acquire

    def observe(symbols):
        result = acquire(symbols)
        if SYMBOL in result:
            finished.set()
        return result

    acquirer.acquire = observe
    acquirer.submit([SYMBOL])
    acquirer.start()
    try:
        assert finished.wait(2), "pending symbol should be retried by the normal task"
    finally:
        acquirer.stop(timeout=2)
    assert client.timeseries.get_range.call_count == 2
    assert acquirer.pending == frozenset()
    assert acquirer.get(SYMBOL).source == "databento"


def test_definition_snapshot_cannot_mutate_acquisition_cache() -> None:
    acquirer, pool, client = _setup([_row()])
    acquirer.acquire([SYMBOL])
    snapshot = acquirer.definitions
    snapshot.clear()
    assert acquirer.get(SYMBOL).symbol == SYMBOL


def test_cost_quote_failure_does_not_authorize_download() -> None:
    acquirer, pool, client = _setup()
    client.metadata.get_cost.side_effect = RuntimeError("vendor unavailable")
    assert acquirer.acquire([SYMBOL]) == {}
    client.timeseries.get_range.assert_not_called()
    assert pool.inserts == []


def test_incomplete_vendor_response_stays_pending_without_synthetic_fallback() -> None:
    acquirer, pool, client = _setup(records=[_record(expiration=0)])
    acquirer.submit([SYMBOL])
    assert acquirer.acquire([SYMBOL]) == {}
    assert SYMBOL in acquirer.pending
    assert pool.inserts == []
    assert acquirer.get(SYMBOL) is None


def test_batched_requests_include_only_missing_raw_symbols() -> None:
    symbols = [SYMBOL, "NQZ6 C30000", "QNEV6 P24000"]
    acquirer, pool, client = _setup([_row()], records=[], batch_size=1)
    result = acquirer.acquire(symbols)
    assert set(result) == {SYMBOL}
    requested = [call.kwargs["symbols"] for call in client.timeseries.get_range.call_args_list]
    assert sorted(requested) == [["NQZ6 C30000"], ["QNEV6 P24000"]]


def test_latest_publication_is_selected_regardless_of_response_order() -> None:
    newer_event = datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)
    acquirer, pool, client = _setup(records=[
        _record(ts_event=int(newer_event.timestamp() * 1e9), underlying="NQZ6"),
        _record(underlying="NQU6"),
    ])
    result = acquirer.acquire([SYMBOL])
    assert result[SYMBOL].ts_event == newer_event
    assert result[SYMBOL].underlying == "NQZ6"
    assert len(pool.inserts) == 1


def test_available_future_endpoint_is_clamped_to_current_time() -> None:
    acquirer, pool, client = _setup()
    client.metadata.get_dataset_range.return_value["schema_ranges"]["definition"]["end"] = (
        "2026-10-07T00:00:00Z"
    )
    assert SYMBOL in acquirer.acquire([SYMBOL])
    end = pd.Timestamp(client.timeseries.get_range.call_args.kwargs["end"])
    assert end == pd.Timestamp(NOW)


def test_incomplete_existing_vendor_row_remains_pending_for_acquisition() -> None:
    incomplete = list(_row())
    incomplete[5] = None
    acquirer, pool, client = _setup([tuple(incomplete)], records=[])
    acquirer.submit([SYMBOL])
    assert acquirer.acquire([SYMBOL]) == {}
    client.timeseries.get_range.assert_called_once()
    assert SYMBOL in acquirer.pending
    assert acquirer.get(SYMBOL) is None


def test_stop_during_fetch_prevents_later_definition_write() -> None:
    acquirer, pool, client = _setup()

    def fetch(**kwargs):
        acquirer.stop(timeout=0)
        return [_record()]

    client.timeseries.get_range.side_effect = fetch
    assert acquirer.acquire([SYMBOL]) == {}
    assert pool.inserts == []
    assert acquirer.get(SYMBOL) is None


def test_quarterly_chain_uses_vendor_contract_terms_without_symbol_parser() -> None:
    symbol = "NQZ6 C31450"
    expiration = datetime(2026, 12, 18, 14, 30, tzinfo=timezone.utc)
    acquirer, pool, client = _setup(records=[_record(
        symbol, instrument_class="C", strike_price=31_450_000_000_000,
        expiration=int(expiration.timestamp() * 1e9),
    )])
    result = acquirer.acquire([symbol])
    assert result[symbol].instrument_class == "C"
    assert result[symbol].strike_price == 31450.0
    assert result[symbol].expiration == expiration
    assert result[symbol].underlying == "NQZ6"
    assert pool.rows[symbol][-1] == "databento"
