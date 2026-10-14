"""Independent acceptance checks use generated archives and fake read-only DBs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.data import validate_gap_recovery as validator
from src.acquisition.gap_recovery import RecoveryRequest

pytestmark = pytest.mark.acquisition


def request(**overrides):
    values = dict(request_id="fixture", schema="bbo-1s", stype_in="raw_symbol", symbols=("NQZ6",),
                  start="2026-10-06T16:20:00Z", end="2026-10-06T17:10:00Z")
    values.update(overrides)
    return RecoveryRequest(**values)


def fake_archive(monkeypatch, tmp_path, req, *, symbols=None, timestamps=None, schema=None):
    path = tmp_path / "archive.dbn.zst"
    content = b"generated SDK fixture; this is not live market data"
    path.write_bytes(content)
    receipt = dict(path=str(path), request_sha256=req.identity, size_bytes=len(content),
                   file_sha256=hashlib.sha256(content).hexdigest())
    stamps = timestamps or [pd.Timestamp(req.start).value]
    dtype = [("ts_recv", "u8"), ("ts_event", "u8"), ("instrument_id", "u4"), ("raw_symbol", "S71")]
    array = np.zeros(len(stamps), dtype=dtype)
    array["ts_recv"] = stamps
    array["ts_event"] = np.uint64(2**64 - 1)  # BBO snapshot: map by receive date.
    array["instrument_id"] = 261401
    array["raw_symbol"] = b"NQZ6"
    metadata = SimpleNamespace(start=pd.Timestamp(req.start).value, end=pd.Timestamp(req.end).value,
                               symbols=list(req.symbols), stype_in=req.stype_in, stype_out="instrument_id",
                               limit=None, partial=[], not_found=[], mappings={
                                   "NQZ6": [{"start_date":"2026-10-06", "end_date":"2026-10-07", "symbol":"261401"}]})
    if symbols is not None:
        metadata.symbols = symbols
    store = SimpleNamespace(schema=schema or req.schema, dataset=req.dataset, metadata=metadata,
                            to_ndarray=lambda **kwargs: iter([array[:1], array[1:]]))
    monkeypatch.setattr(validator.db.DBNStore, "from_file", lambda path: store)
    return receipt, metadata


def test_full_ten_minute_metadata_interior_excludes_partial_edges() -> None:
    req = request(start="2026-10-06T16:25:06Z", end="2026-10-06T16:47:33Z")
    assert [value.isoformat() for value in validator.aligned_interior(req)] == [
        "2026-10-06T16:30:00+00:00", "2026-10-06T16:40:00+00:00"]
    assert validator.aligned_interior(request(start="2026-10-06T16:25:06Z", end="2026-10-06T16:27:33Z")) is None


def test_definition_metadata_interior_uses_complete_utc_days() -> None:
    req = request(schema="definition", start="2026-10-01T05:00:00Z", end="2026-10-06T16:50:00Z")
    assert [value.isoformat() for value in validator.aligned_interior(req)] == [
        "2026-10-02T00:00:00+00:00", "2026-10-06T00:00:00+00:00"]


def test_array_decoder_counts_repeated_rows_and_sdk_receive_date_mapping(monkeypatch, tmp_path) -> None:
    req = request()
    stamp = pd.Timestamp(req.start).value
    receipt, _ = fake_archive(monkeypatch, tmp_path, req, timestamps=[stamp, stamp])
    audit = validator.inspect_archive(req, receipt, chunk_size=1)
    assert audit["records"] == 2
    assert audit["instrument_ids"] == [261401]
    assert audit["delivered_symbols"] == ["NQZ6"]
    assert audit["hour_symbol_counts"] == [{"hour":"2026-10-06T16:00:00+00:00", "symbol":"NQZ6", "count":2}]


def test_raw_right_edge_rounding_stays_in_archive_count(monkeypatch, tmp_path) -> None:
    req = request(end="2026-10-06T17:00:00Z")
    receipt, _ = fake_archive(monkeypatch, tmp_path, req, timestamps=[pd.Timestamp(req.end).value - 1])
    audit = validator.inspect_archive(req, receipt)
    assert audit["hour_symbol_counts"][0]["hour"] == "2026-10-06T17:00:00+00:00"


def test_invalid_target_time_cannot_expand_database_scan_window(monkeypatch, tmp_path) -> None:
    req = request(schema="tbbo")
    receipt, _ = fake_archive(monkeypatch, tmp_path, req)
    with pytest.raises(ValueError, match="invalid target timestamp"):
        validator.inspect_archive(req, receipt)


@pytest.mark.parametrize("bad", ["identity", "digest", "symbols", "schema", "interval", "limited"])
def test_archive_provenance_and_header_fail_closed(monkeypatch, tmp_path, bad) -> None:
    req = request()
    receipt, metadata = fake_archive(monkeypatch, tmp_path, req, symbols=["ESZ6"] if bad == "symbols" else None,
                                     schema="tbbo" if bad == "schema" else None)
    if bad == "identity":
        receipt["request_sha256"] = "invalid"
    elif bad == "digest":
        receipt["file_sha256"] = "invalid"
    elif bad == "interval":
        metadata.end += 1
    elif bad == "limited":
        metadata.limit = 1
    with pytest.raises(ValueError):
        validator.inspect_archive(req, receipt)


def test_symbol_hour_deficit_cannot_be_hidden_by_another_symbol_extra() -> None:
    hour = "2026-10-06T16:00:00+00:00"
    result = validator.compare_counts([dict(hour=hour,symbol="NQZ6",count=5),
                                       dict(hour=hour,symbol="NQH7",count=2)],
                                      [(pd.Timestamp(hour),"NQZ6",4),(pd.Timestamp(hour),"NQH7",20)])
    assert result["passed"] is False
    assert result["deficit_groups"][0]["missing"] == 1
    assert result["retained_extra_occurrences"] == 18


def test_ingestion_receipts_record_existing_zero_insert_retry_proof(tmp_path) -> None:
    directory = tmp_path / "ingestion"
    directory.mkdir()
    first = dict(request_id="fixture",records=2,inserted=1,matched=1)
    second = dict(request_id="fixture",records=2,inserted=0,matched=2)
    (directory / "first.json").write_text(json.dumps(first))
    (directory / "second.json").write_text(json.dumps(second))
    evidence = validator.ingestion_evidence(tmp_path, [request()])
    assert evidence["all_receipts_present"]
    assert evidence["zero_insert_receipts"][0]["prior_receipts"] == 1


def test_vendor_count_is_metadata_only_and_exactly_aligned() -> None:
    client = MagicMock()
    client.metadata.get_record_count.return_value = 12
    req = request()
    audit = dict(aligned_interval=[req.start,req.end],aligned_count=12)
    result = validator.vendor_count_check(req, audit, client=client)
    assert result["status"] == "equal"
    client.metadata.get_record_count.assert_called_once_with(**req.params())
    client.timeseries.get_range.assert_not_called()


def test_database_aggregation_is_read_only_and_batches_shared_chain_window() -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [
        (datetime(2026,10,6,16,tzinfo=timezone.utc),"NQZ6",3),
        (datetime(2026,10,6,16,tzinfo=timezone.utc),"NQH7",2)]
    audits=[]
    for name, count in [("NQZ6",2),("NQH7",2)]:
        audits.append(dict(request_id=name,schema="bbo-1s",dataset="GLBX.MDP3",start="2026-10-06T16:20:00Z",
                           end="2026-10-06T17:10:00Z",records=count,delivered_symbols=[name],
                           target_min_us=1791303600000000,target_max_us=1791303600000000,
                           hour_symbol_counts=[dict(hour="2026-10-06T16:00:00+00:00",symbol=name,count=count)]))
    result=validator.validate_database(audits, connection_factory=lambda: conn)
    assert result["passed"]
    statements = [call.args[0].as_string() if hasattr(call.args[0],"as_string") else call.args[0]
                  for call in conn.execute.call_args_list]
    assert statements[0] == "SET TRANSACTION READ ONLY"
    queries=[statement for statement in statements if statement.startswith("SELECT")]
    assert len(queries) == 1
    assert "symbol=ANY(%s)" in queries[0]
    assert "GROUP BY 1,2" in queries[0]
    assert all("INSERT" not in statement and "COPY" not in statement for statement in statements)


def test_empty_sparse_market_delivery_is_explicitly_accepted_without_scan() -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    result = validator.validate_database([dict(request_id="sparse",schema="statistics",records=0)],
                                          connection_factory=lambda: conn)
    assert result["passed"]
    assert result["requests"]["sparse"]["status"] == "empty_delivered_archive"
    assert all(not str(call.args[0]).startswith("SELECT") for call in conn.execute.call_args_list)
