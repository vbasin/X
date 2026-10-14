"""Recovery acquisition tests use fake endpoints and isolated temporary files."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.acquisition.gap_recovery import (
    AvailabilityError, CostGateError, RecoveryBatchError, RecoveryRequest,
    check_disk_budget, download_request, download_requests, file_sha256,
    find_saved_receipt, load_requests,
)

pytestmark = pytest.mark.acquisition


def request(**changes):
    values = dict(request_id="repair", schema="bbo-1s", stype_in="raw_symbol",
                  start="2026-10-06T16:25:06Z", end="2026-10-06T16:27:33Z",
                  symbols=("Q1BV6 P25000", "Q1BV6 C25000"))
    values.update(changes)
    return RecoveryRequest(**values)


class Client:
    def __init__(self, cost="0", *, ranges=None, fail=False):
        self.events = []
        self.fail = fail
        self.metadata = SimpleNamespace(
            get_dataset_range=Mock(side_effect=self.available),
            get_cost=Mock(side_effect=lambda **params: self.quote(cost, params)),
        )
        self.timeseries = SimpleNamespace(get_range=Mock(side_effect=self.download))
        self.ranges = ranges if ranges is not None else {"schema": {"bbo-1s": {
            "start": "2026-10-02T00:00:00Z", "end": "2026-10-06T17:00:00Z"}}}

    def available(self, **params):
        self.events.append("availability")
        return self.ranges

    def quote(self, cost, params):
        self.events.append("quote")
        return cost

    def download(self, **params):
        self.events.append("download")
        with Path(params["path"]).open("xb") as stream:
            stream.write(b"fake DBN bytes")
        if self.fail:
            raise TimeoutError("fake partial download")


def acquire(req, tmp_path, client, **kwargs):
    return download_request(req, job_dir=tmp_path / "job", output_dir=tmp_path / "raw",
                            client_factory=lambda: client, reserve_bytes=0, projected_bytes=0,
                            **kwargs)


def test_identity_exact_scope_and_nanos():
    original = request()
    assert original.symbols == ("Q1BV6 C25000", "Q1BV6 P25000")
    assert original.roots == ("Q1BV6",)
    assert original.identity == request(symbols=tuple(reversed(original.symbols)), request_id="alias").identity
    assert original.identity != request(symbols=("Q1BV6 C25000",)).identity
    assert original.identity != request(stype_in="parent").identity
    assert original.identity != request(end="2026-10-06T16:27:33.000000001Z").identity
    assert request(start="2026-10-06T11:25:06-05:00").identity == original.identity


@pytest.mark.parametrize("changes", [
    {"symbols": ()}, {"symbols": ("ALL_SYMBOLS",)}, {"symbols": tuple(f"s{i}" for i in range(2001))},
    {"start": "2026-10-06T16:25:06"}, {"end": "2026-10-06T16:25:06Z"},
    {"stype_in": "instrument_id"},
])
def test_unbounded_or_invalid_request_rejected(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_manifest_resolves_symbol_set_and_ignores_old_quote(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "interval_semantics": "half-open [start,end)",
        "symbol_sets": {"scope": ["Q1BV6 P25000", "Q1BV6 C25000"]}, "requests": [{
            **request().params(), "id": "repair", "symbols": None, "symbol_set": "scope",
            "quoted_cost_usd": 999}]}))
    assert load_requests(plan) == [request()]


@pytest.mark.parametrize("cost", ["0.01", "-0.01", "NaN", "Infinity", "-Infinity", None, "bad"])
def test_every_nonzero_or_nonfinite_fresh_price_prevents_download(tmp_path, cost):
    client = Client(cost)
    with pytest.raises(CostGateError):
        acquire(request(), tmp_path, client)
    client.timeseries.get_range.assert_not_called()
    assert not list(tmp_path.rglob("*.dbn.zst"))


@pytest.mark.parametrize("cost", [0, 0.0, "0.000", "-0"])
def test_zero_gate_exact_params_and_timeout(tmp_path, cost):
    req = request()
    client = Client(cost)
    receipt = acquire(req, tmp_path, client)
    assert client.events == ["availability", "quote", "download"]
    client.metadata.get_cost.assert_called_once_with(**req.params())
    params = client.timeseries.get_range.call_args.kwargs.copy()
    assert params.pop("path") == receipt.path
    assert params == req.params()
    assert client.metadata.TIMEOUT == 20
    assert client.timeseries.TIMEOUT == 60
    assert receipt.request_sha256 == req.identity
    assert receipt.file_sha256 == file_sha256(receipt.path)


@pytest.mark.parametrize("ranges", [
    {"start": "2020-01-01T00:00:00Z", "end": "2030-01-01T00:00:00Z", "schema": {}},
    {"schema": {"bbo-1s": {"start": "2026-10-02T00:00:00Z", "end": "2026-10-06T16:27:32.999999999Z"}}},
    {"schema": {"bbo-1s": {"start": "2026-10-06T16:25:06.000000001Z", "end": "2026-10-06T17:00:00Z"}}},
])
def test_schema_specific_availability_blocks_even_if_dataset_is_newer(tmp_path, ranges):
    client = Client(ranges=ranges)
    with pytest.raises(AvailabilityError):
        acquire(request(), tmp_path, client)
    client.metadata.get_cost.assert_not_called()
    client.timeseries.get_range.assert_not_called()


def test_schema_ranges_fallback_and_exact_endpoint(tmp_path):
    req = request()
    ranges = {"schema_ranges": {"bbo-1s": {"start": req.start, "end": req.end}}}
    assert acquire(req, tmp_path, Client(ranges=ranges)).size_bytes > 0


def test_saved_receipt_reuse_avoids_network_and_checks_hash(tmp_path):
    req = request()
    receipt = acquire(req, tmp_path, Client())
    factory = Mock(side_effect=AssertionError("reuse must not contact vendor"))
    assert download_request(req, job_dir=tmp_path / "job", output_dir=tmp_path / "raw",
                            client_factory=factory) == receipt
    assert find_saved_receipt(req, tmp_path / "job", tmp_path / "raw") == receipt
    alias = request(request_id="new_label")
    assert acquire(alias, tmp_path, Client()).request_id == "new_label"
    # Mutate only this test's disposable fixture to demonstrate corrupt bytes.
    Path(receipt.path).write_bytes(b"wrong DBN bytes")
    with pytest.raises(ValueError, match="integrity"):
        find_saved_receipt(req, tmp_path / "job", tmp_path / "raw")


def test_failed_partial_is_preserved_and_retry_uses_new_attempt(tmp_path):
    first = Client(fail=True)
    with pytest.raises(TimeoutError):
        acquire(request(), tmp_path, first)
    partial = Path(first.timeseries.get_range.call_args.kwargs["path"])
    assert partial.read_bytes() == b"fake DBN bytes"
    receipt = acquire(request(), tmp_path, Client())
    assert Path(receipt.path) != partial
    assert partial.read_bytes() == b"fake DBN bytes"
    assert len(list((tmp_path / "job" / "acquisition").glob("*.json"))) == 1


def test_disk_budget_blocks_before_any_vendor_call(tmp_path, monkeypatch):
    monkeypatch.setattr("src.acquisition.gap_recovery.shutil.disk_usage", lambda _: SimpleNamespace(free=25))
    check_disk_budget(tmp_path, reserve_bytes=20, projected_bytes=5)
    factory = Mock()
    with pytest.raises(OSError):
        download_request(request(), job_dir=tmp_path / "job", output_dir=tmp_path / "raw",
                         client_factory=factory, reserve_bytes=20, projected_bytes=6)
    factory.assert_not_called()


def test_parallel_host_budget_and_ordered_results(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("src.acquisition.gap_recovery.get_vm_config", lambda: SimpleNamespace(
        workers_for=lambda **kw: calls.append(kw) or 2))
    barrier = threading.Barrier(2)
    class ConcurrentClient(Client):
        def download(self, **params):
            barrier.wait(timeout=5)
            return super().download(**params)
    items = [request(request_id=str(i), symbols=(f"Q1BV6 C{25000+i}",)) for i in range(2)]
    receipts = download_requests(items, job_dir=tmp_path / "job", output_dir=tmp_path / "raw",
                                client_factory=ConcurrentClient, reserve_bytes=0, projected_bytes=0)
    assert [r.request_id for r in receipts] == ["0", "1"]
    assert calls == [{"mem_per_worker_gb": 0.5}]
    assert receipts[0].path != receipts[1].path


def test_batch_reports_completed_receipts_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.acquisition.gap_recovery.get_vm_config", lambda: SimpleNamespace(workers_for=lambda **kw: 1))
    clients = iter([Client(), Client("0.01")])
    items = [request(request_id="good"), request(request_id="bad", symbols=("Q1BV6 C26000",))]
    with pytest.raises(RecoveryBatchError) as caught:
        download_requests(items, job_dir=tmp_path / "job", output_dir=tmp_path / "raw",
                          client_factory=lambda: next(clients), reserve_bytes=0, projected_bytes=0)
    assert [r.request_id for r in caught.value.receipts] == ["good"]
    assert isinstance(caught.value.errors["bad"], CostGateError)


def test_complete_batch_resume_requires_no_new_disk_projection_or_vendor(tmp_path, monkeypatch):
    req = request()
    receipt = acquire(req, tmp_path, Client())
    monkeypatch.setattr("src.acquisition.gap_recovery.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    factory = Mock(side_effect=AssertionError("no network on completed resume"))
    progress = Mock()
    assert download_requests([req], job_dir=tmp_path / "job", output_dir=tmp_path / "raw",
                             client_factory=factory, progress=progress) == [receipt]
    progress.assert_called_once_with(1, 1, req, receipt)
    factory.assert_not_called()
