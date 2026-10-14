"""Cost-gated, immutable historical acquisitions for the October capture gaps.

This module downloads raw files only. Ingestion validates their DBN contents
separately; a receipt hash establishes identity and transport integrity, not
market-data completeness. No database or credentials are needed by the tests.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

import databento as db
import pandas as pd

from src.acquisition.schemas import DATASET
from src.common.data_paths import raw_dir
from src.common.vm_config import get_vm_config

GIB = 1024**3
DEFAULT_RESERVE_BYTES = 20 * GIB
DEFAULT_PROJECTED_BYTES = 6 * GIB


class CostGateError(RuntimeError):
    """The exact request was not freshly quoted at finite, exact zero."""


class AvailabilityError(RuntimeError):
    """The requested schema interval is not completely available."""


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("recovery timestamps must be explicit, timezone-aware values")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class RecoveryRequest:
    request_id: str
    schema: str
    stype_in: str
    start: str
    end: str
    symbols: tuple[str, ...]
    dataset: str = DATASET

    def __post_init__(self) -> None:
        names = tuple(sorted(set(self.symbols)))
        if (not self.request_id or not self.schema or not self.dataset
                or not names or len(names) > 2000
                or any(not isinstance(name, str) or not name.strip() or name == "ALL_SYMBOLS" for name in names)
                or self.stype_in not in ("parent", "raw_symbol")):
            raise ValueError("recovery requires an explicit bounded symbol request")
        start, end = _timestamp(self.start), _timestamp(self.end)
        if start >= end:
            raise ValueError("recovery interval must have start < end")
        object.__setattr__(self, "symbols", names)
        object.__setattr__(self, "start", start.isoformat())
        object.__setattr__(self, "end", end.isoformat())

    def params(self) -> dict[str, Any]:
        """Exact parameters shared by the price quote and data request."""
        return {"dataset": self.dataset, "schema": self.schema,
                "stype_in": self.stype_in, "start": self.start,
                "end": self.end, "symbols": list(self.symbols)}

    @property
    def identity(self) -> str:
        """Content identity excludes the human-readable request label."""
        canonical = json.dumps(self.params(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(sorted({symbol.split()[0] for symbol in self.symbols}))


def load_requests(plan_path: Path | str) -> list[RecoveryRequest]:
    """Resolve exact symbol sets from the reviewed request manifest."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("version") != 1 or plan.get("interval_semantics") != "half-open [start,end)":
        raise ValueError("unsupported recovery plan or interval semantics")
    requests = []
    seen = set()
    for row in plan["requests"]:
        if row["id"] in seen:
            raise ValueError("duplicate recovery request id")
        seen.add(row["id"])
        names = row.get("symbols")
        if row.get("symbol_set"):
            if names is not None:
                raise ValueError("request must use symbols or a symbol_set, not both")
            names = plan["symbol_sets"][row["symbol_set"]]
        requests.append(RecoveryRequest(
            row["id"], row["schema"], row["stype_in"], row["start"], row["end"],
            tuple(names or ()), row.get("dataset", DATASET),
        ))
    return requests


@dataclass(frozen=True)
class DownloadReceipt:
    request_id: str
    request_sha256: str
    path: str
    file_sha256: str
    size_bytes: int
    cost_usd: str
    quoted_at: str
    downloaded_at: str


def file_sha256(path: Path | str) -> str:
    """Hash a raw archive without decoding or loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_raw_dir() -> Path:
    return raw_dir() / "backfill" / "gaps_20261002_20261006"


def find_saved_receipt(
    request: RecoveryRequest, job_dir: Path | str, output_dir: Path | str | None = None,
) -> DownloadReceipt | None:
    """Reuse only an exact receipt whose immutable archive still hashes correctly."""
    root = Path(output_dir) if output_dir is not None else recovery_raw_dir()
    for path in sorted((Path(job_dir) / "acquisition").glob(f"{request.identity}_*.json")):
        saved = json.loads(path.read_text())
        if saved.get("params") != request.params():
            raise ValueError(f"receipt request mismatch: {path}")
        receipt = DownloadReceipt(**saved["receipt"])
        archive = Path(receipt.path)
        if (receipt.request_sha256 != request.identity or not archive.resolve().is_relative_to(root.resolve())
                or not archive.is_file() or archive.stat().st_size != receipt.size_bytes
                or receipt.size_bytes <= 0 or file_sha256(archive) != receipt.file_sha256):
            raise ValueError(f"receipt archive integrity failure: {path}")
        try:
            cost = Decimal(receipt.cost_usd)
        except InvalidOperation as exc:
            raise ValueError("receipt has an invalid cost") from exc
        if not cost.is_finite() or cost != 0:
            raise ValueError("receipt was not acquired at exact zero cost")
        return replace(receipt, request_id=request.request_id)
    return None


def check_disk_budget(
    output_dir: Path | str, *, reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    projected_bytes: int = DEFAULT_PROJECTED_BYTES,
) -> None:
    """Require free space for both the acquisition projection and retained reserve."""
    if reserve_bytes < 0 or projected_bytes < 0:
        raise ValueError("disk budgets cannot be negative")
    existing = Path(output_dir)
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < reserve_bytes + projected_bytes:
        raise OSError(f"insufficient recovery space: {free} free; need {reserve_bytes + projected_bytes}")


def _client(factory: Callable[[], Any] | None) -> Any:
    client = (factory or db.Historical)()
    # The SDK reads TIMEOUT per HTTP endpoint for connect/read inactivity.
    # This does not limit the total duration of a healthy large download.
    client.metadata.TIMEOUT = 20
    client.timeseries.TIMEOUT = 60
    return client


def _check_available(client: Any, request: RecoveryRequest) -> None:
    metadata = client.metadata.get_dataset_range(dataset=request.dataset)
    available = (metadata.get("schema") or {}).get(request.schema)
    if not available:
        available = (metadata.get("schema_ranges") or {}).get(request.schema)
    if not available or not available.get("start") or not available.get("end"):
        raise AvailabilityError(f"no availability range for schema {request.schema}")
    if (_timestamp(request.start) < _timestamp(available["start"])
            or _timestamp(request.end) > _timestamp(available["end"])):
        raise AvailabilityError(f"request {request.request_id} exceeds schema availability {available}")


def download_request(
    request: RecoveryRequest, *, job_dir: Path | str,
    client_factory: Callable[[], Any] | None = None, output_dir: Path | str | None = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES, projected_bytes: int = DEFAULT_PROJECTED_BYTES,
) -> DownloadReceipt:
    """Freshly quote then stream one exact request into a new immutable attempt.

    Failed partial files are retained without receipts; the next attempt gets a
    different directory. Successfully receipted files are hash-checked on reuse.
    """
    root = Path(output_dir) if output_dir is not None else recovery_raw_dir()
    previous = find_saved_receipt(request, job_dir, root)
    if previous is not None:
        return previous
    check_disk_budget(root, reserve_bytes=reserve_bytes, projected_bytes=projected_bytes)
    client = _client(client_factory)
    _check_available(client, request)
    attempt = uuid4().hex
    directory = root / request.identity / attempt
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "records.dbn.zst"
    state_dir = Path(job_dir) / "acquisition"
    state_dir.mkdir(parents=True, exist_ok=True)
    params = request.params()
    try:
        cost = Decimal(str(client.metadata.get_cost(**params)))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CostGateError(f"invalid fresh quote for {request.request_id}") from exc
    if not cost.is_finite() or cost != 0:
        raise CostGateError(f"request {request.request_id} requires approval: quoted USD {cost}")
    quoted_at = datetime.now(timezone.utc).isoformat()
    client.timeseries.get_range(**params, path=str(archive))
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError(f"download did not produce a nonempty archive: {archive}")
    receipt = DownloadReceipt(
        request.request_id, request.identity, str(archive.resolve()), file_sha256(archive),
        archive.stat().st_size, str(cost), quoted_at, datetime.now(timezone.utc).isoformat(),
    )
    with (state_dir / f"{request.identity}_{attempt}.json").open("x") as stream:
        json.dump({"params": params, "receipt": asdict(receipt)}, stream, indent=2)
    return receipt


class RecoveryBatchError(RuntimeError):
    """Completed acquisitions remain available when another request fails."""

    def __init__(self, receipts: list[DownloadReceipt], errors: dict[str, Exception]) -> None:
        self.receipts = receipts
        self.errors = errors
        super().__init__(f"recovery acquisition failures: {', '.join(errors)}")


def download_requests(
    requests: Iterable[RecoveryRequest], *, job_dir: Path | str,
    client_factory: Callable[[], Any] | None = None, output_dir: Path | str | None = None,
    workers: int | None = None, reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    projected_bytes: int = DEFAULT_PROJECTED_BYTES,
    progress: Callable[[int, int, RecoveryRequest, DownloadReceipt | Exception], None] | None = None,
) -> list[DownloadReceipt]:
    """Download independent requests using host-sized threads and a total disk budget."""
    items = list(requests)
    if not items:
        return []
    if len({item.request_id for item in items}) != len(items):
        raise ValueError("duplicate recovery request ids")
    root = Path(output_dir) if output_dir is not None else recovery_raw_dir()
    completed: dict[str, DownloadReceipt] = {}
    pending = []
    for item in items:
        previous = find_saved_receipt(item, job_dir, root)
        if previous is None:
            pending.append(item)
        else:
            completed[item.request_id] = previous
            if progress is not None:
                progress(len(completed), len(items), item, previous)
    if not pending:
        return [completed[item.request_id] for item in items]
    check_disk_budget(root, reserve_bytes=reserve_bytes, projected_bytes=projected_bytes)
    host_workers = get_vm_config().workers_for(mem_per_worker_gb=0.5)
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")
    count = min(len(pending), host_workers, workers if workers is not None else host_workers)
    errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = {executor.submit(
            download_request, item, job_dir=job_dir, client_factory=client_factory,
            output_dir=root, reserve_bytes=reserve_bytes, projected_bytes=0,
        ): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                completed[item.request_id] = future.result()
            except Exception as exc:
                errors[item.request_id] = exc
            if progress is not None:
                progress(len(completed) + len(errors), len(items), item,
                         errors.get(item.request_id) or completed[item.request_id])
    receipts = [completed[item.request_id] for item in items if item.request_id in completed]
    if errors:
        raise RecoveryBatchError(receipts, errors)
    return receipts
