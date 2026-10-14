#!/usr/bin/env python3
"""Read-only acceptance audit of immutable recovery DBN files and bounded DB scopes.

This checks delivered record counts and hour/symbol coverage, not full event
identity. The append receipts and ingestion multiset checks supply that evidence.
No download, COPY, update, compression, service change, or replay is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import databento as db
from databento.common.symbology import InstrumentMap
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.acquisition.gap_recovery import RecoveryRequest, load_requests
from src.common.vm_config import get_vm_config
from src.ingestion.gap_recovery import SPECS, _live_timestamps, enable_strict_dbn_validation
from src.ingestion.loader import get_dsn

COUNT_RULE_URL = "https://databento.com/docs/api-reference-historical/metadata/metadata-get-record-count"
SOURCE_GAP = ["2026-10-05T22:50:00Z", "2026-10-06T00:00:00Z"]
HOUR_US = 3_600_000_000
DAY_NS = 86_400_000_000_000


def aligned_interior(request: RecoveryRequest) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Metadata counts are precise only for full 10-minute (definition: day) bins."""
    frequency = "1D" if request.schema == "definition" else "10min"
    start, end = pd.Timestamp(request.start).ceil(frequency), pd.Timestamp(request.end).floor(frequency)
    return (start, end) if start < end else None


def load_download_receipts(job_dir: Path) -> dict[str, dict]:
    receipts = {}
    for path in sorted(job_dir.glob("downloads_*.json"), key=lambda item: item.stat().st_mtime_ns):
        receipts.update(json.loads(path.read_text()))
    return receipts


def ingestion_evidence(job_dir: Path, requests: list[RecoveryRequest]) -> dict:
    histories: dict[str, list[dict]] = defaultdict(list)
    paths = list((job_dir / "ingestion").glob("*.json")) + list(job_dir.glob("*_early_ingestion.json"))
    for path in sorted(paths, key=lambda item: (item.stat().st_mtime_ns, item.name)):
        row = json.loads(path.read_text())
        if "request_id" in row and "records" in row:
            histories[row["request_id"]].append({**row, "receipt_path": str(path)})
    missing = [request.request_id for request in requests if request.request_id not in histories]
    retries = []
    for request_id, rows in histories.items():
        for position, row in enumerate(rows):
            if row.get("records", 0) > 0 and row.get("inserted") == 0 and row.get("matched") == row.get("records"):
                retries.append({"request_id": request_id, "records": row["records"],
                                "receipt_path": row["receipt_path"], "prior_receipts": position,
                                "prior_matching_receipts": sum(prior["records"] == row["records"] for prior in rows[:position])})
    return {"all_receipts_present": not missing, "missing": missing,
            "latest": {key: rows[-1] for key, rows in histories.items()},
            "zero_insert_receipts": retries,
            "limit": "Zero-insert receipts prove their bounded retries only; no broad replay was performed."}


def _iso_ns(value: int | None) -> str | None:
    return pd.Timestamp(value, unit="ns", tz="UTC").isoformat() if value is not None else None


def inspect_archive(request: RecoveryRequest, receipt: dict, *, chunk_size: int = 200_000) -> dict:
    """Decode exact fixed-point arrays and resolve each historical ID/date via SDK."""
    path = Path(receipt["path"])
    if receipt["request_sha256"] != request.identity:
        raise ValueError("download receipt request identity differs")
    if path.stat().st_size != receipt["size_bytes"]:
        raise ValueError("archive size differs from immutable download receipt")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != receipt["file_sha256"]:
        raise ValueError("archive digest differs from immutable download receipt")
    store = db.DBNStore.from_file(path)
    metadata = store.metadata
    if str(store.schema) != request.schema or str(store.dataset) != request.dataset:
        raise ValueError("archive schema/dataset mismatch")
    if int(metadata.start) != pd.Timestamp(request.start).value or int(metadata.end) != pd.Timestamp(request.end).value:
        raise ValueError("archive header interval mismatch")
    if set(metadata.symbols) != set(request.symbols) or str(metadata.stype_in) != request.stype_in:
        raise ValueError("archive header subscription mismatch")
    if metadata.limit is not None:
        raise ValueError("limited archive cannot establish requested coverage")
    mapping = InstrumentMap()
    mapping.insert_metadata(metadata)
    cache: dict[tuple[int, int], str] = {}
    bins: Counter = Counter()
    ids, names = set(), set()
    records = aligned_records = 0
    receive_min = receive_max = target_min_us = target_max_us = None
    aligned = aligned_interior(request)
    raw_start, raw_end = pd.Timestamp(request.start).value, pd.Timestamp(request.end).value
    for array in store.to_ndarray(schema=request.schema, count=chunk_size):
        if not len(array):
            continue
        receive = array["ts_recv"]
        if request.schema != "definition" and ((receive < raw_start) | (receive >= raw_end)).any():
            raise ValueError("record outside exact requested receive interval")
        records += len(array)
        lo, hi = int(receive.min()), int(receive.max())
        receive_min = lo if receive_min is None else min(receive_min, lo)
        receive_max = hi if receive_max is None else max(receive_max, hi)
        if aligned:
            aligned_records += int(((receive >= aligned[0].value) & (receive < aligned[1].value)).sum())
        ids.update(map(int, np.unique(array["instrument_id"])))
        if request.schema == "definition":
            chunk_names = {bytes(value).rstrip(b"\0").decode() for value in np.unique(array["raw_symbol"])}
            if not chunk_names <= set(request.symbols):
                raise ValueError("definition archive contains unrequested symbol")
            names.update(chunk_names)
            continue
        normalized = _live_timestamps(pd.Series(array[SPECS[request.schema].time_column]))
        if normalized.isna().any():
            raise ValueError("market archive contains an invalid target timestamp")
        timestamp = normalized.astype("int64").to_numpy() // 1000
        lo, hi = int(timestamp.min()), int(timestamp.max())
        target_min_us = lo if target_min_us is None else min(target_min_us, lo)
        target_max_us = hi if target_max_us is None else max(target_max_us, hi)
        # SDK DataFrameIterator maps these schemas by its first ordered field,
        # ts_recv. BBO snapshots may carry an undefined or stale ts_event.
        frame = pd.DataFrame({"hour": timestamp // HOUR_US, "id": array["instrument_id"],
                              "date": receive // DAY_NS})
        for (hour, instrument_id, day), count in frame.groupby(["hour", "id", "date"], sort=False).size().items():
            key = int(instrument_id), int(day)
            if key not in cache:
                cache[key] = mapping.resolve(key[0], pd.Timestamp(key[1], unit="D").date())
            name = cache[key]
            if not name:
                raise ValueError("archive has an instrument without a valid dated SDK symbol mapping")
            if request.stype_in == "raw_symbol" and name not in request.symbols:
                raise ValueError("market archive contains unrequested symbol")
            if request.stype_in == "parent" and (" " in name or not any(name.startswith(parent.split(".")[0]) for parent in request.symbols)):
                raise ValueError("parent archive contains a symbol outside its futures parent")
            names.add(name)
            bins[(int(hour), name)] += int(count)
    return {"request_id": request.request_id, "request_sha256": request.identity, "path": str(path),
            "schema": request.schema, "dataset": request.dataset, "start": request.start, "end": request.end,
            "records": records, "receive_min_ns": receive_min, "receive_max_ns": receive_max,
            "receive_min": _iso_ns(receive_min), "receive_max": _iso_ns(receive_max),
            "target_min_us": target_min_us, "target_max_us": target_max_us,
            "instrument_ids": sorted(ids), "delivered_symbols": sorted(names),
            "requested_symbols_without_records": sorted(set(request.symbols) - names) if request.stype_in == "raw_symbol" else [],
            "metadata_partial": list(metadata.partial), "metadata_not_found": list(metadata.not_found),
            "hour_symbol_counts": [{"hour": pd.Timestamp(hour * HOUR_US, unit="us", tz="UTC").isoformat(),
                                    "symbol": name, "count": count} for (hour, name), count in sorted(bins.items())],
            "aligned_count": aligned_records,
            "aligned_interval": [item.isoformat() for item in aligned] if aligned else None}


def compare_counts(source_rows: list[dict], database_rows: list[tuple]) -> dict:
    """Allow retained overlap, but reject a deficit in any delivered symbol/hour."""
    existing = {(pd.Timestamp(hour).isoformat(), symbol): count for hour, symbol, count in database_rows}
    deficits, extras = [], 0
    for row in source_rows:
        available = int(existing.get((row["hour"], row["symbol"]), 0))
        if available < row["count"]:
            deficits.append({**row, "database": available, "missing": row["count"] - available})
        extras += max(0, available - row["count"])
    return {"passed": not deficits, "deficit_groups": deficits, "retained_extra_occurrences": extras,
            "interpretation": "Counts establish delivered symbol/hour coverage; retained capture and initial snapshots may add occurrences."}


def validate_database(audits: list[dict], *, connection_factory: Callable | None = None) -> dict:
    """Server-side aggregation grouped across chains sharing each bounded window."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for audit in audits:
        if audit["schema"] != "definition" and audit["records"]:
            groups[(audit["schema"], audit["dataset"], audit["start"], audit["end"])].append(audit)
    results = {audit["request_id"]: {"passed": True, "status": "empty_delivered_archive",
                                    "deficit_groups": [], "retained_extra_occurrences": 0}
               for audit in audits if audit["schema"] != "definition" and not audit["records"]}
    with (connection_factory or (lambda: psycopg.connect(get_dsn())))() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET LOCAL timezone = 'UTC'")
        conn.execute("SET LOCAL statement_timeout = '180s'")
        conn.execute("SET LOCAL lock_timeout = '2s'")
        for (schema, dataset, _, _), members in groups.items():
            spec = SPECS[schema]
            names = sorted({name for member in members for name in member["delivered_symbols"]})
            first = pd.Timestamp(min(member["target_min_us"] for member in members), unit="us", tz="UTC")
            last = pd.Timestamp(max(member["target_max_us"] for member in members) + 1, unit="us", tz="UTC")
            timestamp = sql.Identifier(spec.time_column)
            statement = sql.SQL("SELECT date_trunc('hour',{}),symbol,count(*) FROM {} WHERE {} >= %s AND {} < %s AND symbol=ANY(%s) AND dataset=%s GROUP BY 1,2").format(
                timestamp, sql.Identifier(*spec.table.split(".")), timestamp, timestamp)
            rows = conn.execute(statement, (first.to_pydatetime(), last.to_pydatetime(), names, dataset)).fetchall()
            for member in members:
                results[member["request_id"]] = compare_counts(member["hour_symbol_counts"], rows)
        definition_names = sorted({name for audit in audits if audit["schema"] == "definition" for name in audit["delivered_symbols"]})
        definitions = []
        if definition_names:
            rows = conn.execute("SELECT symbol,source,instrument_class,strike_price,expiration,underlying FROM databento.definitions WHERE symbol=ANY(%s)", (definition_names,)).fetchall()
            usable = {row[0] for row in rows if row[1] == "databento" and row[2] in ("C", "P")
                      and row[3] is not None and np.isfinite(row[3]) and row[3] > 0 and row[4] is not None and row[5]}
            definitions = sorted(set(definition_names) - usable)
    return {"passed": all(row["passed"] for row in results.values()) and not definitions,
            "requests": results, "definition_symbols_without_usable_vendor_row": definitions,
            "limit": "Current definition rows establish vendor provenance, not historical revisions; immutable DBN histories support as-of rebuilds."}


def vendor_count_check(request: RecoveryRequest, audit: dict, *, client: Any | None = None,
                       timeout_seconds: int = 20) -> dict:
    aligned = audit["aligned_interval"]
    if not aligned:
        return {"status": "skipped_no_aligned_interior", "rule_source": COUNT_RULE_URL}
    client = client or db.Historical()
    client.metadata.TIMEOUT = timeout_seconds
    parameters = request.params()
    parameters.update(start=aligned[0], end=aligned[1])
    count = int(client.metadata.get_record_count(**parameters))
    delivered = audit["aligned_count"]
    return {"status": "equal" if delivered == count else "difference_requires_snapshot_or_vendor_review",
            "metadata_count": count, "decoded_aligned_count": delivered, "difference": delivered - count,
            "interval": aligned, "rule_source": COUNT_RULE_URL,
            "limit": "Metadata counts and time-series initial snapshots can differ; unaligned request counts are not parity evidence."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor-counts", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("validation output must be a new artifact")
    enable_strict_dbn_validation()
    requests = load_requests(args.plan)
    receipts = load_download_receipts(args.job_dir)
    evidence = ingestion_evidence(args.job_dir, requests)
    audits, errors, vendor = [], {}, {}
    workers = get_vm_config().workers_for(mem_per_worker_gb=1.0)

    def audit_request(request):
        if request.request_id not in receipts:
            raise ValueError("missing immutable download receipt")
        audit = inspect_archive(request, receipts[request.request_id])
        vendor_result = None
        if args.vendor_counts:
            try:
                vendor_result = vendor_count_check(request, audit)
            except Exception as error:
                vendor_result = {"status": "metadata_error", "error": str(error)}
        return audit, vendor_result

    with ThreadPoolExecutor(max_workers=min(workers, len(requests))) as executor:
        futures = {executor.submit(audit_request, request): request for request in requests}
        for done, future in enumerate(as_completed(futures), 1):
            request = futures[future]
            try:
                audit, vendor_result = future.result()
                audits.append(audit)
                latest = evidence["latest"].get(request.request_id)
                if latest and latest["records"] != audit["records"]:
                    errors[request.request_id] = "independent decoded count differs from ingestion receipt"
                if vendor_result is not None:
                    vendor[request.request_id] = vendor_result
            except Exception as error:
                errors[request.request_id] = f"{type(error).__name__}: {error}"
            filled = int(20 * done / len(requests))
            print(f"[{'█' * filled}{'░' * (20-filled)}] Archive validation {done}/{len(requests)} {request.request_id}", flush=True)
    # Successful ingestion can finish while immutable files are being decoded.
    evidence = ingestion_evidence(args.job_dir, requests)
    for audit in audits:
        latest = evidence["latest"].get(audit["request_id"])
        if latest and latest["records"] != audit["records"]:
            errors[audit["request_id"]] = "independent decoded count differs from ingestion receipt"
    database = {"status": "DEFERRED_UNTIL_ALL_RAW_ACCEPTED"}
    if not args.raw_only and evidence["all_receipts_present"] and not errors:
        try:
            database = validate_database(audits)
        except Exception as error:
            database = {"passed": False, "status": "database_validation_error",
                        "error": f"{type(error).__name__}: {error}"}
    report = {"validated_at": datetime.now(timezone.utc).isoformat(), "raw_passed": not errors,
              "errors": errors, "sources": sorted(audits, key=lambda row: row["request_id"]),
              "ingestion_evidence": evidence, "database": database, "vendor_counts": vendor,
              "unresolved_vendor_source_gap": SOURCE_GAP,
              "limits": ["Sparse vendor source coverage remains unresolved; restoration does not fabricate it.",
                         "Counts and successful append receipts do not independently prove every event payload.",
                         "No downloads, persistent writes, or record replay were performed by this validator."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(f"✅ Validation done in {int(time.monotonic()-started)}s: {args.output}", flush=True)
    return 1 if errors or database.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
