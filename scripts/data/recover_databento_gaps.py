#!/usr/bin/env python3
"""Execute the reviewed October recovery with durable per-request receipts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import databento as db
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.acquisition.definition_acquisition import DEFINITION_COLUMNS, DEFINITION_UPSERT_TAIL, VendorDefinition
from src.acquisition.gap_recovery import RecoveryBatchError, RecoveryRequest, download_requests, find_saved_receipt, load_requests
from src.common.vm_config import get_vm_config
from src.ingestion.gap_recovery import enable_strict_dbn_validation, ingest_recovery_file
from src.ingestion.loader import get_dsn


def write_result(directory: Path, name: str, result: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (name + "_" + uuid4().hex + ".json")
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, default=str)
    return destination


def ingest_definitions(path: Path, request: RecoveryRequest) -> dict[str, Any]:
    """Archive history remains intact; upgrade synthetic/missing current rows."""
    import csv
    import io
    from psycopg import sql

    store = db.DBNStore.from_file(path)
    if str(store.schema) != "definition" or str(store.dataset) != request.dataset:
        raise ValueError("definition archive metadata mismatch")
    latest = {}
    latest_publication_ns = {}
    records = 0
    for record in store:
        if not hasattr(record, "raw_symbol"):
            continue
        definition = VendorDefinition.from_record(record)
        if definition.symbol not in request.symbols:
            raise ValueError("unrequested definition symbol")
        records += 1
        publication_ns = int(record.ts_event)
        if publication_ns >= latest_publication_ns.get(definition.symbol, -1):
            latest[definition.symbol] = definition
            latest_publication_ns[definition.symbol] = publication_ns
    if not latest:
        raise ValueError("requested definition archive contains no definitions")
    values = pd.DataFrame([definition.row() for definition in latest.values()], columns=DEFINITION_COLUMNS)
    columns = sql.SQL(",").join(map(sql.Identifier, DEFINITION_COLUMNS))
    with psycopg.connect(get_dsn()) as conn:
        before = conn.execute("SELECT source,count(*) FROM databento.definitions WHERE symbol = ANY(%s) GROUP BY source",
                              (list(latest),)).fetchall()
        conn.execute("SET lock_timeout='2s'")
        conn.execute("CREATE TEMP TABLE recovery_definitions AS SELECT * FROM databento.definitions WITH NO DATA")
        buffer = io.StringIO()
        values.to_csv(buffer, header=False, index=False, na_rep="", quoting=csv.QUOTE_MINIMAL)
        with conn.cursor().copy(sql.SQL("COPY recovery_definitions ({}) FROM STDIN WITH (FORMAT CSV)").format(columns)) as copy:
            copy.write(buffer.getvalue())
        result = conn.execute(sql.SQL("INSERT INTO databento.definitions ({}) SELECT {} FROM recovery_definitions ").format(columns, columns)
                              + sql.SQL(DEFINITION_UPSERT_TAIL))
        changed = result.rowcount
        after = conn.execute("SELECT source,count(*) FROM databento.definitions WHERE symbol = ANY(%s) GROUP BY source",
                             (list(latest),)).fetchall()
    return {"request_id": request.request_id, "records": records, "symbols": len(latest),
            "absent_from_delivery": sorted(set(request.symbols) - set(latest)),
            "changed": changed, "before": before, "after": after}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/operations/DATABENTO_GAP_BACKFILL_REQUESTS_2026_10_06.json"))
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("download", "ingest", "all"), default="all")
    args = parser.parse_args()
    enable_strict_dbn_validation()
    started = time.monotonic()
    requests = load_requests(args.plan)
    args.job_dir.mkdir(parents=True, exist_ok=True)
    with (args.job_dir / "run.lock").open("a") as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def progress(done, total, request, result):
            filled = int(20 * done / total)
            outcome = "ERROR " + type(result).__name__ if isinstance(result, Exception) else "saved"
            print(f"[{'█' * filled}{'░' * (20-filled)}] Download {done}/{total} {request.request_id}: {outcome}", flush=True)

        if args.phase == "ingest":
            receipts = [find_saved_receipt(request, args.job_dir) for request in requests]
            if any(receipt is None for receipt in receipts):
                parser.error("ingest requires all hash-verified archives; run download or all for missing files")
        else:
            try:
                receipts = download_requests(requests, job_dir=args.job_dir, progress=progress)
            except RecoveryBatchError as error:
                write_result(args.job_dir, "acquisition_errors", {key: str(value) for key, value in error.errors.items()})
                print(str(error), flush=True)
                return 2
        write_result(args.job_dir, "downloads", {receipt.request_id: asdict(receipt) for receipt in receipts})
        if args.phase == "download":
            print(f"✅ Download complete in {int(time.monotonic()-started)}s", flush=True)
            return 0
        by_id = {receipt.request_id: receipt for receipt in receipts}
        results = []
        for request in requests:
            if request.schema != "definition":
                continue
            result = ingest_definitions(Path(by_id[request.request_id].path), request)
            write_result(args.job_dir / "ingestion", request.identity, result)
            results.append(result)
            print(f"Definitions {request.request_id}: {result['changed']} changed, {result['symbols']} supplied", flush=True)
        market = [request for request in requests if request.schema != "definition"]
        workers = get_vm_config().workers_for(mem_per_worker_gb=7.0)
        errors = {}

        def recover(request):
            receipt = ingest_recovery_file(by_id[request.request_id].path, request,
                progress=lambda stage, count: print(f"{request.request_id}: {stage} {count:,}", flush=True))
            result = asdict(receipt)
            write_result(args.job_dir / "ingestion", request.identity, result)
            return result

        with ThreadPoolExecutor(max_workers=min(workers, len(market))) as executor:
            futures = {executor.submit(recover, request): request for request in market}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"ACCEPTED {request.request_id}: {result['inserted']:,} inserted; {result['matched']:,} matched", flush=True)
                except Exception as error:
                    errors[request.request_id] = str(error)
                    print(f"FAILED {request.request_id}: {type(error).__name__}: {error}", flush=True)
        report = {"completed_at": datetime.now(timezone.utc).isoformat(), "requests": len(requests),
                  "results": results, "errors": errors, "raw_accepted": not errors,
                  "unresolved_source_gap": ["2026-10-05T22:50:00Z", "2026-10-06T00:00:00Z"]}
        write_result(args.job_dir, "recovery_report", report)
        print(f"✅ Recovery finished in {int(time.monotonic()-started)}s; failures={len(errors)}", flush=True)
        return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
