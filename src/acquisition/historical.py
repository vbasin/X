"""Historical data backfill from Databento."""

from __future__ import annotations

import logging
from pathlib import Path

import databento as db

from src.acquisition.schemas import DATASET

logger = logging.getLogger("vtech.acquisition.historical")


def cost_check(
    symbols: str | list[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str = "parent",
) -> float:
    """Return estimated cost (USD) without downloading anything."""
    client = db.Historical()
    cost = client.metadata.get_cost(
        dataset=DATASET,
        symbols=symbols,
        schema=schema,
        stype_in=stype_in,
        start=start,
        end=end,
    )
    logger.info("cost_check", extra={"symbols": symbols, "schema": schema, "cost": cost})
    return cost


def download_range(
    symbols: str | list[str],
    schema: str,
    start: str,
    end: str,
    output_dir: Path,
    stype_in: str = "parent",
) -> Path:
    """Stream historical data to a local .dbn.zst file.

    Best for requests <5 GB. For larger payloads use submit_batch().
    """
    client = db.Historical()
    output_dir.mkdir(parents=True, exist_ok=True)

    sym_label = symbols if isinstance(symbols, str) else "_".join(symbols)
    filename = f"{sym_label}_{schema}_{start}_{end}.dbn.zst".replace(".", "_")
    out_path = output_dir / filename

    logger.info("download_range start", extra={
        "symbols": symbols, "schema": schema, "start": start, "end": end, "path": str(out_path),
    })

    client.timeseries.get_range(
        dataset=DATASET,
        symbols=symbols,
        schema=schema,
        stype_in=stype_in,
        start=start,
        end=end,
        path=str(out_path),
    )

    logger.info("download_range done", extra={"path": str(out_path), "size_mb": out_path.stat().st_size / 1e6})
    return out_path


def submit_batch(
    symbols: str | list[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str = "parent",
    split_duration: str = "day",
) -> str:
    """Submit a batch job for large requests (>5 GB). Returns job ID."""
    client = db.Historical()
    job = client.batch.submit_job(
        dataset=DATASET,
        symbols=symbols,
        schema=schema,
        stype_in=stype_in,
        start=start,
        end=end,
        encoding="dbn",
        compression="zstd",
        split_duration=split_duration,
        split_symbols=False,
    )
    job_id = job["id"]
    logger.info("batch_submitted", extra={"job_id": job_id, "symbols": symbols, "schema": schema})
    return job_id


def download_batch(job_id: str, output_dir: Path) -> list[Path]:
    """Download files from a completed batch job."""
    client = db.Historical()
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = client.batch.download(job_id=job_id, output_dir=str(output_dir))
    paths = [output_dir / f for f in filenames]
    logger.info("batch_downloaded", extra={"job_id": job_id, "files": len(paths)})
    return paths


def poll_batch_until_done(job_id: str, poll_secs: int = 30) -> None:
    """Block until the batch job reaches state='done'. Raises on failure/expiry."""
    import time

    client = db.Historical()
    while True:
        jobs = client.batch.list_jobs()
        job = next((j for j in jobs if j.get("id") == job_id), None)
        if job is None:
            raise RuntimeError(f"Batch job {job_id!r} not found in list_jobs()")
        state = job.get("state", "unknown")
        progress = job.get("progress", "?")
        logger.info("batch_poll", extra={"job_id": job_id, "state": state, "progress": progress})
        print(f"  [{time.strftime('%H:%M:%S')}] batch {job_id}: state={state} progress={progress}%", flush=True)
        if state == "done":
            return
        if state in ("failed", "expired", "error"):
            raise RuntimeError(f"Batch job {job_id!r} ended with state={state!r}")
        time.sleep(poll_secs)
