#!/usr/bin/env python3
"""
sync_parquet.py — boto3-based replacement for `aws s3 sync`.

Pulls GPU-consumed feature caches from vtech_hot using the S3-compatible
OCI Object Storage endpoint. Skips files whose size already matches;
does NOT delete local files that are absent from the bucket (safe default).

Usage (called by bootstrap.sh and the systemd ExecStartPre):
    python sync_parquet.py <sources/features>

Required environment variables (loaded from .env before this is called):
    MLFLOW_S3_ENDPOINT_URL  — OCI S3-compat endpoint
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION      — e.g. us-phoenix-1
"""

import os
import sys
import pathlib
import boto3
from botocore.exceptions import BotoCoreError, ClientError

BUCKET = "vtech_hot"

# Must match config/runtime/oci_sync.conf vtech_hot feature prefixes.
# (prefix in bucket, subdirectory under sources/features/)
# Keep in sync with src.common.data_paths.HIVE_TIMEFRAMES.
_HIVE_TIMEFRAMES = ("5s", "15s", "1m", "5m", "1h", "1d")
_FLAT_DATASETS: tuple[tuple[str, str], ...] = (
    ("parquet_raw1s/", "parquet_raw1s"),
    ("parquet_path1s/", "parquet_path1s"),
    ("parquet_pathlabels/", "parquet_pathlabels"),
    ("parquet_labels_15s/", "parquet_labels_15s"),
    ("parquet_labels_5s/", "parquet_labels_5s"),
)
DATASETS: tuple[tuple[str, str], ...] = _FLAT_DATASETS + tuple(
    (f"timeframe={tf}/", f"timeframe={tf}") for tf in _HIVE_TIMEFRAMES
)


def features_root_from_arg(path: pathlib.Path) -> pathlib.Path:
    """Accept ``sources/features`` or a leftover parquet_5s / hive-tf dest."""
    if path.name in ("parquet_5s", "parquet") or path.name.startswith("timeframe="):
        return path.parent
    return path


def sync_prefix(s3, dest_dir: pathlib.Path, prefix: str) -> tuple[int, int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = skipped = errors = 0
    pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel or rel.endswith("/"):
                continue

            local = dest_dir / rel
            if local.exists() and local.stat().st_size == obj["Size"]:
                skipped += 1
                continue

            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(BUCKET, key, str(local))
                downloaded += 1
            except (BotoCoreError, ClientError) as exc:
                print(f"  WARNING: failed to download {key}: {exc}", file=sys.stderr)
                errors += 1
    return downloaded, skipped, errors


def sync(dest_dir: pathlib.Path) -> int:
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    if not endpoint:
        print("ERROR: MLFLOW_S3_ENDPOINT_URL is not set", file=sys.stderr)
        return 1

    s3 = boto3.client("s3", endpoint_url=endpoint)
    root = features_root_from_arg(dest_dir)
    total_dl = total_sk = total_err = 0

    for prefix, subdir in DATASETS:
        dest = root / subdir
        try:
            downloaded, skipped, errors = sync_prefix(s3, dest, prefix)
        except (BotoCoreError, ClientError) as exc:
            print(f"ERROR: S3 list failed prefix={prefix}: {exc}", file=sys.stderr)
            return 1
        total_dl += downloaded
        total_sk += skipped
        total_err += errors
        print(
            f"    {subdir}: {downloaded} downloaded, {skipped} skipped, {errors} errors"
        )

    print(
        f"    Parquet sync: {total_dl} downloaded, {total_sk} skipped, {total_err} errors"
    )
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dest_dir>", file=sys.stderr)
        sys.exit(1)
    sys.exit(sync(pathlib.Path(sys.argv[1])))
