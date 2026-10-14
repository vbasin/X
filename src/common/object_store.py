"""Shared object-store abstraction — Phase 23.

A minimal put/get/list/delete/exists interface over either a local filesystem
directory (tests / single-host dev) or OCI Object Storage via its S3-compatible
endpoint (boto3). Used by the GPU job queue (``src/gpu_runner/queue.py``) and the
search results index (``src/search/results_index.py``) so both share one storage
implementation and one env-driven backend-selection rule.

Backend selection (``make_object_store``):
  * ``local`` — a local directory (explicit ``local_dir=`` or an env var), fully
    deterministic; no cloud creds needed.
  * ``s3``    — OCI Object Storage via ``MLFLOW_S3_ENDPOINT_URL`` + ``AWS_*`` creds,
    the same path used by MLflow artifacts and ``sync_to_oci.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_DEFAULT_BUCKET = "vtech_hot"


class ObjectStore:
    """Minimal object-store interface: put/get/list/delete/exists on string keys."""

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...          # None if absent
    def list(self, prefix: str) -> list[str]: ...          # full keys under prefix
    def list_sizes(self, prefix: str) -> list[tuple[str, int]]: ...
    def list_meta(self, prefix: str) -> list[tuple[str, int, float]]: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalBackend(ObjectStore):
    """Filesystem-backed store rooted at a directory (for tests / single-host dev)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self._root / key

    def put(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get(self, key: str) -> bytes | None:
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def list(self, prefix: str) -> list[str]:
        return [k for k, _, _ in self.list_meta(prefix)]

    def list_sizes(self, prefix: str) -> list[tuple[str, int]]:
        return [(k, s) for k, s, _ in self.list_meta(prefix)]

    def list_meta(self, prefix: str) -> list[tuple[str, int, float]]:
        # String-prefix match to mirror S3 list_objects_v2 semantics (NOT directory
        # scoped): a prefix may end mid-path-component (e.g. ".../NQ.futures" matching
        # the dir ".../NQ.futures.<hash>/..."). Walk from root and filter by prefix.
        out: list[tuple[str, int, float]] = []
        for f in self._root.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(self._root)).replace(os.sep, "/")
                if rel.startswith(prefix):
                    st = f.stat()
                    out.append((rel, st.st_size, float(st.st_mtime)))
        return sorted(out)

    def delete(self, key: str) -> None:
        p = self._p(key)
        if p.exists():
            p.unlink()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()


class S3Backend(ObjectStore):
    """OCI Object Storage via the S3-compatible endpoint (lazy boto3)."""

    def __init__(self, bucket: str, *, endpoint: str | None = None,
                 region: str | None = None) -> None:
        self._bucket = bucket
        self._endpoint = endpoint or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        self._region = region or os.environ.get("AWS_DEFAULT_REGION")
        self._client: Any = None

    def _c(self) -> Any:
        if self._client is None:
            import boto3  # noqa: PLC0415  (lazy: not needed for the local backend)
            from botocore.config import Config  # noqa: PLC0415
            # OCI Object Storage's S3-compat endpoint rejects botocore's default
            # (>=1.36) streaming CRC32 checksum + aws-chunked encoding with
            # "NotImplemented: AWS chunked encoding not supported". Forcing checksums to
            # when_required restores the pre-1.36 wire format OCI accepts (same class of
            # fix used across the OCI + boto3 ecosystem).
            cfg = Config(
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                max_pool_connections=32,
            )
            self._client = boto3.client(
                "s3", endpoint_url=self._endpoint, region_name=self._region, config=cfg,
            )
        return self._client

    def put(self, key: str, data: bytes) -> None:
        self._c().put_object(Bucket=self._bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError  # noqa: PLC0415
        try:
            return self._c().get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def list(self, prefix: str) -> list[str]:
        return [k for k, _, _ in self.list_meta(prefix)]

    def list_sizes(self, prefix: str) -> list[tuple[str, int]]:
        return [(k, s) for k, s, _ in self.list_meta(prefix)]

    def list_meta(self, prefix: str) -> list[tuple[str, int, float]]:
        out: list[tuple[str, int, float]] = []
        token: str | None = None
        while True:
            kw: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._c().list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                lm = o.get("LastModified")
                ts = float(lm.timestamp()) if lm is not None and hasattr(lm, "timestamp") else 0.0
                out.append((o["Key"], int(o["Size"]), ts))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return sorted(out)

    def delete(self, key: str) -> None:
        self._c().delete_object(Bucket=self._bucket, Key=key)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415
        try:
            self._c().head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False


def make_object_store(
    *,
    local_dir: str | None = None,
    local_env: str | None = None,
    bucket: str | None = None,
    bucket_env: str = "OCI_BUCKET",
    default_bucket: str = _DEFAULT_BUCKET,
    endpoint: str | None = None,
    region: str | None = None,
) -> tuple[ObjectStore, str]:
    """Return ``(store, kind)`` where kind is ``"local"`` or ``"s3"``.

    Selects the LocalBackend when ``local_dir`` is given or ``local_env`` names a
    set environment variable; otherwise the S3Backend against ``bucket`` (or the
    ``bucket_env`` env var, default ``vtech_hot``).
    """
    ld = local_dir or (os.environ.get(local_env) if local_env else None)
    if ld:
        return LocalBackend(ld), "local"
    b = bucket or os.environ.get(bucket_env, default_bucket)
    return S3Backend(b, endpoint=endpoint, region=region), "s3"
