"""Content-addressed code bundle for the GPU execution plane — ADR-022.

Replaces the git-sha code-identity contract (ADR-016). Instead of the GPU VM doing
``git fetch && git checkout <sha>`` (GitHub on the hot path, push/dirty guard, one-sha-
per-process limit), the CPU packs the runtime code into a deterministic tar, zstd-compresses
it, and keys it by the **sha256 of the raw tar**::

    code/<digest>.tzst

uploaded to Object Storage **only if absent**. The job message carries ``code_digest``; the
GPU fetches by digest (cached on its block volume), **verifies the hash**, and extracts to a
per-digest directory. The digest *is* the content, so divergence is structurally impossible
and different digests coexist (concurrent multi-version jobs).

Only the fast-changing code (``src/`` + ``config/``) travels here; slow-changing dependencies
are baked into the GPU base image (see ``Dockerfile.gpu``), so a bundle is typically sub-MB.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import os
import shutil
import tarfile
import threading
from pathlib import Path

import zstandard as zstd

from src.common.object_store import ObjectStore

_BUNDLE_PREFIX = "code"
_INCLUDE_DEFAULT: tuple[str, ...] = ("src", "config")
_SKIP_DIR_PARTS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
_SKIP_SUFFIXES = (".pyc", ".pyo")
_ZSTD_LEVEL = 19


def _iter_files(root: Path, include: tuple[str, ...]):
    """Yield included files under *root*, sorted, skipping caches/compiled artifacts."""
    for top in include:
        base = root / top
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if _SKIP_DIR_PARTS.intersection(p.parts):
                continue
            if p.name.endswith(_SKIP_SUFFIXES):
                continue
            yield p


def build_bundle(
    repo_root: str | Path, include: tuple[str, ...] = _INCLUDE_DEFAULT
) -> tuple[str, bytes]:
    """Build a deterministic, content-addressed code bundle.

    Returns ``(digest, compressed_bytes)`` where ``digest`` is the sha256 hex of the *raw*
    (uncompressed) tar. Metadata is zeroed (mtime/uid/gid/mode fixed, sorted entries) so byte-
    identical inputs always produce the same digest regardless of filesystem timestamps.
    """
    root = Path(repo_root).resolve()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for p in _iter_files(root, include):
            arc = str(p.relative_to(root)).replace(os.sep, "/")
            data = p.read_bytes()
            ti = tarfile.TarInfo(name=arc)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.type = tarfile.REGTYPE
            tf.addfile(ti, io.BytesIO(data))
    raw_bytes = raw.getvalue()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    compressed = zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(raw_bytes)
    return digest, compressed


def bundle_key(digest: str) -> str:
    return f"{_BUNDLE_PREFIX}/{digest}.tzst"


def publish_bundle(store: ObjectStore, digest: str, compressed: bytes) -> str:
    """Upload the bundle **only if absent** (content-addressed → identical bytes never re-upload).
    Returns the object key."""
    key = bundle_key(digest)
    if not store.exists(key):
        store.put(key, compressed)
    return key


def build_and_publish(
    store: ObjectStore, repo_root: str | Path, include: tuple[str, ...] = _INCLUDE_DEFAULT
) -> str:
    """Convenience: build the bundle and publish-if-absent. Returns the ``digest``."""
    digest, compressed = build_bundle(repo_root, include)
    publish_bundle(store, digest, compressed)
    return digest


def _safe_extract(raw_tar: bytes, dest: Path) -> None:
    """Extract a tar from bytes into *dest*, rejecting absolute paths and ``..`` traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest.resolve():
                raise ValueError(f"unsafe path in bundle: {member.name!r}")
        try:
            tf.extractall(dest, filter="data")   # py>=3.12: reject unsafe members + strip metadata
        except TypeError:
            tf.extractall(dest)                  # py<3.12: filter arg unsupported (members pre-validated)


def fetch_and_extract(
    store: ObjectStore, digest: str, cache_root: str | Path
) -> Path:
    """Fetch ``code/<digest>.tzst``, verify its hash, and extract to ``<cache_root>/<digest>``.

    Idempotent + cached: if the destination already carries the completion marker it is returned
    immediately (no download, no re-extract). The sha256 of the decompressed tar MUST equal
    *digest*, else a ``ValueError`` is raised (tamper/corruption guard).

    Concurrency-safe: multiple runner **threads/processes** claiming jobs that pin the same
    fresh digest serialise on an exclusive ``flock`` per digest — the first extracts, the rest
    wait then reuse the marker. (Without this, two workers racing the first extraction shared a
    pid-keyed tmp dir and could publish a marker-stamped but incomplete tree — observed live
    2026-07-11 as ``No module named 'src.gpu_runner'`` on the GPU VM.)
    """
    root = Path(cache_root)
    try:
        root.mkdir(parents=True, exist_ok=True)      # ensure the cache dir exists (GPU block volume)
    except OSError as exc:
        raise RuntimeError(
            f"cannot create bundle cache dir {root} (check VTECH_DATA_DIR is a writable, "
            f"mounted path on this host): {exc}") from exc
    dest = root / digest
    marker = dest / ".bundle_ok"
    if marker.exists():
        return dest

    with open(root / f".{digest}.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if marker.exists():                          # another worker extracted while we waited
            return dest

        key = bundle_key(digest)
        compressed = store.get(key)
        if compressed is None:
            raise KeyError(f"bundle not found in object store: {key}")

        raw_tar = zstd.ZstdDecompressor().decompress(compressed)
        got = hashlib.sha256(raw_tar).hexdigest()
        if got != digest:
            raise ValueError(f"bundle digest mismatch: requested {digest[:12]} got {got[:12]}")

        tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        _safe_extract(raw_tar, tmp)
        (tmp / ".bundle_ok").write_text(digest)
        if dest.exists():                            # stale marker-less tree → replace
            shutil.rmtree(dest)
        tmp.rename(dest)
    return dest
