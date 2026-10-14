"""Tests for the content-addressed code bundle (ADR-022). No cloud required."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.code_bundle import (
    build_and_publish,
    build_bundle,
    bundle_key,
    fetch_and_extract,
    publish_bundle,
)
from src.common.object_store import make_object_store


def _make_repo(root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_text("print('hello')\n")
    (root / "config" / "obj.yaml").write_text("objective_id: nq-50pt\n")
    # noise that must be excluded from the digest
    (root / "src" / "pkg" / "__pycache__").mkdir()
    (root / "src" / "pkg" / "__pycache__" / "mod.pyc").write_bytes(b"\x00\x01")
    (root / "data").mkdir()
    (root / "data" / "big.parquet").write_bytes(b"x" * 1000)


def test_digest_is_deterministic_and_ignores_noise(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    d1, _ = build_bundle(repo)
    d2, comp = build_bundle(repo)
    assert d1 == d2 and len(d1) == 64
    # __pycache__/.pyc and data/ are excluded → changing them does not change the digest
    (repo / "src" / "pkg" / "__pycache__" / "mod.pyc").write_bytes(b"\xff")
    (repo / "data" / "big.parquet").write_bytes(b"y" * 2000)
    d3, _ = build_bundle(repo)
    assert d3 == d1


def test_editing_code_changes_digest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    d1, _ = build_bundle(repo)
    (repo / "src" / "pkg" / "mod.py").write_text("print('changed')\n")
    d2, _ = build_bundle(repo)
    assert d1 != d2


def test_publish_is_idempotent_and_skips_when_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    store, kind = make_object_store(local_dir=str(tmp_path / "store"))
    assert kind == "local"
    digest, comp = build_bundle(repo)
    key = publish_bundle(store, digest, comp)
    assert key == bundle_key(digest)
    assert store.exists(key)
    # second publish is a no-op (already present) — bytes unchanged
    publish_bundle(store, digest, comp)
    assert store.get(key) == comp


def test_fetch_and_extract_roundtrip_and_cache(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    store, _ = make_object_store(local_dir=str(tmp_path / "store"))
    digest = build_and_publish(store, repo)

    cache = tmp_path / "cache"
    dest = fetch_and_extract(store, digest, cache)
    assert (dest / "src" / "pkg" / "mod.py").read_text() == "print('hello')\n"
    assert (dest / "config" / "obj.yaml").read_text() == "objective_id: nq-50pt\n"
    assert (dest / ".bundle_ok").read_text() == digest

    # cache hit: deleting the object still returns the cached extraction (no re-download)
    store.delete(bundle_key(digest))
    dest2 = fetch_and_extract(store, digest, cache)
    assert dest2 == dest


def test_fetch_rejects_tampered_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    store, _ = make_object_store(local_dir=str(tmp_path / "store"))
    digest, comp = build_bundle(repo)
    publish_bundle(store, digest, comp)
    # Corrupt the stored bytes under the (unchanged) digest key → hash verification must fail.
    other_digest, other_comp = build_bundle(repo / "..")  # different content
    store.put(bundle_key(digest), other_comp)
    with pytest.raises(ValueError):
        fetch_and_extract(store, digest, tmp_path / "cache")
