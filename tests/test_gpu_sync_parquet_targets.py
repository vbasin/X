"""GPU boot parquet prefixes must be declared in oci_sync.conf."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path("/opt/vtech")


def _load_sync_parquet():
    spec = importlib.util.spec_from_file_location(
        "gpu_sync_parquet", REPO / "deploy" / "gpu" / "sync_parquet.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _oci_hot_prefixes() -> set[str]:
    text = (REPO / "config" / "runtime" / "oci_sync.conf").read_text()
    prefixes: set[str] = set()
    for m in re.finditer(
        r'^  "([^"|]+)\|([^"|]+)\|([^"|]+)"\s*$', text, re.MULTILINE,
    ):
        _local, bucket, prefix = m.group(1), m.group(2), m.group(3)
        if bucket == "vtech_hot":
            prefixes.add(prefix)
    return prefixes


def test_gpu_boot_prefixes_are_oci_sync_targets() -> None:
    mod = _load_sync_parquet()
    hot = _oci_hot_prefixes()
    missing = [p for p, _subdir in mod.DATASETS if p not in hot]
    assert missing == [], f"GPU boot prefixes missing from oci_sync.conf: {missing}"


def test_gpu_boot_has_no_legacy_parquet_prefix() -> None:
    mod = _load_sync_parquet()
    prefixes = [p for p, _ in mod.DATASETS]
    assert "parquet/" not in prefixes
    assert "timeframe=5s/" in prefixes
    text = (REPO / "config" / "runtime" / "oci_sync.conf").read_text()
    assert "|vtech_hot|parquet/" not in text


def test_features_root_from_parquet_5s_arg() -> None:
    mod = _load_sync_parquet()
    root = Path("/opt/vtech/data/sources/features")
    assert mod.features_root_from_arg(root / "parquet_5s") == root
    assert mod.features_root_from_arg(root / "parquet") == root
    assert mod.features_root_from_arg(root) == root


def test_oci_labels_15s_target_is_live_dir() -> None:
    from src.common.data_paths import parquet_labels_15s_dir

    text = (REPO / "config" / "runtime" / "oci_sync.conf").read_text()
    local = None
    for m in re.finditer(
        r'^  "([^"|]+)\|([^"|]+)\|([^"|]+)"\s*$', text, re.MULTILINE,
    ):
        path, bucket, prefix = m.group(1), m.group(2), m.group(3)
        if bucket == "vtech_hot" and prefix == "parquet_labels_15s/":
            local = Path(path)
    assert local is not None
    assert local.resolve() == parquet_labels_15s_dir().resolve()
    assert local.is_dir()
