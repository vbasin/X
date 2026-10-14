"""GPU job_exec parquet size-sync (stale GPU files vs vtech_hot)."""

from __future__ import annotations

import os
from pathlib import Path

from src.common.object_store import LocalBackend
from src.gpu_runner.job_exec import (
    _sync_context_prefixes, _sync_data_prefix, _sync_labels_prefix,
    reconstruct_catalog_entry,
)
from src.gpu_runner.parquet_sync import (
    gpu_context_prefixes, gpu_feature_mirror, push_features,
    push_job_training_data, pull_features,
)
from src.models.catalog import get_entry


def test_sync_data_prefix_replaces_stale_size(tmp_path, monkeypatch):
    store = LocalBackend(tmp_path / "store")
    store.put("parquet/features_2026-04-07.parquet", b"NEWCONTENT")
    store.put("parquet/other.txt", b"ignore")
    dest = tmp_path / "parquet_5s"
    dest.mkdir()
    stale = dest / "features_2026-04-07.parquet"
    stale.write_bytes(b"OLD")
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "store"))
    _sync_data_prefix("parquet/", dest)
    assert stale.read_bytes() == b"NEWCONTENT"


def test_sync_data_prefix_skips_matching_size(tmp_path, monkeypatch):
    store = LocalBackend(tmp_path / "store")
    payload = b"SAME"
    store.put("parquet/features_2026-04-07.parquet", payload)
    dest = tmp_path / "parquet_5s"
    dest.mkdir()
    (dest / "features_2026-04-07.parquet").write_bytes(payload)
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "store"))
    _sync_data_prefix("parquet/", dest)
    assert (dest / "features_2026-04-07.parquet").read_bytes() == payload


def test_list_sizes_local(tmp_path):
    store = LocalBackend(tmp_path / "store")
    store.put("parquet/features_a.parquet", b"abcd")
    got = store.list_sizes("parquet/")
    assert got == [("parquet/features_a.parquet", 4)]
    assert store.list("parquet/") == ["parquet/features_a.parquet"]
    meta = store.list_meta("parquet/")
    assert len(meta) == 1 and meta[0][0] == "parquet/features_a.parquet" and meta[0][1] == 4


def test_sync_replaces_same_size_when_remote_newer(tmp_path, monkeypatch):
    store = LocalBackend(tmp_path / "store")
    dest = tmp_path / "parquet_5s"
    dest.mkdir()
    stale = dest / "features_2026-04-07.parquet"
    stale.write_bytes(b"YYYY")
    os.utime(stale, (1_000_000, 1_000_000))
    store.put("parquet/features_2026-04-07.parquet", b"XXXX")
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "store"))
    _sync_data_prefix("parquet/", dest)
    assert stale.read_bytes() == b"XXXX"


def test_pull_features_labels_glob(tmp_path, monkeypatch):
    from src.gpu_runner.parquet_sync import pull_features

    store = LocalBackend(tmp_path / "store")
    store.put("parquet_pathlabels/labels_2018-01-02.parquet", b"L1")
    store.put("parquet_pathlabels/labels_2018-01-03.parquet", b"L2")
    store.put("parquet_pathlabels/other.txt", b"ignore")
    dest = tmp_path / "pathlabels"
    dest.mkdir()
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "store"))
    n_dl, n_skip = pull_features("parquet_pathlabels/", dest, store=store,
                                 glob="labels_*.parquet")
    assert n_dl == 2 and n_skip == 0
    assert (dest / "labels_2018-01-02.parquet").read_bytes() == b"L1"
    assert (dest / "labels_2018-01-03.parquet").read_bytes() == b"L2"
    assert not (dest / "other.txt").exists()


def test_push_uploads_newer_local(tmp_path):
    store = LocalBackend(tmp_path / "store")
    store.put("parquet/features_2026-04-07.parquet", b"OLD!")
    remote = tmp_path / "store" / "parquet" / "features_2026-04-07.parquet"
    os.utime(remote, (1_000_000, 1_000_000))
    src = tmp_path / "parquet_5s"
    src.mkdir()
    f = src / "features_2026-04-07.parquet"
    f.write_bytes(b"NEW!")
    n_up, _n_skip = push_features("parquet/", src, store=store)
    assert n_up == 1
    assert store.get("parquet/features_2026-04-07.parquet") == b"NEW!"


def test_gpu_feature_mirror_15s_symbol_dir_not_basename():
    """The failed Custom Test used pdir=.../timeframe=15s/symbol=NQ → prefix symbol=NQ/."""
    pdir = Path("/opt/vtech/data/sources/features/timeframe=15s/symbol=NQ")
    root, prefix = gpu_feature_mirror(pdir)
    assert prefix == "timeframe=15s/"
    assert root.name == "timeframe=15s"
    leftover5 = gpu_feature_mirror(Path("/opt/vtech/data/sources/features/parquet_5s"))
    assert leftover5[1] == "timeframe=5s/"
    assert leftover5[0].name == "timeframe=5s"
    hive5 = Path("/opt/vtech/data/sources/features/timeframe=5s/symbol=NQ")
    assert gpu_feature_mirror(hive5)[1] == "timeframe=5s/"
    assert gpu_feature_mirror(hive5)[0].name == "timeframe=5s"
    assert gpu_feature_mirror(Path("/opt/vtech/data/sources/features/parquet_raw1s"))[1] == (
        "parquet_raw1s/"
    )


def test_push_pull_hive_keeps_same_basename_in_two_months(tmp_path):
    store = LocalBackend(tmp_path / "store")
    hive = tmp_path / "timeframe=15s"
    a = hive / "symbol=NQ" / "year=2025" / "month=04" / "features_15s_part1.parquet"
    b = hive / "symbol=NQ" / "year=2025" / "month=05" / "features_15s_part1.parquet"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"APRIL")
    b.write_bytes(b"MAY!!")
    n_up, n_skip = push_features("timeframe=15s/", hive, store=store)
    assert n_up == 2 and n_skip == 0
    assert store.get(
        "timeframe=15s/symbol=NQ/year=2025/month=04/features_15s_part1.parquet"
    ) == b"APRIL"
    dest = tmp_path / "gpu" / "timeframe=15s"
    n_dl, _ = pull_features("timeframe=15s/", dest, store=store)
    assert n_dl == 2
    assert (dest / "symbol=NQ" / "year=2025" / "month=04"
            / "features_15s_part1.parquet").read_bytes() == b"APRIL"
    assert (dest / "symbol=NQ" / "year=2025" / "month=05"
            / "features_15s_part1.parquet").read_bytes() == b"MAY!!"
    assert not (dest / "features_15s_part1.parquet").exists()


def test_push_job_training_data_15s_payload(tmp_path, monkeypatch):
    from src.training.label_store import label_set_dir, tbm_contract_from_objective, write_contract

    hive = tmp_path / "timeframe=15s"
    part = hive / "symbol=NQ" / "year=2026" / "month=06" / "features_15s_part1.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"FEAT")
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path / "data"))
    objective = {
        "search": {"topology": {"feature_store": "15s"}},
        "labels": {"encoding": "dual_binary"},
    }
    lab_dir = label_set_dir("15s", tbm_contract_from_objective(objective))
    lab_dir.mkdir(parents=True)
    (lab_dir / "labels_15s_part1.parquet").write_bytes(b"LAB")
    write_contract(lab_dir, tbm_contract_from_objective(objective))
    store = LocalBackend(tmp_path / "store")
    payload = push_job_training_data(
        objective,
        hive / "symbol=NQ",
        store=store,
    )
    assert payload["data_s3_prefix"] == "timeframe=15s/"
    assert payload["data_s3_dest"] == str(hive)
    assert payload["labels_s3_prefix"] == f"labels/15s/{lab_dir.name}/"
    assert payload["labels_kind"] == "15s"
    assert store.get(
        "timeframe=15s/symbol=NQ/year=2026/month=06/features_15s_part1.parquet"
    ) == b"FEAT"
    assert store.get(f"labels/15s/{lab_dir.name}/labels_15s_part1.parquet") == b"LAB"


def test_sync_labels_prefix_15s(tmp_path, monkeypatch):
    store = LocalBackend(tmp_path / "store")
    store.put("labels/15s/test/labels_15s_part1.parquet", b"L15")
    dest = tmp_path / "gpu_labels"
    dest.mkdir()
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "store"))
    _sync_labels_prefix({
        "labels_s3_prefix": "labels/15s/test/",
        "labels_kind": "15s",
        "labels_s3_dest": str(dest),
    })
    assert (dest / "labels_15s_part1.parquet").read_bytes() == b"L15"


def test_gpu_context_prefixes_follow_topology_context():
    assert gpu_context_prefixes({"search": {"topology": {"feature_store": "15s"}}}) == []
    obj = {
        "search": {
            "topology": {
                "feature_store": "15s",
                "context": [
                    {"native_tf": "1h", "role": "context"},
                    {"native_tf": "1d", "role": "context"},
                ],
            }
        }
    }
    assert gpu_context_prefixes(obj) == ["timeframe=1h/", "timeframe=1d/"]
    obj["search"]["topology"]["feature_store"] = "1h"
    assert gpu_context_prefixes(obj) == []


def test_push_job_training_data_includes_htf_context(tmp_path, monkeypatch):
    from src.training.label_store import label_set_dir, tbm_contract_from_objective, write_contract

    hive = tmp_path / "timeframe=15s"
    part = hive / "symbol=NQ" / "year=2026" / "month=06" / "features_15s_part1.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"FEAT")
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path / "data"))
    objective = {
        "search": {"topology": {
            "feature_store": "15s",
            "context": [
                {"native_tf": "1h", "role": "context"},
                {"native_tf": "1d", "role": "context"},
            ],
        }},
        "labels": {"encoding": "dual_binary"},
    }
    lab_dir = label_set_dir("15s", tbm_contract_from_objective(objective))
    lab_dir.mkdir(parents=True)
    (lab_dir / "labels_15s_part1.parquet").write_bytes(b"LAB")
    write_contract(lab_dir, tbm_contract_from_objective(objective))
    feats = tmp_path / "features"
    for tf, blob in (("1h", b"H1"), ("1d", b"D1")):
        p = feats / f"timeframe={tf}" / "symbol=NQ" / "year=2026" / "month=06" / f"features_{tf}_part1.parquet"
        p.parent.mkdir(parents=True)
        p.write_bytes(blob)
    store = LocalBackend(tmp_path / "store")
    monkeypatch.setattr(
        "src.common.data_paths.native_tf_features_root", lambda: feats,
    )
    payload = push_job_training_data(
        objective,
        hive / "symbol=NQ",
        store=store,
    )
    assert payload["context_s3_prefixes"] == ["timeframe=1h/", "timeframe=1d/"]
    assert store.get(
        "timeframe=1h/symbol=NQ/year=2026/month=06/features_1h_part1.parquet"
    ) == b"H1"
    assert store.get(
        "timeframe=1d/symbol=NQ/year=2026/month=06/features_1d_part1.parquet"
    ) == b"D1"


def test_sync_context_prefixes(tmp_path, monkeypatch):
    store = LocalBackend(tmp_path / "store")
    store.put("timeframe=1h/symbol=NQ/year=2026/month=06/features_1h_part1.parquet", b"H1")
    dest_root = tmp_path / "gpu_features"
    dest_root.mkdir()
    monkeypatch.setenv("VTECH_GPU_QUEUE_LOCAL", str(tmp_path / "store"))
    monkeypatch.setattr(
        "src.common.data_paths.native_tf_features_root", lambda: dest_root,
    )
    _sync_context_prefixes({"context_s3_prefixes": ["timeframe=1h/"]})
    got = dest_root / "timeframe=1h" / "symbol=NQ" / "year=2026" / "month=06" / "features_1h_part1.parquet"
    assert got.read_bytes() == b"H1"


def test_reconstruct_clears_search_space_without_hpo():
    base = get_entry("xgboost")
    entry = reconstruct_catalog_entry({
        "family": "xgboost",
        "candidate_id": "xgboost",
        "spec_params": {"max_depth": 6},
        "feature_selection": {"view": "regime_core"},
        "search_max_rows": None,
    })
    assert entry.name == "xgboost"
    assert entry.plugin_family == "xgboost"
    assert entry.params_default["max_depth"] == 6
    assert entry.params_default["n_estimators"] == base.params_default["n_estimators"]
    assert entry.search_space == {}
    assert entry.feature_selection == {"view": "regime_core"}


def test_reconstruct_keeps_opt_in_search_space():
    space = {"max_depth": {"type": "int", "low": 4, "high": 12}}
    entry = reconstruct_catalog_entry({
        "family": "xgboost",
        "candidate_id": "xgboost",
        "spec_params": {"colsample_bytree": 1.0},
        "search_space": space,
        "tune_trials": 12,
    })
    assert entry.search_space == space
    assert entry.params_default["colsample_bytree"] == 1.0
