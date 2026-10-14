"""5s hive migration (ADR-079): day files → monthly parts, bar_end_ts, latest-mtime wins."""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from src.features.hive_5s import (
    concat_keep_latest,
    frame_with_bar_end_ts,
    list_5s_day_files,
    migrate_parquet_5s,
)
from src.features.native_tf import load_hive_frame, write_month_parts
from src.training.data_pipeline import DataPipeline, ensure_utc_index


def test_list_5s_day_files_skips_oof(tmp_path):
    (tmp_path / "features_2026-04-07.parquet").write_bytes(b"x")
    (tmp_path / "oof_xgb_tbm_full.parquet").write_bytes(b"y")
    (tmp_path / "features_nope.parquet").write_bytes(b"z")
    got = [p.name for p in list_5s_day_files(tmp_path)]
    assert got == ["features_2026-04-07.parquet"]


def test_frame_with_bar_end_ts_from_index():
    df = pd.DataFrame(
        {"cs_close": [1.0, 2.0]},
        index=pd.to_datetime(["2026-04-07 13:00:00+00:00", "2026-04-07 13:00:05+00:00"]),
    )
    out = frame_with_bar_end_ts(df)
    assert list(out.columns[:1]) == ["bar_end_ts"]
    assert str(out["bar_end_ts"].dtype).startswith("datetime64")
    assert out["bar_end_ts"].dt.tz is not None


def test_concat_keep_latest_prefers_newer_mtime():
    ts = pd.Timestamp("2026-04-07 13:00:00", tz="UTC")
    a = pd.DataFrame({"bar_end_ts": [ts], "cs_close": [1.0], "_src_mtime_ns": [10], "_src_ord": [0]})
    b = pd.DataFrame({"bar_end_ts": [ts], "cs_close": [9.0], "_src_mtime_ns": [20], "_src_ord": [0]})
    out = concat_keep_latest([a, b])
    assert len(out) == 1
    assert float(out["cs_close"].iloc[0]) == 9.0


def test_migrate_writes_hive_keeps_source(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    src = tmp_path / "sources" / "features" / "parquet_5s"
    src.mkdir(parents=True)
    idx = pd.date_range("2026-04-07 13:00:00", periods=3, freq="5s", tz="UTC")
    day = pd.DataFrame({"cs_close": [1.0, 2.0, 3.0], "bp_spread_mean": [0.1, 0.2, 0.3]}, index=idx)
    src_file = src / "features_2026-04-07.parquet"
    day.to_parquet(src_file)

    written = migrate_parquet_5s(src_dir=src, workers=1, max_bytes=50_000_000)
    assert src_file.exists()
    assert len(written) == 1
    assert written[0].name == "features_5s_part1.parquet"
    assert written[0].parent.name == "month=04"
    got = pd.read_parquet(written[0])
    assert list(got.columns[:1]) == ["bar_end_ts"]
    assert got.index.name is None or not isinstance(got.index, pd.DatetimeIndex) or got.index.name != "bar_end_ts"
    assert len(got) == 3
    assert got["bar_end_ts"].is_monotonic_increasing

    loaded = load_hive_frame("5s", "2026-04-07", "2026-04-07")
    assert len(loaded) == 3
    assert loaded.index.name == "ts"
    assert "cs_close" in loaded.columns


def test_migrate_duplicate_keeps_later_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    src = tmp_path / "parquet_5s"
    src.mkdir()
    ts = pd.Timestamp("2026-08-26 13:02:00", tz="UTC")
    a = pd.DataFrame({"cs_close": [1.0, 2.0]}, index=pd.to_datetime(
        ["2026-08-25 19:44:55+00:00", ts],
    ))
    b = pd.DataFrame({"cs_close": [9.0, 3.0]}, index=pd.to_datetime(
        [ts, "2026-08-26 13:05:00+00:00"],
    ))
    pa = src / "features_2026-08-25.parquet"
    pb = src / "features_2026-08-26.parquet"
    a.to_parquet(pa)
    time.sleep(0.02)
    b.to_parquet(pb)
    os.utime(pb, None)
    migrate_parquet_5s(src_dir=src, workers=1)
    loaded = load_hive_frame("5s")
    assert loaded.index.is_unique
    assert float(loaded.loc[ts, "cs_close"]) == 9.0
    assert len(loaded) == 3


def test_write_5s_parts_split_on_size(tmp_path):
    n = 400
    ts = pd.date_range("2026-04-07 13:00:00", periods=n, freq="5s", tz="UTC")
    df = pd.DataFrame({
        "bar_end_ts": ts,
        **{f"c{i}": pd.Series(range(n), dtype="float64") for i in range(30)},
    })
    parts = write_month_parts(df, tmp_path, timeframe="5s", max_bytes=8_000)
    assert len(parts) >= 2
    assert [p.name for p in parts] == [f"features_5s_part{i}.parquet" for i in range(1, len(parts) + 1)]
    restored = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert len(restored) == n
    assert list(restored.columns[:1]) == ["bar_end_ts"]


def test_data_pipeline_tmp_dir_still_uses_day_files(tmp_path):
    a = pd.DataFrame(
        {"cs_close": [1.0, 2.0]},
        index=pd.to_datetime(
            ["2026-08-25 19:44:55+00:00", "2026-08-26 13:02:00+00:00"]
        ),
    )
    b = pd.DataFrame(
        {"cs_close": [9.0, 3.0]},
        index=pd.to_datetime(
            ["2026-08-26 13:02:00+00:00", "2026-08-26 13:05:00+00:00"]
        ),
    )
    a.to_parquet(tmp_path / "features_2026-08-25.parquet")
    b.to_parquet(tmp_path / "features_2026-08-26.parquet")
    df = DataPipeline(tmp_path).load_features("2026-08-25", "2026-08-26")
    assert df.index.is_unique
    ts = pd.Timestamp("2026-08-26 13:02:00", tz="UTC")
    assert float(df.loc[ts, "cs_close"]) == 9.0


def test_data_pipeline_hive_skips_out_of_range_parts(tmp_path, monkeypatch):
    """Ranged load must not open months outside [start, end] (corrupt-part isolation)."""
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    from src.common.data_paths import native_tf_dir

    jul = native_tf_dir("5s", year=2026, month=7)
    aug = native_tf_dir("5s", year=2026, month=8)
    jul.mkdir(parents=True)
    aug.mkdir(parents=True)
    ts = pd.date_range("2026-07-02 13:00:00", periods=3, freq="5s", tz="UTC")
    good = pd.DataFrame({"bar_end_ts": ts, "cs_close": [1.0, 2.0, 3.0]})
    good.to_parquet(jul / "features_5s_part1.parquet", index=False)
    (aug / "features_5s_part1.parquet").write_bytes(b"not-a-parquet")

    root = native_tf_dir("5s")
    df = DataPipeline(root).load_features("2026-07-01", "2026-07-31")
    assert len(df) == 3
    assert float(df["cs_close"].iloc[0]) == 1.0


def test_ensure_utc_index_bar_end_ts():
    df = pd.DataFrame({
        "bar_end_ts": pd.to_datetime(["2026-04-07 13:00:00+00:00"]),
        "cs_close": [1.0],
    })
    out = ensure_utc_index(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_persist_session_day_writes_under_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    from src.features.hive_5s import persist_session_day

    df = pd.DataFrame(
        {"cs_close": [1.0]},
        index=pd.to_datetime(["2026-04-07 13:00:00+00:00"]),
    )
    written = persist_session_day(df)
    assert written
    assert all(str(p).startswith(str(tmp_path)) for p in written)
    from src.features.hive_5s import load_session_day
    got = load_session_day("2026-04-07")
    assert len(got) == 1
    assert float(got["cs_close"].iloc[0]) == 1.0
