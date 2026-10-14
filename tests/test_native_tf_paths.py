"""Native-TF hive paths (ADR-070): timeframe=/symbol=/year=/month=."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from src.common.data_paths import (
    HIVE_TIMEFRAMES,
    NATIVE_TF_TIMEFRAMES,
    clip_hive_part_files,
    ensure_native_tf_dirs,
    iter_hive_part_files,
    native_tf_dir,
    native_tf_features_root,
    native_tf_parquet_path,
    native_tf_part_files,
    native_tf_part_name,
    native_tf_part_path,
)


def test_native_tf_dir_hive_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    root = native_tf_features_root()
    assert root == tmp_path / "sources" / "features"
    assert native_tf_dir("15s") == root / "timeframe=15s" / "symbol=NQ"
    got = native_tf_dir("1m", year=2025, month=1)
    assert got == root / "timeframe=1m" / "symbol=NQ" / "year=2025" / "month=01"
    assert native_tf_dir("5m", symbol="ES", year=2026, month=12).name == "month=12"


def test_native_tf_parquet_path_day_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    p = native_tf_parquet_path("1h", "2025-04-07")
    assert p.name == "features_2025-04-07.parquet"
    assert p.parent.name == "month=04"
    assert p.parent.parent.name == "year=2025"
    assert native_tf_parquet_path("1d", date(2026, 8, 25)).name == (
        "features_2026-08-25.parquet"
    )
    assert native_tf_parquet_path("15s", datetime(2026, 1, 2, 14, 0)).name == (
        "features_2026-01-02.parquet"
    )


def test_native_tf_dir_rejects_unknown_and_month_without_year():
    with pytest.raises(ValueError, match="unknown native timeframe"):
        native_tf_dir("10s")
    with pytest.raises(ValueError, match="month requires year"):
        native_tf_dir("15s", month=1)
    with pytest.raises(ValueError, match="month must be 1..12"):
        native_tf_dir("15s", year=2025, month=13)


def test_native_tf_dir_accepts_5s(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    p = native_tf_dir("5s", year=2026, month=8)
    assert p == (
        tmp_path / "sources" / "features" / "timeframe=5s" / "symbol=NQ"
        / "year=2026" / "month=08"
    )
    assert "5s" in HIVE_TIMEFRAMES
    assert "5s" not in NATIVE_TF_TIMEFRAMES


def test_ensure_native_tf_dirs_creates_month_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    created = ensure_native_tf_dirs(
        start_year=2025, start_month=11, end_year=2026, end_month=2,
    )
    # 5 TFs × 4 months (Nov, Dec, Jan, Feb)
    assert len(created) == 20
    assert all(p.is_dir() for p in created)
    assert (tmp_path / "sources" / "features" / "timeframe=15s" / "symbol=NQ"
            / "year=2025" / "month=11").is_dir()
    assert (tmp_path / "sources" / "features" / "timeframe=1d" / "symbol=NQ"
            / "year=2026" / "month=02").is_dir()
    assert not (tmp_path / "sources" / "features" / "timeframe=15s" / "symbol=NQ"
                / "year=2025" / "month=10").exists()
    assert not (tmp_path / "sources" / "features" / "timeframe=5s").exists()
    again = ensure_native_tf_dirs(
        start_year=2025, start_month=11, end_year=2026, end_month=2,
    )
    assert len(again) == 20
    assert NATIVE_TF_TIMEFRAMES == ("15s", "1m", "5m", "1h", "1d")
    assert HIVE_TIMEFRAMES[0] == "5s"


def test_native_tf_part_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    assert native_tf_part_name("15s", 1) == "features_15s_part1.parquet"
    assert native_tf_part_name("15s", 12) == "features_15s_part12.parquet"
    p = native_tf_part_path("15s", 2025, 4, 2)
    assert p.name == "features_15s_part2.parquet"
    assert p.parent.name == "month=04"
    assert p.parent.parent.name == "year=2025"
    month = native_tf_dir("15s", year=2025, month=4)
    month.mkdir(parents=True)
    (month / "features_15s_part2.parquet").write_bytes(b"x")
    (month / "features_15s_part1.parquet").write_bytes(b"y")
    (month / "features_2025-04-07.parquet").write_bytes(b"z")
    got = native_tf_part_files("15s", 2025, 4)
    assert [g.name for g in got] == [
        "features_15s_part1.parquet",
        "features_15s_part2.parquet",
    ]
    with pytest.raises(ValueError, match="part must be"):
        native_tf_part_name("15s", 0)


def test_iter_hive_part_files_clips_utc_month(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    a = native_tf_dir("5s", year=2025, month=4)
    b = native_tf_dir("5s", year=2025, month=5)
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "features_5s_part1.parquet").write_bytes(b"a")
    (b / "features_5s_part1.parquet").write_bytes(b"b")
    (b / "features_2025-05-01.parquet").write_bytes(b"skip")
    got = iter_hive_part_files("5s", start="2025-05-01", end="2025-05-31")
    assert [p.name for p in got] == ["features_5s_part1.parquet"]
    assert got[0].parent.name == "month=05"


def test_clip_hive_part_files_drops_out_of_range_months(tmp_path):
    a = tmp_path / "year=2026" / "month=07"
    b = tmp_path / "year=2026" / "month=08"
    c = tmp_path / "year=2026" / "month=09"
    for d in (a, b, c):
        d.mkdir(parents=True)
    fa = a / "features_5s_part1.parquet"
    fb = b / "features_5s_part1.parquet"
    fc = c / "features_5s_part1.parquet"
    for p in (fa, fb, fc):
        p.write_bytes(b"x")
    got = clip_hive_part_files([fa, fb, fc], "2026-06-01", "2026-07-31")
    assert got == [fa]
    got2 = clip_hive_part_files([fa, fb, fc], "2026-06-01", "2026-09-10")
    assert got2 == [fa, fb, fc]
