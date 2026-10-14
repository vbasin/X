"""Incremental feature extend: new 5s bars append without recomputing history."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.features.engine import (
    FeatureEngine,
    last_closed_bar_ts,
    merge_feature_rows,
    timestep_seconds,
)


def test_timestep_seconds():
    assert timestep_seconds("5s") == 5
    assert timestep_seconds("10s") == 10


def test_last_closed_bar_ts_left_labeled_5s():
    # Bar [13:00:00, 13:00:05) complete only after 13:00:05.
    t = pd.Timestamp("2026-08-14 13:00:07", tz="UTC")
    assert last_closed_bar_ts(t, timestep="5s") == pd.Timestamp(
        "2026-08-14 13:00:00", tz="UTC"
    )
    t = pd.Timestamp("2026-08-14 13:00:05", tz="UTC")
    assert last_closed_bar_ts(t, timestep="5s") == pd.Timestamp(
        "2026-08-14 13:00:00", tz="UTC"
    )
    t = pd.Timestamp("2026-08-14 13:00:04", tz="UTC")
    assert last_closed_bar_ts(t, timestep="5s") == pd.Timestamp(
        "2026-08-14 12:59:55", tz="UTC"
    )


def test_merge_feature_rows_appends_and_ignores_overlap():
    idx = pd.date_range("2026-08-14 13:00:00", periods=3, freq="5s", tz="UTC")
    existing = pd.DataFrame({"a": [1, 2, 3], "_close": [1.0, 2.0, 3.0]}, index=idx)
    new = pd.DataFrame(
        {"a": [4], "_close": [4.0]},
        index=pd.DatetimeIndex([idx[-1] + pd.Timedelta("5s")]),
    )
    out = merge_feature_rows(existing, new)
    assert len(out) == 4
    assert list(out["a"]) == [1, 2, 3, 4]

    # Overlapping index from new is dropped (history not recomputed).
    overlap = pd.DataFrame({"a": [99]}, index=idx[-1:])
    out2 = merge_feature_rows(existing, overlap)
    assert list(out2["a"]) == [1, 2, 3]


def test_extend_date_noop_when_uptodate(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")

    from src.features.hive_5s import load_session_day, persist_session_day

    eng = FeatureEngine()
    d = date(2026, 8, 14)
    last = last_closed_bar_ts(
        pd.Timestamp("2026-08-14 14:00:07", tz="UTC"), timestep="5s"
    )
    idx = pd.date_range(last - pd.Timedelta("15s"), periods=4, freq="5s", tz="UTC")
    df = pd.DataFrame({"_close": range(len(idx))}, index=idx)
    persist_session_day(df)

    called = {"build": 0, "compute": 0}

    def _build(_d):
        called["build"] += 1
        return Path("nope")

    def _compute(*_a, **_k):
        called["compute"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(eng, "build_date", _build)
    monkeypatch.setattr(eng, "_compute_feature_frame", _compute)

    out = eng.extend_date(d, asof=pd.Timestamp("2026-08-14 14:00:07", tz="UTC"))
    assert out.exists()
    assert called["build"] == 0
    assert called["compute"] == 0
    assert len(load_session_day(d)) == 4


def test_extend_date_appends_only_new_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")

    from src.features.hive_5s import load_session_day, persist_session_day

    eng = FeatureEngine()
    d = date(2026, 8, 14)
    asof = pd.Timestamp("2026-08-14 14:00:12", tz="UTC")
    last_closed = last_closed_bar_ts(asof, timestep="5s")  # 14:00:05
    existing_idx = pd.date_range(
        "2026-08-14 13:59:50", periods=3, freq="5s", tz="UTC"
    )
    existing = pd.DataFrame(
        {"rv_x": [1.0, 2.0, 3.0], "_close": [10.0, 11.0, 12.0]},
        index=existing_idx,
    )
    persist_session_day(existing)

    window_idx = pd.date_range(
        "2026-08-14 13:59:00", periods=20, freq="5s", tz="UTC"
    )
    computed = pd.DataFrame(
        {
            "rv_x": range(len(window_idx)),
            "_close": [float(i) for i in range(len(window_idx))],
        },
        index=window_idx,
    )

    def _compute(*_a, **_k):
        return computed

    monkeypatch.setattr(eng, "_compute_feature_frame", _compute)
    monkeypatch.setattr(
        eng, "build_date", lambda _d: (_ for _ in ()).throw(AssertionError("no full rebuild"))
    )

    out = eng.extend_date(d, asof=asof)
    assert out.exists()
    result = load_session_day(d)
    assert result.loc[existing_idx[0], "rv_x"] == 1.0
    assert result.loc[existing_idx[-1], "rv_x"] == 3.0
    assert last_closed in result.index
    assert result.loc[last_closed, "rv_x"] == computed.loc[last_closed, "rv_x"]
    assert len(result) == 3 + 1


def test_extend_date_bootstrap_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")

    eng = FeatureEngine()
    d = date(2026, 8, 14)
    called = {"n": 0}
    built = tmp_path / "built.parquet"

    def _build(_d):
        called["n"] += 1
        built.write_bytes(b"x")
        return built

    monkeypatch.setattr(eng, "build_date", _build)
    out = eng.extend_date(d, asof=pd.Timestamp("2026-08-14 14:00:00", tz="UTC"))
    assert out == built
    assert called["n"] == 1
