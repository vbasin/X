"""Nightly feature-builder patch: missing file / missing rows / NaN cells."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.features.engine import (
    FeatureEngine,
    patch_feature_frames,
)


def test_patch_feature_frames_fills_nan_and_missing_rows_keeps_live():
    idx = pd.date_range("2026-08-14 13:00:00", periods=4, freq="5s", tz="UTC")
    existing = pd.DataFrame(
        {"rv_x": [1.0, np.nan, 3.0, np.nan], "cs_y": [0.1, 0.2, 0.3, 0.4]},
        index=idx,
    )
    extra = idx[-1] + pd.Timedelta("5s")
    built = pd.DataFrame(
        {
            "rv_x": [9.0, 2.0, 9.0, 4.0, 5.0],
            "cs_y": [9.0, 9.0, 9.0, 9.0, 0.5],
            "iv_z": [0.2, 0.2, 0.2, 0.2, 0.2],
        },
        index=idx.append(pd.DatetimeIndex([extra])),
    )
    out = patch_feature_frames(existing, built)
    assert len(out) == 5
    # finite live values kept
    assert out.loc[idx[0], "rv_x"] == 1.0
    assert out.loc[idx[2], "rv_x"] == 3.0
    assert out.loc[idx[0], "cs_y"] == 0.1
    # NaNs filled from built
    assert out.loc[idx[1], "rv_x"] == 2.0
    assert out.loc[idx[3], "rv_x"] == 4.0
    # new row + new column
    assert out.loc[extra, "rv_x"] == 5.0
    assert out.loc[extra, "iv_z"] == 0.2


def test_expected_bar_index_08_to_11(monkeypatch):
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")
    eng = FeatureEngine(timezone="America/Chicago", timestep="5s")
    idx = eng._expected_bar_index(date(2026, 8, 14), "08:00", "11:00")
    assert len(idx) == 2160
    assert idx[0] == pd.Timestamp("2026-08-14 13:00:00", tz="UTC")
    assert idx[-1] == pd.Timestamp("2026-08-14 15:59:55", tz="UTC")


def test_expected_bar_index_08_to_1445(monkeypatch):
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")
    eng = FeatureEngine(timezone="America/Chicago", timestep="5s")
    idx = eng._expected_bar_index(date(2026, 8, 14), "08:00", "14:45")
    assert idx[-1] == pd.Timestamp("2026-08-14 19:44:55", tz="UTC")
    assert len(idx) == 4860


def test_patch_date_skips_when_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")

    from src.features.hive_5s import load_session_day, persist_session_day

    eng = FeatureEngine(timezone="America/Chicago", timestep="5s")
    monkeypatch.setattr(eng, "_widest_patch_hours", lambda *_a, **_k: ("08:00", "11:00"))
    d = date(2026, 8, 14)
    idx = eng._expected_bar_index(d, "08:00", "11:00")
    df = pd.DataFrame({"rv_x": np.ones(len(idx))}, index=idx)
    persist_session_day(df)

    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("must not recompute a complete day")

    monkeypatch.setattr(eng, "_compute_feature_frame", _boom)
    monkeypatch.setattr(eng, "build_date", _boom)
    out = eng.patch_date(d)
    assert out.exists()
    assert called["n"] == 0
    assert len(load_session_day(d)) == 2160


def test_patch_date_fills_missing_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PGUSER", "x")
    monkeypatch.setenv("PGPASSWORD", "x")
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "x")
    monkeypatch.setattr("src.features.engine.get_dsn", lambda: "postgresql://x")

    from src.features.hive_5s import load_session_day, persist_session_day

    eng = FeatureEngine(timezone="America/Chicago", timestep="5s")
    monkeypatch.setattr(eng, "_widest_patch_hours", lambda *_a, **_k: ("08:00", "11:00"))
    d = date(2026, 8, 14)
    full = eng._expected_bar_index(d, "08:00", "11:00")
    morning = full[:100]
    existing = pd.DataFrame({"rv_x": np.ones(len(morning))}, index=morning)
    persist_session_day(existing)

    built = pd.DataFrame({"rv_x": np.full(len(full), 2.0)}, index=full)

    def _compute(*_a, **_k):
        return built

    monkeypatch.setattr(eng, "_compute_feature_frame", _compute)
    out = eng.patch_date(d)
    assert out.exists()
    result = load_session_day(d)
    assert len(result) == 2160
    assert float(result["rv_x"].iloc[0]) == 1.0
    assert float(result["rv_x"].iloc[-1]) == 2.0
