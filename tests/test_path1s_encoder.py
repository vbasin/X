"""1s path features, folds, and path_tcn plugin (ADR-053 Encoder A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.catalog import get_entry
from src.models.heads import encode_hazard_class, n_hazard_classes
from src.objectives.loader import load_objective
from src.search.controller import _load_plugin
from src.training.path_features import PATH_FEATURE_COLS, path_ohlc_features
from src.training.path_labels import CENSORED, STOP, TARGET


def _ohlcv(n: int, start: str = "2026-03-02 13:00:00+00:00") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    close = 20000 + np.cumsum(rng.normal(0, 0.25, size=n))
    high = close + rng.uniform(0.25, 1.0, size=n)
    low = close - rng.uniform(0.25, 1.0, size=n)
    opn = close + rng.normal(0, 0.1, size=n)
    vol = rng.integers(1, 50, size=n).astype("float64")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_path_features_warmup_fills_session_open():
    session = _ohlcv(120, start="2026-03-02 13:30:00+00:00")
    cold = path_ohlc_features(session)
    assert not np.isfinite(cold["log_atr"].iloc[0])
    prior = _ohlcv(ATR_WARM := 700, start="2026-03-02 13:00:00+00:00")
    warm = path_ohlc_features(session, warmup=prior)
    assert np.isfinite(warm["log_atr"].iloc[0])
    assert len(warm) == len(session)
    # A spike only in the warmup tail must move the session-open ATR.
    prior2 = prior.copy()
    prior2.iloc[-2, prior2.columns.get_loc("high")] = prior2.iloc[-2]["close"] + 40
    warm2 = path_ohlc_features(session, warmup=prior2)
    assert warm2["log_atr"].iloc[0] != warm["log_atr"].iloc[0]


def test_path_features_are_causal_and_named():
    raw = _ohlcv(900)
    f = path_ohlc_features(raw)
    assert list(f.columns) == list(PATH_FEATURE_COLS)
    # warmup is NaN; after ATR window the rest is finite
    tail = f.iloc[700:]
    assert np.isfinite(tail.to_numpy()).all()
    # ATR at t cannot use tr[t] (shift-1). Spike the last bar only — earlier log_atr unchanged.
    raw2 = raw.copy()
    raw2.iloc[-1, raw2.columns.get_loc("high")] = raw2.iloc[-1]["close"] + 50
    f2 = path_ohlc_features(raw2)
    np.testing.assert_allclose(f["log_atr"].iloc[-2], f2["log_atr"].iloc[-2], rtol=1e-9)


def test_objective_is_path1s_path_tcn_no_vix():
    obj = load_objective("nq-50pt-path-hazard")
    assert obj["search"]["input"] == "path1s"
    assert obj["eligible_model_families"] == ["path_tcn"]
    assert obj["eval"]["walk_forward"]["purge_seconds"] == 600
    assert obj["search"].get("entry_stride") == 5
    assert get_entry("path_tcn").params_default["seq_len"] == 1024
    assert _load_plugin("path_tcn").name == "path_tcn"


def _label_frame(ts: pd.DatetimeIndex, *, usable: bool = True) -> pd.DataFrame:
    n = len(ts)
    event_up = np.full(n, CENSORED, dtype=np.int8)
    bucket_up = np.full(n, 20, dtype=np.int16)
    event_up[::10] = TARGET
    bucket_up[::10] = 2
    return pd.DataFrame({
        "ts": ts, "label": np.where(event_up == TARGET, 2, 1).astype(np.int32),
        "event_up": event_up, "bucket_up": bucket_up,
        "event_down": np.full(n, STOP, dtype=np.int8),
        "bucket_down": np.zeros(n, dtype=np.int16),
        "offset_up": 80, "offset_down": 5,
        "mfe_up_points": 1.0, "mae_up_points": 0.5,
        "mfe_down_points": 0.5, "mae_down_points": 1.0,
        "target_points": 40.0, "stop_points": 5.0,
        "r_value": 80.0, "stop_ticks": 20.0,
        "usable": usable, "ambiguous": False, "truncated": False,
        "mfe_up_r": 0.2, "mae_up_r": 0.1, "mfe_down_r": 0.1, "mae_down_r": 0.2,
    })


def test_load_session_joins_labels(tmp_path, monkeypatch):
    from src.search import path1s_folds as pf

    path_dir = tmp_path / "p1s"
    lab_dir = tmp_path / "lab"
    path_dir.mkdir()
    lab_dir.mkdir()
    raw = _ohlcv(7200, start="2026-03-02 13:00:00+00:00")  # 13:00–15:00 UTC covers 08:30 CT labels
    raw.to_parquet(path_dir / "path_2026-03-02.parquet")
    # labels in 08:30 CT = 14:30 UTC
    lab_idx = pd.date_range("2026-03-02 14:30:00", periods=120, freq="1s", tz="UTC")
    _label_frame(lab_idx).to_parquet(lab_dir / "labels_2026-03-02.parquet", index=False)
    sess = pf.load_session(
        "2026-03-02", path_dir=path_dir, lab_dir=lab_dir,
        usable_only=True, entry_stride=5,
    )
    assert sess is not None
    assert sess.X.shape[1] == len(PATH_FEATURE_COLS)
    assert len(sess.entry_idx) == 120 // 5
    assert sess.y_up[0] == encode_hazard_class(TARGET, 2).item()


def test_purge_last_session_with_ndarray_mask():
    """pandas 2.x may return a ndarray from DatetimeIndex comparison — not a Series."""
    from src.search.path1s_folds import PathSession, _purge_last_session
    from src.training.path_features import PATH_FEATURE_COLS

    n_t, n_e = 2000, 400
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-06-01 14:30:00", periods=n_t, freq="1s", tz="UTC").to_numpy()
    # Spread entries across the session so a 600s tail-purge keeps some rows.
    entry_idx = np.linspace(0, n_t - 1, n_e, dtype=np.int64)
    sess = PathSession(
        date="2020-06-01",
        X=rng.normal(size=(n_t, 6)).astype("float32"),
        ts=ts, entry_idx=entry_idx,
        y=np.ones(n_e, dtype=np.int32),
        y_up=np.ones(n_e, dtype=np.int32),
        y_down=np.ones(n_e, dtype=np.int32),
        y_quantile=np.zeros((n_e, 4), dtype=np.float32),
        r_value=np.full(n_e, 80.0),
        feature_cols=PATH_FEATURE_COLS,
    )
    out = _purge_last_session([sess], 600)
    assert len(out) == 1
    assert 0 < len(out[0].entry_idx) < n_e


def test_path_tcn_fits_tiny_sessions():
    from tensorflow import keras
    from src.models.base import ModelSpec
    from src.search.path1s_folds import PathSession
    from src.training.path_features import PATH_FEATURE_COLS

    keras.backend.clear_session()
    rng = np.random.default_rng(0)
    n_t, n_e, f, seq_len = 80, 20, 6, 16
    n_cls = n_hazard_classes()
    X = rng.normal(size=(n_t, f)).astype("float32")
    entry_idx = np.arange(n_t - n_e, n_t)
    sess = PathSession(
        date="2026-03-02",
        X=X, ts=pd.date_range("2026-03-02 14:30:00", periods=n_t, freq="1s", tz="UTC").to_numpy(),
        entry_idx=entry_idx.astype(np.int64),
        y=rng.integers(0, 3, size=n_e, dtype=np.int32),
        y_up=rng.integers(0, n_cls, size=n_e, dtype=np.int32),
        y_down=rng.integers(0, n_cls, size=n_e, dtype=np.int32),
        y_quantile=rng.random(size=(n_e, 4)).astype("float32"),
        r_value=np.full(n_e, 80.0),
        feature_cols=PATH_FEATURE_COLS,
    )
    plugin = _load_plugin("path_tcn")
    plugin.build(ModelSpec(family="path_tcn", params={
        "seq_len": seq_len, "dilations": [1, 2], "filters": 8, "kernel_size": 2,
        "epochs": 1, "batch_size": 4, "include_quantile": False, "dropout": 0.0,
    }), {})
    fitted = plugin.fit(None, [{
        "sessions_train": [sess],
        "sessions_oof": [sess],
        "sessions_val": [],
    }])
    assert fitted.oof_preds is not None
    assert set(fitted.oof_preds["direction"]).issubset({"up", "down", "flat"})
    keras.backend.clear_session()
